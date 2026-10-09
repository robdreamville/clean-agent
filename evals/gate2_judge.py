"""Gate 2 judge: groundedness check.

Every claim in the answer must trace to a retrieved chunk. Catches the
invented-policy failure: the model stating what the corpus never said.

Design decisions (Oct 8, Roberto's calls):
- Judge ALWAYS runs on the local Ollama model (gemma4:e2b). Never Gemini.
  $0 per run offline. Groundedness is a narrow mechanical check; it does not
  need a frontier model.
- Input context is limited to exactly what the judge needs: the question,
  the answer, and the retrieved chunks. Nothing else. Chunks are truncated.
- Pydantic regulates the input and output schema. The judge returns JSON;
  parse failures fail closed (FLAG) so an unjudged answer never passes silently.
- Abstentions ARE judged. A correct refusal passes; a false refusal (chunks
  contained the answer) flags as false_abstention.
- Tracing: span "evals.gate2_judge", kind CHAIN, per the Oct 7 span-kind
  convention (hitl.decision=CHAIN, retrieval=RETRIEVER).
- Gate 1 feeds in: a WATCH verdict means ambiguous retrieval, so the judge
  runs strict. A stub for the tenth category (cross-run consistency) lives
  at the bottom, built later.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from opentelemetry import trace
from pydantic import BaseModel, Field, ValidationError

from tools.rag import init_tracing

tracer = trace.get_tracer("clean-agent.evals")

# ---------------------------------------------------------------------------
# Config: kept small on purpose. The judge sees only what it needs.
# ---------------------------------------------------------------------------
JUDGE_MODEL = "gemma4:e2b"          # always Ollama, never Gemini
JUDGE_NUM_PREDICT = 512             # cap judge output: verdict JSON, no rambling
JUDGE_NUM_CTX = 4096                # small context window: question + answer + chunks
CHUNK_TRUNC_CHARS = 500             # per-chunk cap, matches CAPTURE_CONTENT convention
MAX_CHUNKS = 5                      # more chunks than this never helped a verdict

# Cheap deterministic pre-check for refusals. The model then verifies correctness.
ABSTAIN_PATTERNS = [
    "don't have", "do not have", "no information", "cannot answer",
    "can't answer", "unable to answer", "doesn't say", "does not say",
    "not in the documents", "i don't know",
]


# ---------------------------------------------------------------------------
# Pydantic schemas: input and output are regulated, not free text.
# ---------------------------------------------------------------------------
class JudgeInput(BaseModel):
    question: str
    answer: str
    chunks: list[str] = Field(max_length=MAX_CHUNKS)
    gate1_verdict: Literal["FAIL", "WATCH", "PASS"]


class ClaimCheck(BaseModel):
    claim: str
    supporting_chunk: int | None = Field(
        default=None, description="0-based chunk index, or null if unsupported"
    )


class GroundednessResult(BaseModel):
    claims: list[ClaimCheck]
    verdict: Literal["PASS", "FLAG"]


class AbstentionCheck(BaseModel):
    answerable_from_chunks: bool
    evidence: str = Field(max_length=280)


class Gate2Verdict(BaseModel):
    verdict: Literal["PASS", "FLAG"]
    category: str | None = Field(
        default=None,
        description="invented_policy | false_abstention | judge_error | None when PASS",
    )
    failed_claims: list[str] = Field(default_factory=list)
    is_abstention: bool = False
    gate1_verdict: str


# ---------------------------------------------------------------------------
# Judge model: always Ollama, always small context, always capped output.
# ---------------------------------------------------------------------------
def _judge_model():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=JUDGE_MODEL,
        keep_alive="30m",
        num_ctx=JUDGE_NUM_CTX,
        num_predict=JUDGE_NUM_PREDICT,
    )


def _numbered_chunks(chunks: list[str]) -> str:
    return "\n".join(
        f"[{i}] {c[:CHUNK_TRUNC_CHARS]}" for i, c in enumerate(chunks[:MAX_CHUNKS])
    )


def _looks_like_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(p in lowered for p in ABSTAIN_PATTERNS)


def _ask_judge(prompt: str) -> str:
    model = _judge_model()
    resp = model.invoke(prompt)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    # Tolerate code fences; the schema is what matters.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def judge_gate2(
    question: str,
    answer: str,
    chunks: list[str],
    gate1_verdict: Literal["FAIL", "WATCH", "PASS"] = "PASS",
) -> Gate2Verdict:
    """Run gate 2 over one answer. Returns a regulated verdict.

    FAIL from gate 1 should never reach here (caller abstains before
    generating). WATCH tightens the judge: ambiguous retrieval gets
    less benefit of the doubt on borderline claims.
    """
    init_tracing()
    strict = gate1_verdict == "WATCH"
    with tracer.start_as_current_span("evals.gate2_judge") as span:
        span.set_attribute("openinference.span.kind", "JUDGE")
        span.set_attribute("eval.gate1_verdict_in", gate1_verdict)
        span.set_attribute("eval.gate2_judge.model", JUDGE_MODEL)

        judged = JudgeInput(
            question=question[:1000],
            answer=answer[:4000],
            chunks=[c[:CHUNK_TRUNC_CHARS] for c in chunks[:MAX_CHUNKS]],
            gate1_verdict=gate1_verdict,
        )

        try:
            if _looks_like_abstention(judged.answer):
                verdict = _judge_abstention(judged)
            else:
                verdict = _judge_groundedness(judged, strict=strict)
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            # Fail closed: an unjudged answer never passes silently.
            verdict = Gate2Verdict(
                verdict="FLAG",
                category="judge_error",
                failed_claims=[f"judge output failed validation: {type(e).__name__}"],
                is_abstention=False,
                gate1_verdict=gate1_verdict,
            )

        span.set_attributes(
            {
                "eval.gate2_verdict": verdict.verdict,
                "eval.gate2_category": verdict.category or "none",
                "eval.gate2_failed_claims": len(verdict.failed_claims),
                "eval.gate2_is_abstention": verdict.is_abstention,
            }
        )
        return verdict


def _judge_abstention(judged: JudgeInput) -> Gate2Verdict:
    """A refusal is only correct when the chunks truly lack the answer."""
    prompt = (
        "You are a groundedness judge. The assistant REFUSED to answer.\n"
        f"QUESTION: {judged.question}\n"
        f"CHUNKS:\n{_numbered_chunks(judged.chunks)}\n"
        "Could the question be answered from these chunks? "
        'Reply with JSON only: {"answerable_from_chunks": true/false, '
        '"evidence": "one sentence"}.'
    )
    raw = _ask_judge(prompt)
    check = AbstentionCheck.model_validate(json.loads(raw))
    if check.answerable_from_chunks:
        return Gate2Verdict(
            verdict="FLAG",
            category="false_abstention",
            failed_claims=[f"refused but chunks held the answer: {check.evidence}"],
            is_abstention=True,
            gate1_verdict=judged.gate1_verdict,
        )
    return Gate2Verdict(
        verdict="PASS",
        category=None,
        failed_claims=[],
        is_abstention=True,
        gate1_verdict=judged.gate1_verdict,
    )


def _judge_groundedness(judged: JudgeInput, strict: bool) -> Gate2Verdict:
    strict_note = (
        "Be strict: retrieval was ambiguous, so borderline claims count as unsupported."
        if strict
        else "Judge normally."
    )
    prompt = (
        "You are a groundedness judge. List each factual claim in the ANSWER. "
        "For each claim, name the 0-based chunk number that supports it, or null "
        "if no chunk supports it.\n"
        f"{strict_note}\n"
        f"QUESTION: {judged.question}\n"
        f"ANSWER: {judged.answer}\n"
        f"CHUNKS:\n{_numbered_chunks(judged.chunks)}\n"
        'Reply with JSON only: {"claims": [{"claim": "...", '
        '"supporting_chunk": 0 or null}], "verdict": "PASS" or "FLAG"}. '
        "Verdict is FLAG if any claim has null support."
    )
    raw = _ask_judge(prompt)
    result = GroundednessResult.model_validate(json.loads(raw))
    failed = [c.claim for c in result.claims if c.supporting_chunk is None]
    computed = "FLAG" if failed else "PASS"
    return Gate2Verdict(
        verdict=computed,
        category="invented_policy" if failed else None,
        failed_claims=failed,
        is_abstention=False,
        gate1_verdict=judged.gate1_verdict,
    )


# ---------------------------------------------------------------------------
# Tenth category stub: cross-run consistency. Built later (Lessons 12-14).
# Same question, N runs, identical retrieval margins, divergent answers
# means generation judgment is unstable. Not tonight's build.
# ---------------------------------------------------------------------------
def check_consistency(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    raise NotImplementedError("tenth category: cross-run consistency check (later)")


if __name__ == "__main__":
    import sys

    print("evals.gate2_judge loaded. Import judge_gate2() to run the judge.")
    print(f"judge model: {JUDGE_MODEL} (always Ollama, never Gemini)")
    sys.exit(0)
