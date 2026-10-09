"""RAG pipeline with OpenTelemetry instrumentation — Lesson 5: Instrumented RAG QA.

Pipeline: embed -> search -> gate1 -> rerank -> gate2 -> assemble_context.
Gate 1 is the cheap score tripwire (pre-rerank); gate 2 is the LLM
retrieval-sufficiency check (post-rerank, pre-generation). Each stage opens
its own span; spans nest via ambient OTel context.

Standalone:  python -m tools.rag        -> `retrieval` is the root span in Phoenix.
In harness:  agent calls rag_search()  -> spans nest under the harness trace, zero code changes.

Docs: loaded from docs.txt (looks next to this file, then project root,
then $RAG_DOCS_PATH). Each doc needs an id — parsed from the `doc_id:` field
in the Metadata header, falling back to doc-001, doc-002, ...
"""

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from google import genai

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL = "gemini-embedding-001" 
GEN_MODEL = "gemini-2.5-flash"          # used only for count_tokens
PHOENIX_ENDPOINT = "http://localhost:6006/v1/traces"
CAPTURE_CONTENT = True               # True = also store chunk text on spans (dev only)

TOP_K_SEARCH = 5
TOP_K_RERANK = 3
MAX_CONTEXT_TOKENS = 2000

# Gate 1 is a LOOSE pre-filter, not the final say. It stops only obvious junk
# so the expensive gate-2 LLM call isn't wasted. Gate 2 decides what passes.
# Relevance floor: top-1 absolute score. Below this, nothing is even in the
# ballpark — FAIL, skip gate 2.
# PROVISIONAL (picked blind, Oct 9): 0.55 was verified WRONG — it killed a
# good retrieval ("user's name" scored below it). 0.30 is a guess. Calibrate
# properly: run the batch, record top-1 scores with human good/bad labels,
# set the floor below the worst good retrieval. Do not ship a number you
# can't justify from data.
GATE1_RELEVANCE_FLOOR = 0.54
# Separation line: top-1 minus top-2 margin. Below this with relevance held,
# retrieval is ambiguous — WATCH.
GATE1_MARGIN_LINE = 0.10


# ---------------------------------------------------------------------------
# Tracing setup
# ---------------------------------------------------------------------------
def init_tracing() -> None:
    """Idempotent tracer setup. Call once per process before retrieving.

    Phoenix routes spans into projects via the ``openinference.project.name``
    resource attribute (``service.name`` is ignored for project routing).
    Override the project per run with the PHOENIX_PROJECT env var, e.g.:
        $env:PHOENIX_PROJECT = "lesson6-batch"
    """
    project = os.environ.get("PHOENIX_PROJECT", "clean-agent-rag")
    provider = trace.get_tracer_provider()
    # Only install our provider if nobody else has (e.g. the harness later owns this).
    if isinstance(provider, trace.ProxyTracerProvider) or not hasattr(provider, "add_span_processor"):
        resource = Resource.create(
            {
                "service.name": "clean-agent-rag",
                "openinference.project.name": project,
            }
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT))
        )
        try:
            trace.set_tracer_provider(provider)
        except Exception:
            pass  # another module already set it — spans still nest via context


tracer = trace.get_tracer("clean-agent.rag")
_genai_client: genai.Client | None = None


def _genai() -> genai.Client:
    """Lazy Gemini client (reads GEMINI_API_KEY / GOOGLE_API_KEY from env)."""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client()
    return _genai_client


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
@dataclass
class Document:
    id: str
    category: str
    text: str
    indexed_at: str = ""


@dataclass
class Candidate:
    doc: Document
    score: float


@dataclass
class RetrievalResult:
    query: str
    documents: list[Document]
    scores: list[float]
    context_text: str
    total_tokens: int
    truncated: list[str] = field(default_factory=list)


def load_documents(path: str | None = None) -> list[Document]:
    """Parse docs.txt into Documents. doc_id/category come from the Metadata header."""
    candidates = [
        path,
        os.environ.get("RAG_DOCS_PATH"),
        os.path.join(os.path.dirname(__file__), "docs.txt"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs.txt"),
        "docs.txt",
    ]
    src = next((p for p in candidates if p and os.path.exists(p)), None)
    if src is None:
        raise FileNotFoundError(
            "docs.txt not found. Put it next to tools/rag.py, at project root, "
            "or set $RAG_DOCS_PATH."
        )

    raw = open(src, encoding="utf-8").read()
    # Split on "Expanded Document N:" headers; first chunk is preamble (dropped).
    parts = re.split(r"Expanded Document \d+:", raw)
    docs: list[Document] = []
    for i, part in enumerate(parts[1:], start=1):
        doc_id = f"doc-{i:03d}"
        category = "uncategorized"
        m_id = re.search(r"doc_id:\s*([^\s|]+)", part)
        if m_id:
            doc_id = m_id.group(1).strip()
        m_cat = re.search(r"category:\s*([^\s|]+)", part)
        if m_cat:
            category = m_cat.group(1).strip()
        # Body = everything after the Metadata line; drop a leading "Content:" header.
        lines = part.splitlines()
        body_start = 0
        for j, line in enumerate(lines):
            if line.strip().lower().startswith("metadata:"):
                body_start = j + 1
                break
        body = "\n".join(lines[body_start:]).strip()
        body = re.sub(r"^\s*Content:\s*", "", body).strip()
        docs.append(Document(id=doc_id, category=category, text=body,
                            indexed_at=time.strftime("%Y-%m-%dT%H:%M:%S")))
    return docs


# ---------------------------------------------------------------------------
# Embeddings (cached: index-time vs query-time stay separate)
# ---------------------------------------------------------------------------
_embedding_cache: dict[str, list[float]] = {}


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if os.getenv("MODEL", "gemini") == "ollama":
        # offline: local embeddings (needs `ollama pull nomic-embed-text` once)
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text", keep_alive=1800).embed_documents(texts)  # seconds, not "30m"
    resp = _genai().models.embed_content(model=EMBED_MODEL, contents=texts)
    return [list(e.values) for e in resp.embeddings]


def _active_embed_model() -> str:
    """The embedding model actually in use (follows the MODEL switch)."""
    return "nomic-embed-text" if os.getenv("MODEL", "gemini") == "ollama" else EMBED_MODEL


def build_index(docs: list[Document]) -> dict[str, list[float]]:
    """Embed every doc once. Returns {doc_id: vector}."""
    uncached = [d for d in docs if d.id not in _embedding_cache]
    if uncached:
        vecs = _embed_texts([d.text for d in uncached])
        for d, v in zip(uncached, vecs):
            _embedding_cache[d.id] = v
    return {d.id: _embedding_cache[d.id] for d in docs}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _count_tokens(text: str) -> int:
    """Real token count via Gemini (free call, no quota impact)."""
    if os.getenv("MODEL", "gemini") == "ollama":
        return len(text) // 4  # rough offline estimate (~4 chars/token)
    return _genai().models.count_tokens(model=GEN_MODEL, contents=text).total_tokens


def init_rag() -> int:
    """Load docs, embed once, build the in-memory index. Call once at startup."""
    init_tracing()
    docs = load_documents()
    _DOCS_BY_ID.update({d.id: d for d in docs})
    _INDEX.update(build_index(docs))
    return len(docs)
# ---------------------------------------------------------------------------
# Pipeline stages — each opens its own span (the image's CAPTURE boxes)
# ---------------------------------------------------------------------------
def embed_query(query: str) -> list[float]:
    with tracer.start_as_current_span("rag.embed_query") as span:
        t0 = time.perf_counter()
        vec = _embed_texts([query])[0]
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        span.set_attributes({
            "embedding.model": _active_embed_model(),
            "query.length_chars": len(query),
            "embedding.latency_ms": latency_ms,
            "query.rewritten": False,
        })
        return vec


def search_store(qvec: list[float], index: dict[str, list[float]],
                 docs_by_id: dict[str, Document],
                 top_k: int = TOP_K_SEARCH) -> list[Candidate]:
    with tracer.start_as_current_span("rag.search_store") as span:
        t0 = time.perf_counter()
        scored = sorted(
            ((doc_id, _cosine(qvec, dvec)) for doc_id, dvec in index.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        cands = [Candidate(doc=docs_by_id[doc_id], score=s) for doc_id, s in scored[:top_k]]
        span.set_attributes({
            "retrieval.doc_ids": [c.doc.id for c in cands],
            "retrieval.scores": [round(c.score, 4) for c in cands],
            "retrieval.candidate_count": len(cands),
            "store.latency_ms": latency_ms,
            "store.index": "in-memory",
            "store.filters": "none",
        })
        if CAPTURE_CONTENT:
            span.set_attribute("retrieval.doc_texts", [c.doc.text[:500] for c in cands])
        return cands


def gate1_check(cands: list[Candidate]) -> str:
    """Gate 1: cheap advisory signal on the pre-rerank candidates.

    Own stage, own span (rag.gate1), sibling to the other pipeline stages.
    ADVISORY ONLY — it cannot stop anything. A wrong FAIL once blinded the
    agent by skipping gate 2 (the "user's name" false negative). Never again.
    Gate 2 always runs and has the final say.
    FAIL: scores look like junk — gate 2 runs extra strict.
    WATCH: material exists but ambiguous — gate 2 runs strict.
    PASS: strong signal — gate 2 runs normally.
    Returns the verdict; gate 2 reads it for strictness, retrieve() reads it
    for doc count.
    """
    with tracer.start_as_current_span("rag.gate1") as span:
        margin = round(cands[0].score - cands[1].score, 4) if len(cands) > 1 else 0.0
        top1 = round(cands[0].score, 4)
        if top1 < GATE1_RELEVANCE_FLOOR:
            verdict = "FAIL"
        elif len(cands) == 1 or margin >= GATE1_MARGIN_LINE:
            verdict = "PASS"
        else:
            verdict = "WATCH"
        span.set_attributes({
            "retrieval.top1_score": top1,
            "retrieval.margin": margin,
            "eval.gate1_verdict": verdict,
            "eval.gate1_ambiguous": verdict == "WATCH",
        })
        return verdict


# ---------------------------------------------------------------------------
# Gate 2: LLM retrieval-sufficiency check (pre-generation pipeline stage).
# Lives here, not in evals/: it runs inside every retrieval, before the
# agent generates. One question, yes or no: can this question be answered
# from these chunks?
# ---------------------------------------------------------------------------
GATE2_GEMINI_MODEL = "gemini-2.5-flash"  # primary: free tier, fast, native JSON mode
GATE2_OLLAMA_MODEL = "gemma4:e2b"        # offline fallback
GATE2_NUM_PREDICT = 512
GATE2_NUM_CTX = 4096
GATE2_CHUNK_TRUNC = 500
GATE2_MAX_CHUNKS = 5


class Gate2Input(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    chunks: list[str] = Field(min_length=1, max_length=GATE2_MAX_CHUNKS)
    gate1_verdict: Literal["FAIL", "WATCH", "PASS"]


class SufficiencyCheck(BaseModel):
    """Raw model output: one yes/no plus evidence (2 short sentences max)."""

    sufficient: bool
    evidence: str = Field(min_length=1, max_length=300)


class Gate2Verdict(BaseModel):
    verdict: Literal["PASS", "FLAG"]
    category: Optional[Literal["insufficient_retrieval", "gate2_error"]] = None
    evidence: str = ""
    gate1_verdict: Literal["FAIL", "WATCH", "PASS"]


def _parse_sufficiency(raw: str) -> SufficiencyCheck:
    """Parse gate-2 output. Full JSON first; salvage the boolean from
    truncated output (e.g. '{"sufficient": true,' cut off at the token cap).
    Only raises if no boolean is recoverable at all."""
    try:
        return SufficiencyCheck.model_validate(json.loads(raw))
    except Exception:
        pass
    m = re.search(r'"sufficient"\s*:\s*(true|false)', raw, re.IGNORECASE)
    if m:
        return SufficiencyCheck(
            sufficient=m.group(1).lower() == "true",
            evidence="recovered from partial output",
        )
    raise ValueError(f"unparseable gate 2 output: {raw[:200]}")


def _gate2_model_primary():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=GATE2_GEMINI_MODEL,
        response_mime_type="application/json",
        max_output_tokens=GATE2_NUM_PREDICT,
    )


def _gate2_model_fallback():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=GATE2_OLLAMA_MODEL,
        format="json",  # constrain to valid JSON; the small model needs the guardrail
        keep_alive="30m",
        num_ctx=GATE2_NUM_CTX,
        num_predict=GATE2_NUM_PREDICT,
    )


def _invoke_traced(model, model_name: str, prompt: str) -> tuple[str, str]:
    """Invoke the gate-2 model inside its own LLM span, nested under rag.gate2."""
    with tracer.start_as_current_span("rag.gate2.llm") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", model_name)
        t0 = time.perf_counter()
        resp = model.invoke(prompt)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        usage = getattr(resp, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        span.set_attributes({
            "llm.latency_ms": latency_ms,
            "llm.token_count.prompt": prompt_tokens,
            "llm.token_count.completion": completion_tokens,
            "llm.token_count.total": prompt_tokens + completion_tokens,
        })
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        if CAPTURE_CONTENT:
            span.set_attribute("llm.input_messages", prompt[:2000])
            span.set_attribute("llm.output_messages", text[:2000])
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return (m.group(0) if m else text), model_name


def _ask_gate2(prompt: str) -> tuple[str, str]:
    """Try Gemini Flash first; fall back to Ollama offline. Returns (json, model)."""
    for make, name in ((_gate2_model_primary, GATE2_GEMINI_MODEL),
                       (_gate2_model_fallback, GATE2_OLLAMA_MODEL)):
        try:
            return _invoke_traced(make(), name, prompt)
        except Exception:
            continue
    raise RuntimeError("gate 2: all models failed")


def _numbered(chunks: list[str]) -> str:
    return "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks))


def gate2_check(
    question: str,
    ranked: list[Candidate],
    gate1_verdict: Literal["FAIL", "WATCH", "PASS"],
) -> Gate2Verdict:
    """Gate 2: can this question be answered from these chunks?

    Runs post-rerank, pre-generation, nested under the retrieval span.
    WATCH from gate 1 runs strict: ambiguous retrieval gets no benefit
    of the doubt. Fail-closed: a broken check flags, never passes.
    """
    chunks = [c.doc.text[:GATE2_CHUNK_TRUNC] for c in ranked[:GATE2_MAX_CHUNKS]]
    # Strict when gate 1 is skeptical (WATCH or FAIL): ambiguous or
    # junk-looking retrieval gets no benefit of the doubt.
    strict = gate1_verdict in ("WATCH", "FAIL")
    with tracer.start_as_current_span("rag.gate2") as span:
        span.set_attribute("openinference.span.kind", "EVALUATOR")
        span.set_attribute("eval.gate1_verdict_in", gate1_verdict)

        if not chunks:
            # Nothing retrieved: trivially insufficient, no model call needed.
            verdict = Gate2Verdict(
                verdict="FLAG",
                category="insufficient_retrieval",
                evidence="no chunks retrieved",
                gate1_verdict=gate1_verdict,
            )
            span.set_attributes({
                "eval.gate2.model": "none",
                "eval.gate2_verdict": verdict.verdict,
                "eval.gate2_category": verdict.category,
                "eval.gate2_evidence": verdict.evidence,
                "eval.gate2_strict": strict,
            })
            return verdict

        judged = Gate2Input(question=question[:1000], chunks=chunks,
                            gate1_verdict=gate1_verdict)
        strict_note = (
            "Retrieval was ambiguous. When in doubt, answer false."
            if strict else ""
        )
        prompt = (
            "You are a retrieval sufficiency check. Answer one question: "
            "can the user's question be answered from these document chunks?\n\n"
            f"Question: {judged.question}\n\n"
            f"Chunks:\n{_numbered(judged.chunks)}\n\n"
            f"{strict_note}\n"
            "Reply with JSON only, exactly this shape. "
            "Evidence is 2 short sentences max — keep it brief, the token "
            "budget is tight.\n"
            '{"sufficient": true, "evidence": "two short sentences naming what supports it"}\n'
            '{"sufficient": false, "evidence": "two short sentences saying what is missing"}'
        )

        model_used = GATE2_GEMINI_MODEL
        check = None
        last_error = None
        attempts = 0
        # Retry once on transient failure (e.g. truncated JSON). A judge that
        # fails on first try often succeeds on the second; only after both
        # fail do we flag gate2_error.
        for _ in range(2):
            attempts += 1
            try:
                raw, model_used = _ask_gate2(prompt)
                check = _parse_sufficiency(raw)
                break
            except Exception as e:
                last_error = e
                continue
        if check is None:
            # Fail closed: a broken check flags, never passes silently.
            verdict = Gate2Verdict(
                verdict="FLAG",
                category="gate2_error",
                evidence=f"gate 2 failed after retry: {type(last_error).__name__}",
                gate1_verdict=gate1_verdict,
            )
        else:
            verdict = Gate2Verdict(
                verdict="PASS" if check.sufficient else "FLAG",
                category=None if check.sufficient else "insufficient_retrieval",
                evidence=check.evidence,
                gate1_verdict=gate1_verdict,
            )

        span.set_attributes({
            "eval.gate2.model": model_used,
            "eval.gate2_verdict": verdict.verdict,
            "eval.gate2_category": verdict.category or "none",
            "eval.gate2_evidence": verdict.evidence[:500],
            "eval.gate2_strict": strict,
            "eval.gate2_attempts": attempts,
        })
        return verdict


def rerank(candidates: list[Candidate], top_k: int = TOP_K_RERANK) -> list[Candidate]:
    """STUB: top-k cutoff, no re-scoring. Span still captures pre/post order."""
    with tracer.start_as_current_span("rag.rerank") as span:
        pre_order = [c.doc.id for c in candidates]
        ranked = candidates[:top_k]
        post_order = [c.doc.id for c in ranked]
        span.set_attributes({
            "reranker.model": "stub-top-k",
            "rerank.pre_order": pre_order,
            "rerank.post_order": post_order,
            "rerank.score_deltas": [0.0] * len(ranked),
            "rerank.survived_count": len(ranked),
        })
        return ranked


def assemble_context(ranked: list[Candidate],
                     max_tokens: int = MAX_CONTEXT_TOKENS,
                     keep: int = 3) -> RetrievalResult:
    with tracer.start_as_current_span("rag.assemble_context") as span:
        # How many docs the agent sees is decided by retrieve(), from the
        # gate verdicts. Gate 1 filters junk; gate 2 has the final say.
        shaped = ranked[:keep] if keep > 0 else []
        kept: list[Candidate] = []
        truncated: list[str] = []
        running = 0
        for c in shaped:
            t = _count_tokens(c.doc.text)
            if running + t > max_tokens:
                truncated.append(c.doc.id)
                continue
            running += t
            kept.append(c)

        context_text = "\n\n---\n\n".join(
            f"[doc_id={c.doc.id} | category={c.doc.category}]\n{c.doc.text}"
            for c in kept
        )
        total_tokens = _count_tokens(context_text) if context_text else 0
        span.set_attributes({
            "context.doc_ids": [c.doc.id for c in kept],
            "context.doc_freshness": ",".join(
                f"{c.doc.id}@{c.doc.indexed_at or 'unknown'}" for c in kept
            ),
            "context.total_tokens": total_tokens,
            "context.truncated": truncated,
            "context.max_tokens": max_tokens,
            "context.shaped_keep": keep,
        })
        return RetrievalResult(
            query="",  # filled by retrieve()
            documents=[c.doc for c in kept],
            scores=[c.score for c in kept],
            context_text=context_text,
            total_tokens=total_tokens,
            truncated=truncated,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
_INDEX: dict[str, list[float]] = {}
_DOCS_BY_ID: dict[str, Document] = {}


def retrieve(query: str) -> RetrievalResult:
    """Full pipeline. Opens the parent `retrieval` span; stages nest under it.

    Stage order: embed -> search -> gate1 -> rerank -> gate2 -> assemble.
    Gate 1 is advisory: a cheap signal tuning gate 2's strictness. It cannot
    blind the agent. Gate 2 ALWAYS runs — even on gate-1 FAIL — and has the
    final say on what the agent sees. That's how it catches gate 1's
    mistakes (the "user's name" false negative: gate 1 said FAIL, gate 2
    correctly said the chunks held the answer).
    """
    with tracer.start_as_current_span("retrieval") as span:
        span.set_attributes({"query.text": query, "pipeline.name": "rag"})
        span.set_attribute("openinference.span.kind", "RETRIEVER")
        qvec = embed_query(query)
        cands = search_store(qvec, _INDEX, _DOCS_BY_ID)
        gate1_verdict = gate1_check(cands)
        ranked = rerank(cands)
        # Always run gate 2. Skipping it on FAIL is what caused the false
        # negative — gate 2 never saw the chunks that held the answer.
        gate2_verdict = gate2_check(query, ranked, gate1_verdict)
        # Shaping: gate 2 has final say, EXCEPT when it errored. A broken
        # judge is our infrastructure flakiness, not the retrieval's fault —
        # so on gate2_error we defer to gate 1 rather than blinding the
        # agent. The FLAG/gate2_error verdict stays honest in the trace.
        if gate2_verdict.category == "insufficient_retrieval":
            keep = 0
        elif gate1_verdict == "PASS":
            keep = 2
        else:  # WATCH/FAIL, or gate2_error
            keep = 3
        result = assemble_context(ranked, keep=keep)
        result.query = query
        span.set_attribute("retrieval.result_count", len(result.documents))
        span.set_attribute("eval.gate1_verdict", gate1_verdict)
        span.set_attribute("eval.gate2_verdict", gate2_verdict.verdict)
        return result


# ---------------------------------------------------------------------------
# LangChain tool wrapper (wire into ALL_TOOLS when ready — not tonight)
# ---------------------------------------------------------------------------
def _rag_tool_fn(query: str) -> str:
    result = retrieve(query)
    if not result.documents:
        return "No relevant documents found."
    return result.context_text


try:
    from langchain_core.tools import tool as _lc_tool

    rag_search = _lc_tool(
        "rag_search",
        description=(
            "Search Roberto's personal documents and notes by semantic similarity. "
            "Always pass the user's FULL question as the query text, not keywords or names. "
            "Returns the top matching documents with IDs, categories, and similarity scores."
        ),
    )(_rag_tool_fn)
except ImportError:  # langchain not installed (standalone testing)
    rag_search = None


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_tracing()
    docs = load_documents()
    _DOCS_BY_ID.update({d.id: d for d in docs})
    _INDEX.update(build_index(docs))
    print(f"Indexed {len(docs)} docs: {[d.id for d in docs]}")

    test_queries = [
        "Where does Roberto currently work?",
        #"Which projects use vector databases?",
        #"What was done about the water heater?",
        #"What is Roberto's favorite food?",
    ]
    for q in test_queries:
        print(f"\n=== QUERY: {q}")
        r = retrieve(q)
        for d, s in zip(r.documents, r.scores):
            print(f"  [{d.id} | {d.category}] score={s:.4f}")
        print(f"  tokens={r.total_tokens} truncated={r.truncated}")

    # Flush spans so they land in Phoenix before exit.
    trace.get_tracer_provider().force_flush()
    print("\nDone. Check Phoenix at http://localhost:6006")
