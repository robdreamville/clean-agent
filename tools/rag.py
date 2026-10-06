"""RAG pipeline with OpenTelemetry instrumentation — Lesson 5: Instrumented RAG QA.

Pipeline: embed -> search -> rerank (stub) -> assemble_context.
Each stage opens its own span; spans nest via ambient OTel context.

Standalone:  python -m tools.rag        -> `retrieval` is the root span in Phoenix.
In harness:  agent calls rag_search()  -> spans nest under the harness trace, zero code changes.

Docs: loaded from docs.txt (looks next to this file, then project root,
then $RAG_DOCS_PATH). Each doc needs an id — parsed from the `doc_id:` field
in the Metadata header, falling back to doc-001, doc-002, ...
"""

import math
import os
import re
import time
from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# Tracing setup
# ---------------------------------------------------------------------------
def init_tracing() -> None:
    """Idempotent tracer setup. Call once per process before retrieving."""
    provider = trace.get_tracer_provider()
    # Only install our provider if nobody else has (e.g. the harness later owns this).
    if isinstance(provider, trace.ProxyTracerProvider) or not hasattr(provider, "add_span_processor"):
        resource = Resource.create({"service.name": "clean-agent-rag"})
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
        return OllamaEmbeddings(model="nomic-embed-text", keep_alive="30m").embed_documents(texts)
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
                     max_tokens: int = MAX_CONTEXT_TOKENS) -> RetrievalResult:
    with tracer.start_as_current_span("rag.assemble_context") as span:
        kept: list[Candidate] = []
        truncated: list[str] = []
        running = 0
        for c in ranked:
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
    """Full pipeline. Opens the parent `retrieval` span; stages nest under it."""
    with tracer.start_as_current_span("retrieval") as span:
        span.set_attributes({"query.text": query, "pipeline.name": "rag"})
        qvec = embed_query(query)
        cands = search_store(qvec, _INDEX, _DOCS_BY_ID)
        ranked = rerank(cands)
        result = assemble_context(ranked)
        result.query = query
        span.set_attribute("retrieval.result_count", len(result.documents))
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
