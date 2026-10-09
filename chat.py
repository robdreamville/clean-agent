# chat.py — talk to the agent like a chatbot
from langgraph.types import Command
from opentelemetry import trace as otel_trace
from agent import agent
from langchain_core.runnables import RunnableConfig
from evals.gate2_judge import judge_gate2
from tools.rag import LAST_RETRIEVAL

config: RunnableConfig = {"configurable": {"thread_id": "chat"}}


config = {"configurable": {"thread_id": "chat"}}


def get_text(msg):
    content = msg.content
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif isinstance(b, str):
            parts.append(b)
    return "".join(parts).strip()


print("chat with the agent (quit to exit)")
while True:
    text = input("you: ").strip()
    if text.lower() in ("quit", "exit"):
        break

    # Fresh turn: drop last turn's retrieval so the judge never scores
    # this answer against stale chunks when the agent doesn't retrieve.
    LAST_RETRIEVAL["chunks"] = []
    LAST_RETRIEVAL["gate1_verdict"] = "PASS"

    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config=config,
    )

    while "__interrupt__" in result:
        req = result["__interrupt__"][0].value
        print("\n--- approval needed ---")
        print(req)
        answer = input("approve or reject? ").strip().lower()
        decision = "approve" if answer.startswith("a") else "reject"
        hitl_tracer = otel_trace.get_tracer("clean-agent.hitl")
        with hitl_tracer.start_as_current_span("hitl.decision") as hspan:
            hspan.set_attribute("hitl.decision", decision)
            hspan.set_attribute("openinference.span.kind", "CHAIN")
            result = agent.invoke(
                Command(resume={"decisions": [{"type": decision}]}),
                config=config,
            )

    answer_text = get_text(result["messages"][-1])
    print("agent:", answer_text, "\n")

    # Gate 2: judge the final answer against what retrieval actually saw.
    # Only runs when this turn retrieved; LAST_RETRIEVAL resets every turn
    # so a stale previous turn never gets judged against.
    if LAST_RETRIEVAL["chunks"]:
        v = judge_gate2(
            question=text,
            answer=answer_text,
            chunks=LAST_RETRIEVAL["chunks"],
            gate1_verdict=LAST_RETRIEVAL["gate1_verdict"],
        )
        tag = f"[gate2: {v.verdict}" + (f" ({v.category})" if v.category else "") + "]"
        print(tag)
        if v.failed_claims:
            for fc in v.failed_claims:
                print(f"  - {fc}")
        print()
