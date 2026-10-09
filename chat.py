# chat.py — talk to the agent like a chatbot
from langgraph.types import Command
from opentelemetry import trace as otel_trace
from agent import agent
from langchain_core.runnables import RunnableConfig

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


    print("agent:", get_text(result["messages"][-1]), "\n")
