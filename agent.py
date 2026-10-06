import os

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


def pick_model() -> str:
    """Runtime model switch: Gemini (needs Wi-Fi) or Ollama (offline).

    Set MODEL=ollama|gemini to skip the prompt. The choice is exported so
    tools/rag.py follows it automatically (embeddings switch too).
    """
    choice = os.getenv("MODEL", "").strip().lower()
    if choice not in ("gemini", "ollama"):
        print("Select model: [1] Gemini (needs Wi-Fi)  [2] Ollama (offline)")
        choice = "ollama" if input("> ").strip() == "2" else "gemini"
    os.environ["MODEL"] = choice
    return choice


USE = pick_model()

if USE == "ollama":
    from langchain_ollama import ChatOllama
    MODEL = ChatOllama(
        model="gemma4:e2b",
        temperature=0,      # deterministic: cleaner tool calls
        keep_alive="30m",   # keep loaded between runs, skip the reload wait
        num_ctx=8192,       # room for long tool histories (default 2048 chops them)
        num_predict=1024,   # cap output length: faster, less rambling
    )
else:
    from langchain_google_genai import ChatGoogleGenerativeAI
    MODEL = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,           # deterministic: cleaner tool calls
        max_output_tokens=1024,  # cap output length: less rambling, cheaper
        # no keep_alive needed (API side), no num_ctx needed (1M context)
    )
print(f"using model: {MODEL}")

from tools import ALL_TOOLS
from middleware import MIDDLEWARE

from tools import init_rag

print(f"rag index ready: {init_rag()} docs")

agent = create_agent(
    model=MODEL,
    tools=ALL_TOOLS,
    system_prompt="You are a helpful assistant. Use tools when they help. Be concise.",
    middleware=MIDDLEWARE,
    checkpointer=InMemorySaver(),
)

# add to the bottom of agent.py, run once
png = agent.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png)
