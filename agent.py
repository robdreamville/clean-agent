from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from tools import ALL_TOOLS
from middleware import MIDDLEWARE

agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=ALL_TOOLS,
    system_prompt="You are a helpful assistant. Use tools when they help. Be concise.",
    middleware=MIDDLEWARE,
    checkpointer=InMemorySaver(),
)

# add to the bottom of agent.py, run once
png = agent.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png)
