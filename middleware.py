from langchain.agents.middleware import HumanInTheLoopMiddleware

MIDDLEWARE = [
    HumanInTheLoopMiddleware(interrupt_on={"save_note": True}),
]
