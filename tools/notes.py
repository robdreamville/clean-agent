from langchain_core.tools import tool


@tool
def save_note(note: str) -> str:
    """Append one line to the agent's state file (state.md)."""
    with open("state.md", "a") as f:
        f.write(note + "\n")
    return "saved to state.md"


@tool
def read_notes() -> str:
    """Read everything currently in the agent's state file."""
    try:
        with open("state.md") as f:
            return f.read()
    except FileNotFoundError:
        return "state.md is empty."
