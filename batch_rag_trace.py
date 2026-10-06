# batch_rag_trace.py — Lesson 6 eval batch: 6 questions x 5 runs, fresh thread per run.
#
# Why fresh threads: chat.py pins one thread_id, so session memory leaks
# between questions and later runs parrot earlier ones without retrieving.
# A new thread_id per run keeps every trace an honest, isolated measurement.
#
# Usage:  python batch_rag_trace.py   (picks model once, then runs unattended)
# Traces land in Phoenix automatically via the existing OTel setup.
# After the batch: open Phoenix, one sentence per trace (first thing wrong
# or "clean"), cluster into failure categories, count, find one silent failure.
import time

from langgraph.types import Command

from agent import agent  # noqa: E402  (prompts for model, builds RAG index on import)

# Fill in questions 3-6 before running. Spread: answerable / unanswerable / near-miss.
QUESTIONS = [
    "Where does Rob work?",            # answerable
    "Who is Rob's manager at FedEx?",  # near-miss: close docs, fact absent
    "FILL ME: second answerable question",
    "FILL ME: second unanswerable question",
    "FILL ME: second near-miss question",
    "FILL ME: sixth question",
]
RUNS_PER_QUESTION = 5

if any(q.startswith("FILL ME") for q in QUESTIONS):
    raise SystemExit("Fill in all 6 questions at the top of batch_rag_trace.py first.")

for qi, q in enumerate(QUESTIONS, 1):
    for r in range(1, RUNS_PER_QUESTION + 1):
        tid = f"batch-q{qi}-r{r}-{int(time.time())}"
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config={"configurable": {"thread_id": tid}},
        )
        # Defensive: auto-approve HITL interrupts so the batch never hangs
        # (middleware only interrupts on save_note; RAG runs shouldn't hit it).
        while "__interrupt__" in result:
            print(f"  [q{qi} r{r}] auto-approving interrupt")
            result = agent.invoke(
                Command(resume={"decisions": [{"type": "approve"}]}),
                config={"configurable": {"thread_id": tid}},
            )
        print(f"done q{qi} run {r}/{RUNS_PER_QUESTION} (thread {tid})")

print("batch complete — 30 traces in Phoenix. One sentence each, then cluster + count.")
