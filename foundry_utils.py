# foundry_utils.py
import time
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import Agent, AgentThread, MessageRole


def ask_agent(
    agents_client: AgentsClient,      # ← was AIProjectClient
    agent:         Agent,
    thread:        AgentThread,
    message:       str,
) -> str:
    agents_client.messages.create(    # ← messages on AgentsClient
        thread_id=thread.id,
        role=MessageRole.USER,
        content=message,
    )

    run = agents_client.runs.create(  # ← runs on AgentsClient
        thread_id=thread.id,
        agent_id=agent.id,
    )

    while run.status in ("queued", "in_progress", "requires_action"):
        time.sleep(0.5)
        run = agents_client.runs.get(
            thread_id=thread.id,
            run_id=run.id,
        )

    if run.status != "completed":
        error = run.last_error.message if run.last_error else "unknown"
        raise RuntimeError(f"Foundry run failed: {run.status} — {error}")

    messages = agents_client.messages.list(thread_id=thread.id)
    for msg in messages:
        if msg.role == "assistant":
            for block in msg.content:
                if hasattr(block, "text"):
                    return block.text.value
    return ""