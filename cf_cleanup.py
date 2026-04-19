# cf_cleanup.py
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import Agent, AgentThread


def cleanup(
    agents_client: AgentsClient,
    agents:        dict[str, Agent],
    all_threads:   list[dict[str, AgentThread]],
) -> None:
    print("\n🧹 Cleaning up...")

    for ticket_threads in all_threads:
        for role, thread in ticket_threads.items():
            agents_client.threads.delete(thread.id)
            print(f"  🗑  thread [{role}]: {thread.id}")

    for role, agent in agents.items():
        agents_client.delete_agent(agent.id)
        print(f"  🗑  agent  [{role}]: {agent.name}")