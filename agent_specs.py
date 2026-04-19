import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import Agent
from azure.identity import DefaultAzureCredential

AGENT_SPECS = {
    "classifier": {
        "name": "ticket-classifier",
        "instructions": (
            "Classify support tickets into exactly one category. "
            "Reply with ONLY one word — billing, technical, or general. "
            "No punctuation, no explanation, just the single word."
        ),
    },
    "billing": {
        "name": "billing-specialist",
        "instructions": (
            "You are a billing support specialist. "
            "Reply to customer billing queries in 2-3 sentences. "
            "Be professional and helpful. "
            "Always mention that a billing specialist will follow up within 24 hours."
        ),
    },
    "tech": {
        "name": "tech-support-engineer",
        "instructions": (
            "You are a technical support engineer. "
            "Reply to customer technical issues in 2-3 sentences. "
            "Always suggest one concrete, actionable first troubleshooting step."
        ),
    },
    "general": {
        "name": "general-support",
        "instructions": (
            "You are a friendly customer support agent. "
            "Reply to general customer enquiries in 2-3 sentences. "
            "Be warm and point the customer to useful resources."
        ),
    },
    "summarizer": {
        "name": "interaction-summarizer",
        "instructions": (
            "Summarise support interactions in exactly one clear sentence. "
            "Include the issue type and resolution offered."
        ),
    },
}

def create_agents_client() -> AgentsClient:
    return AgentsClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )

def create_all_agents(
    agents_client: AgentsClient,
    specs: dict = AGENT_SPECS,
) -> dict[str, Agent]:
    print("\n🏭 Creating Foundry agents...")
    agents = {}

    for role, spec in specs.items():
        agent = agents_client.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            name=spec["name"],
            instructions=spec["instructions"],
        )
        agents[role] = agent
        print(f"  ✅ {role}: {agent.id}")

    return agents

def delete_all_agents(
    agents_client: AgentsClient,
    agents: dict[str, Agent],
) -> None:
    for role, agent in agents.items():
        agents_client.delete_agent(agent.id)
        print(f"  🗑  agent [{role}]: {agent.name}")