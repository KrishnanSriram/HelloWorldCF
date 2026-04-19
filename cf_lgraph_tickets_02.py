import os
import re
from dataclasses import dataclass
from typing import TypedDict, Literal
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import Agent, AgentThread
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RunnableConfig
from foundry_utils import ask_agent
from agent_specs import AGENT_SPECS, create_all_agents, create_agents_client
from cf_cleanup import cleanup
from dotenv import load_dotenv

load_dotenv()

class TicketState(TypedDict):
    user_input: str
    ticket_type: str
    response: str
    summary: str

@dataclass
class TicketContext:
    agents_client: AgentsClient
    agents: dict[str, Agent]
    threads: dict[str, AgentThread]

def trace(node: str, msg: str):
    print(f"  ── [{node}] {msg}")

def create_ticket_threads(
    agents_client: AgentsClient,
    ticket_num: int,
) -> dict[str, AgentThread]:
    print(f"\n  🧵 Creating threads for ticket #{ticket_num}...")
    threads = {}
    for role in AGENT_SPECS:
        thread = agents_client.threads.create()
        threads[role] = thread
        print(f"     {role}: {thread.id}")
    return threads

def classifier(state: TicketState, config: RunnableConfig) -> TicketState:
    trace("classifier", "Posting to Foundry...")
    ctx = config["configurable"]["ctx"]
    raw = ask_agent(
        ctx.agents_client,
        ctx.agents["classifier"],
        ctx.threads["classifier"],
        f"Classify this ticket: {state['user_input']}",
    ).strip().lower()

    ticket_type = "general"
    for word in re.split(r'\W+', raw):
        if word in ("billing", "technical", "general"):
            ticket_type = word
            break

    trace("classifier", f"Classified as '{ticket_type}'")
    return {**state, "ticket_type": ticket_type}

def billing_node(state: TicketState, config: RunnableConfig) -> TicketState:
    trace("billing_node", "Posting to Foundry...")
    ctx = config["configurable"]["ctx"]
    response = ask_agent(
        ctx.agents_client,
        ctx.agents["billing"],
        ctx.threads["billing"],
        state["user_input"],
    )
    trace("billing_node", "Response received")
    return {**state, "response": f"[Billing] {response}"}

def tech_node(state: TicketState, config: RunnableConfig) -> TicketState:
    trace("tech_node", "Posting to Foundry...")
    ctx = config["configurable"]["ctx"]
    response = ask_agent(
        ctx.agents_client,
        ctx.agents["tech"],
        ctx.threads["tech"],
        state["user_input"],
    )
    trace("tech_node", "Response received")
    return {**state, "response": f"[Tech Support] {response}"}

def general_node(state: TicketState, config: RunnableConfig) -> TicketState:
    trace("general_node", "Posting to Foundry...")
    ctx = config["configurable"]["ctx"]
    response = ask_agent(
        ctx.agents_client,
        ctx.agents["general"],
        ctx.threads["general"],
        state["user_input"],
    )
    trace("general_node", "Response received")
    return {**state, "response": f"[Support] {response}"}

def summarizer(state: TicketState, config: RunnableConfig) -> TicketState:
    trace("summarizer", "Posting to Foundry...")
    ctx = config["configurable"]["ctx"]
    summary = ask_agent(
        ctx.agents_client,
        ctx.agents["summarizer"],
        ctx.threads["summarizer"],
        (
            f"Customer asked: {state['user_input']}\n"
            f"Agent responded: {state['response']}"
        ),
    )
    trace("summarizer", f"Summary: {summary}")
    return {**state, "summary": summary}

def route_by_type(
    state: TicketState,
) -> Literal["billing_node", "tech_node", "general_node"]:
    destination = {
        "billing": "billing_node",
        "technical": "tech_node",
        "general": "general_node",
    }.get(state["ticket_type"], "general_node")
    trace("route_by_type", f"'{state['ticket_type']}' → '{destination}'")
    return destination

def build_graph() -> StateGraph:
    graph = StateGraph(TicketState)
    graph.add_node("classifier", classifier)
    graph.add_node("billing_node", billing_node)
    graph.add_node("tech_node", tech_node)
    graph.add_node("general_node", general_node)
    graph.add_node("summarizer", summarizer)
    graph.set_entry_point("classifier")
    graph.add_conditional_edges(
        "classifier",
        route_by_type,
        {
            "billing_node": "billing_node",
            "tech_node": "tech_node",
            "general_node": "general_node",
        },
    )
    graph.add_edge("billing_node", "summarizer")
    graph.add_edge("tech_node", "summarizer")
    graph.add_edge("general_node", "summarizer")
    graph.add_edge("summarizer", END)
    return graph.compile(checkpointer=MemorySaver())

def main():
    agents_client = create_agents_client()
    agents = create_all_agents(agents_client)
    app = build_graph()
    all_threads = []

    tickets = [
        "I was charged twice for my subscription last month, please refund me",
        "The API keeps returning a 504 timeout error on every request",
        "What are your business hours and where are you located?",
    ]

    print("\n" + "=" * 60)
    print("  LangGraph + Azure AI Foundry Agent Service (v3)")
    print(f"  Model  : {os.environ['MODEL_DEPLOYMENT_NAME']}")
    print(f"  Agents : {len(agents)} specialist agents")
    print("=" * 60)

    try:
        for i, ticket in enumerate(tickets):
            threads = create_ticket_threads(agents_client, i + 1)
            all_threads.append(threads)

            ctx = TicketContext(
                agents_client=agents_client,
                agents=agents,
                threads=threads,
            )

            config = {
                "configurable": {
                    "thread_id": f"ticket-{i+1}",
                    "ctx": ctx,
                }
            }

            initial_state: TicketState = {
                "user_input": ticket,
                "ticket_type": "",
                "response": "",
                "summary": "",
            }

            print(f"\n{'─'*60}")
            print(f"  Ticket #{i+1}: \"{ticket}\"")
            print()

            result = app.invoke(initial_state, config=config)

            print()
            print(f"  ✓ Type     : {result['ticket_type']}")
            print(f"  ✓ Response : {result['response'][:100]}...")
            print(f"  ✓ Summary  : {result['summary']}")

    finally:
        cleanup(agents_client, agents, all_threads)

    print(f"\n{'='*60}")
    print("  All tickets processed.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()