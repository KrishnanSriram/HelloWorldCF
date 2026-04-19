import os
import re
import json
import time
from dataclasses import dataclass
from typing import TypedDict, Literal
from dotenv import load_dotenv
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    Agent,
    AgentThread,
    MessageRole,
    FunctionTool,
    ToolSet,
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RunnableConfig

from agent_specs import AGENT_SPECS, create_agents_client
from cf_cleanup import cleanup

load_dotenv()

def lookup_account(customer_id: str) -> str:
    mock_accounts = {
        "CUST-001": {
            "name": "Alice Johnson",
            "plan": "Pro",
            "status": "active",
            "last_charge": "$49.99 on 2026-03-01",
            "previous_charge": "$49.99 on 2026-02-01",
        },
        "CUST-002": {
            "name": "Bob Smith",
            "plan": "Starter",
            "status": "active",
            "last_charge": "$9.99 on 2026-03-15",
            "previous_charge": "$9.99 on 2026-02-15",
        },
    }
    account = mock_accounts.get(
        customer_id,
        {"error": f"No account found for {customer_id}"},
    )
    return json.dumps(account)

def process_refund(customer_id: str, amount: str, reason: str) -> str:
    return json.dumps({
        "status":      "initiated",
        "refund_id":   f"REF-{customer_id}-001",
        "customer_id": customer_id,
        "amount":      amount,
        "reason":      reason,
        "eta":         "3-5 business days",
    })

def search_knowledge_base(query: str) -> str:
    mock_kb = {
        "504": {
            "issue":    "504 Gateway Timeout",
            "cause":    "Upstream service not responding within timeout window",
            "solution": "Check service status page. If persistent, increase timeout "
                        "setting in your API client to 30s and retry with exponential backoff.",
            "doc_link": "https://docs.example.com/errors/504",
        },
        "timeout": {
            "issue":    "Request Timeout",
            "cause":    "Network latency or overloaded endpoint",
            "solution": "Retry with exponential backoff. If persists over 1 hour, "
                        "contact support with your request ID.",
            "doc_link": "https://docs.example.com/errors/timeout",
        },
    }
    for keyword, result in mock_kb.items():
        if keyword in query.lower():
            return json.dumps(result)
    return json.dumps({
        "issue":    query,
        "solution": "No known issue found. Please provide request ID and error logs.",
    })

def build_billing_toolset() -> ToolSet:
    toolset = ToolSet()
    toolset.add(FunctionTool(functions={lookup_account, process_refund}))
    return toolset

def build_tech_toolset() -> ToolSet:
    toolset = ToolSet()
    toolset.add(FunctionTool(functions={search_knowledge_base}))
    return toolset

def build_agent_specs() -> dict:
    return {
        "classifier": {
            **AGENT_SPECS["classifier"],
            "toolset": None,
        },
        "billing": {
            "name": AGENT_SPECS["billing"]["name"],
            "instructions": (
                "You are a billing support specialist. "
                "Always use lookup_account first to check the customer's "
                "account and charge history. "
                "If a duplicate charge is confirmed, use process_refund "
                "to initiate the refund immediately. "
                "Reply professionally in 2-3 sentences and mention "
                "the refund ID if one was created."
            ),
            "toolset": build_billing_toolset(),
        },
        "tech": {
            "name": AGENT_SPECS["tech"]["name"],
            "instructions": (
                "You are a technical support engineer. "
                "Always use search_knowledge_base first to find known "
                "issues and documented solutions. "
                "Reply in 2-3 sentences with a concrete first step "
                "drawn from the knowledge base result."
            ),
            "toolset": build_tech_toolset(),
        },
        "general": {
            **AGENT_SPECS["general"],
            "toolset": None,
        },
        "summarizer": {
            **AGENT_SPECS["summarizer"],
            "toolset": None,
        },
    }

def create_all_agents_with_tools(
    agents_client: AgentsClient,
    specs: dict,
) -> dict[str, Agent]:
    print("\n🏭 Creating Foundry agents...")
    agents = {}

    for role, spec in specs.items():
        if spec["toolset"] is not None:
            agent = agents_client.create_agent(
                model=os.environ["MODEL_DEPLOYMENT_NAME"],
                name=spec["name"],
                instructions=spec["instructions"],
                toolset=spec["toolset"],
            )
            print(f"  ✅ {role}: {agent.id} (with tools)")
        else:
            agent = agents_client.create_agent(
                model=os.environ["MODEL_DEPLOYMENT_NAME"],
                name=spec["name"],
                instructions=spec["instructions"],
            )
            print(f"  ✅ {role}: {agent.id} (no tools)")

        agents[role] = agent

    return agents

def ask_agent_with_tools(
    agents_client: AgentsClient,
    agent: Agent,
    thread: AgentThread,
    message: str,
    toolset: ToolSet | None = None,
) -> str:
    agents_client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=message,
    )

    run = agents_client.runs.create(
        thread_id=thread.id,
        agent_id=agent.id,
    )

    while run.status in ("queued", "in_progress", "requires_action"):
        if run.status == "requires_action" and toolset is not None:
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            print(f"     🔧 Tool call: {[tc.function.name for tc in tool_calls]}")
            tool_outputs = toolset.execute_tool_calls(tool_calls)
            agents_client.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )
        else:
            time.sleep(0.5)

        run = agents_client.runs.get(
            thread_id=thread.id,
            run_id=run.id,
        )

    if run.status != "completed":
        error = run.last_error.message if run.last_error else "unknown"
        raise RuntimeError(f"Run failed: {run.status} — {error}")

    messages = agents_client.messages.list(thread_id=thread.id)
    for msg in messages:
        if msg.role == "assistant":
            for block in msg.content:
                if hasattr(block, "text"):
                    return block.text.value
    return ""

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
    specs: dict

def trace(node: str, msg: str):
    print(f"  ── [{node}] {msg}")

def create_ticket_threads(
    agents_client: AgentsClient,
    specs: dict,
    ticket_num: int,
) -> dict[str, AgentThread]:
    print(f"\n  🧵 Creating threads for ticket #{ticket_num}...")
    threads = {}
    for role in specs:
        thread = agents_client.threads.create()
        threads[role] = thread
        print(f"     {role}: {thread.id}")
    return threads

def classifier(state: TicketState, config: RunnableConfig) -> TicketState:
    trace("classifier", "Posting to Foundry...")
    ctx = config["configurable"]["ctx"]

    raw = ask_agent_with_tools(
        ctx.agents_client,
        ctx.agents["classifier"],
        ctx.threads["classifier"],
        f"Classify this ticket: {state['user_input']}",
        toolset=ctx.specs["classifier"]["toolset"],
    ).strip().lower()

    ticket_type = "general"
    for word in re.split(r'\W+', raw):
        if word in ("billing", "technical", "general"):
            ticket_type = word
            break

    trace("classifier", f"Classified as '{ticket_type}'")
    return {**state, "ticket_type": ticket_type}

def billing_node(state: TicketState, config: RunnableConfig) -> TicketState:
    trace("billing_node", "Posting to Foundry (with tools)...")
    ctx = config["configurable"]["ctx"]

    response = ask_agent_with_tools(
        ctx.agents_client,
        ctx.agents["billing"],
        ctx.threads["billing"],
        state["user_input"],
        toolset=ctx.specs["billing"]["toolset"],
    )

    trace("billing_node", "Response received")
    return {**state, "response": f"[Billing] {response}"}

def tech_node(state: TicketState, config: RunnableConfig) -> TicketState:
    trace("tech_node", "Posting to Foundry (with tools)...")
    ctx = config["configurable"]["ctx"]

    response = ask_agent_with_tools(
        ctx.agents_client,
        ctx.agents["tech"],
        ctx.threads["tech"],
        state["user_input"],
        toolset=ctx.specs["tech"]["toolset"],
    )

    trace("tech_node", "Response received")
    return {**state, "response": f"[Tech Support] {response}"}

def general_node(state: TicketState, config: RunnableConfig) -> TicketState:
    trace("general_node", "Posting to Foundry...")
    ctx = config["configurable"]["ctx"]

    response = ask_agent_with_tools(
        ctx.agents_client,
        ctx.agents["general"],
        ctx.threads["general"],
        state["user_input"],
        toolset=ctx.specs["general"]["toolset"],
    )

    trace("general_node", "Response received")
    return {**state, "response": f"[Support] {response}"}

def summarizer(state: TicketState, config: RunnableConfig) -> TicketState:
    trace("summarizer", "Posting to Foundry...")
    ctx = config["configurable"]["ctx"]

    summary = ask_agent_with_tools(
        ctx.agents_client,
        ctx.agents["summarizer"],
        ctx.threads["summarizer"],
        (
            f"Customer asked: {state['user_input']}\n"
            f"Agent responded: {state['response']}"
        ),
        toolset=ctx.specs["summarizer"]["toolset"],
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
    specs = build_agent_specs()
    agents = create_all_agents_with_tools(agents_client, specs)
    app = build_graph()
    all_threads = []

    tickets = [
        "Customer CUST-001 was charged twice for my subscription, please refund me",
        "The API keeps returning a 504 timeout error on every request",
        "What are your business hours and where are you located?",
    ]

    print("\n" + "=" * 60)
    print("  LangGraph + Azure AI Foundry Agent Service (v3)")
    print(f"  Model  : {os.environ['MODEL_DEPLOYMENT_NAME']}")
    print("  Tools  : billing (lookup + refund) | tech (KB search)")
    print("=" * 60)

    try:
        for i, ticket in enumerate(tickets):
            threads = create_ticket_threads(agents_client, specs, i + 1)
            all_threads.append(threads)

            ctx = TicketContext(
                agents_client=agents_client,
                agents=agents,
                threads=threads,
                specs=specs,
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
            print(f"  ✓ Response : {result['response'][:120]}...")
            print(f"  ✓ Summary  : {result['summary']}")

    finally:
        cleanup(agents_client, agents, all_threads)

    print(f"\n{'='*60}")
    print("  All tickets processed.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()