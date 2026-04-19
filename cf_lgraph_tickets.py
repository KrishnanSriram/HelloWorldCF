import os
import re
from typing import TypedDict, Literal
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = AzureAIOpenAIApiChatModel(
    project_endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
    model=os.environ["MODEL_DEPLOYMENT_NAME"],
    temperature=0,
)

def llm_invoke(prompt: str) -> str:
    response = llm.invoke([HumanMessage(content=prompt)])

    if isinstance(response.content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in response.content
        ).strip()

    return response.content.strip()


class TicketState(TypedDict):
    user_input:  str
    ticket_type: str
    response:    str
    summary:     str


def trace(node: str, msg: str):
    print(f"  ── [{node}] {msg}")


def classifier(state: TicketState) -> TicketState:
    trace("classifier", "Sending to Foundry for classification...")

    prompt = (
        "Classify the following support ticket into exactly one category.\n"
        "Reply with ONLY one word — billing, technical, or general.\n"
        "No punctuation, no explanation, just the single word.\n\n"
        f"Ticket: {state['user_input']}"
    )

    raw         = llm_invoke(prompt).lower()
    ticket_type = "general"
    for word in re.split(r'\W+', raw):
        if word in ("billing", "technical", "general"):
            ticket_type = word
            break

    trace("classifier", f"Foundry classified as '{ticket_type}' (raw: '{raw}')")
    return {**state, "ticket_type": ticket_type}


def billing_node(state: TicketState) -> TicketState:
    trace("billing_node", "Calling Foundry for billing response...")

    prompt = (
        "You are a billing support specialist. "
        "Reply to the following customer query in 2-3 sentences. "
        "Be professional and helpful. Mention that a specialist will follow up.\n\n"
        f"Customer query: {state['user_input']}"
    )

    response = llm_invoke(prompt)
    trace("billing_node", "Response received")
    return {**state, "response": f"[Billing] {response}"}


def tech_node(state: TicketState) -> TicketState:
    trace("tech_node", "Calling Foundry for technical response...")

    prompt = (
        "You are a technical support engineer. "
        "Reply to the following customer issue in 2-3 sentences. "
        "Suggest a concrete first troubleshooting step.\n\n"
        f"Customer issue: {state['user_input']}"
    )

    response = llm_invoke(prompt)
    trace("tech_node", "Response received")
    return {**state, "response": f"[Tech Support] {response}"}


def general_node(state: TicketState) -> TicketState:
    trace("general_node", "Calling Foundry for general response...")

    prompt = (
        "You are a friendly customer support agent. "
        "Reply to the following general enquiry in 2-3 sentences. "
        "Be warm and point the customer to useful resources.\n\n"
        f"Customer enquiry: {state['user_input']}"
    )

    response = llm_invoke(prompt)
    trace("general_node", "Response received")
    return {**state, "response": f"[Support] {response}"}


def summarizer(state: TicketState) -> TicketState:
    trace("summarizer", "Calling Foundry for summary...")

    prompt = (
        "Summarise the following support interaction in exactly one sentence.\n\n"
        f"Customer asked: {state['user_input']}\n"
        f"Agent responded: {state['response']}"
    )

    summary = llm_invoke(prompt)
    trace("summarizer", f"Summary: {summary}")
    return {**state, "summary": summary}


def route_by_type(
    state: TicketState,
) -> Literal["billing_node", "tech_node", "general_node"]:
    destination = {
        "billing":   "billing_node",
        "technical": "tech_node",
        "general":   "general_node",
    }.get(state["ticket_type"], "general_node")

    trace("route_by_type", f"ticket_type='{state['ticket_type']}' → '{destination}'")
    return destination


def build_graph() -> StateGraph:
    graph = StateGraph(TicketState)

    graph.add_node("classifier",   classifier)
    graph.add_node("billing_node", billing_node)
    graph.add_node("tech_node",    tech_node)
    graph.add_node("general_node", general_node)
    graph.add_node("summarizer",   summarizer)

    graph.set_entry_point("classifier")

    graph.add_conditional_edges(
        "classifier",
        route_by_type,
        {
            "billing_node": "billing_node",
            "tech_node":    "tech_node",
            "general_node": "general_node",
        },
    )

    graph.add_edge("billing_node", "summarizer")
    graph.add_edge("tech_node",    "summarizer")
    graph.add_edge("general_node", "summarizer")
    graph.add_edge("summarizer",   END)

    return graph.compile(checkpointer=MemorySaver())


tickets = [
    "I was charged twice for my subscription last month, please refund me",
    "The API keeps returning a 504 timeout error on every request",
    "What are your business hours and where are you located?",
]

print("=" * 60)
print("  LangGraph + Azure AI Foundry — Support Ticket Triage")
print(f"  Model   : {os.environ['MODEL_DEPLOYMENT_NAME']}")
print(f"  Endpoint: {os.environ['PROJECT_ENDPOINT']}")
print("  Routing : billing | technical | general")
print("=" * 60)

app = build_graph()

for i, ticket in enumerate(tickets):
    thread_id = f"ticket-{i+1}"
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state: TicketState = {
        "user_input":  ticket,
        "ticket_type": "",
        "response":    "",
        "summary":     "",
    }

    print(f"\n{'─'*60}")
    print(f"  Ticket #{i+1} | thread_id: '{thread_id}'")
    print(f"  Input: \"{ticket}\"")
    print()

    result = app.invoke(initial_state, config=config)

    print()
    print(f"  ✓ Type     : {result['ticket_type']}")
    print(f"  ✓ Response : {result['response'][:100]}...")
    print(f"  ✓ Summary  : {result['summary']}")

print(f"\n{'='*60}")
print("  All tickets processed.")
print(f"{'='*60}")