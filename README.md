# Why `@dataclass` and not `TypedDict`?

This is the heart of our question. Here's a simple comparison:

```python
# Option A - TypedDict (what TicketState uses)
class TicketContext(TypedDict):
    client:  AIProjectClient
    agents:  dict[str, Agent]
    threads: dict[str, AgentThread]

ctx = {"client": client, "agents": agents, "threads": threads}
ctx["client"]           # access like a dict
```

```python
# Option B - @dataclass (what we use)
@dataclass
class TicketContext:
    client:  AIProjectClient
    agents:  dict[str, Agent]
    threads: dict[str, AgentThread]

ctx = TicketContext(client=client, agents=agents, threads=threads)
ctx.client              # access like an object
```

`@dataclass` wins here for three specific reasons.

---

## 1. It holds live objects, not plain data

`TypedDict` is designed for plain serialisable data — strings, ints, lists, dicts. `TicketContext` holds **live SDK objects** (`AIProjectClient`, `Agent`, `AgentThread`) that have methods, network connections, and internal state.

> A `dataclass` is the right container for **objects**.  
> A `TypedDict` is the right container for **data**.

```python
# These are OBJECTS with methods - belong in a dataclass
ctx.client.agents.runs.create(...)
ctx.agents["billing"].id
ctx.threads["summarizer"].id

# These are DATA values - belong in a TypedDict
state["user_input"]     # str
state["ticket_type"]    # str
state["response"]       # str
```

---

## 2. Dot access is cleaner than dict access for infrastructure

```python
# TypedDict - feels like config
ctx["client"].agents.runs.create(...)
ctx["agents"]["billing"]
ctx["threads"]["summarizer"]

# dataclass - feels like an object, reads naturally
ctx.client.agents.runs.create(...)
ctx.agents["billing"]
ctx.threads["summarizer"]
```

---

## 3. `@dataclass` gives you `__init__` for free with named parameters

```python
# Without @dataclass you'd write this manually:
class TicketContext:
    def __init__(self, client, agents, threads):
        self.client  = client
        self.agents  = agents
        self.threads = threads

# With @dataclass Python generates that __init__ automatically.
# You just declare the fields:
@dataclass
class TicketContext:
    client:  AIProjectClient
    agents:  dict[str, Agent]
    threads: dict[str, AgentThread]

# And construction is clean with named args:
ctx = TicketContext(
    client=client,
    agents=agents,
    threads=threads,
)
```

---

## The rule to remember

Whenever you're deciding where something belongs, ask these two questions:

| Question | Answer | Belongs in |
|---|---|---|
| Does LangGraph need to **READ** this to make a routing decision? | YES | `TicketState` |
| Does a node need this to **DO** its work (call an API, talk to Foundry)? | YES | `TicketContext` |

### Applied to your code

| Field | Location | Reason |
|---|---|---|
| `ticket_type` | `TicketState` | `route_by_type()` reads it to decide the next node |
| `response` | `TicketState` | summarizer reads it to write the summary |
| `client` | `TicketContext` | nodes need it to call Foundry APIs |
| `agents` | `TicketContext` | nodes need the agent reference to run a thread |
| `threads` | `TicketContext` | nodes need the thread to post messages into |

---

> **Signal to watch for:** If you ever find yourself adding a non-string, non-int, non-bool field to `TicketState` — an SDK object, a database connection, a file handle — that's the sign it belongs in a context object instead.