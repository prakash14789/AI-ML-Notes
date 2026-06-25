# 83. AI Agents with LangGraph - Suman - 27 May 2026

# AI Agents with LangGraph

## PDF File: [Click Here](https://drive.google.com/file/d/1OcWzm8ptyrloyOhWr7HvTXcN9O2fNYaR/view?usp=sharing)

## 1. What You'll Learn in This Section

In this lesson, you'll learn to:

- Distinguish AI agents from LLMs and chatbots, and explain what makes agents capable of real action
- Identify the five core components every AI agent needs to function
- Build a LangGraph agent using its four core building blocks: LLM node, state, edges, and graph execution
- Apply LangGraph's key concepts — stateful execution, cyclic workflows, and conditional routing — to design multi-step agent workflows

---

## 2. Detailed Explanation

### What AI Agents Are

An **AI agent** is an intelligent system that perceives its environment, makes autonomous decisions, and acts to achieve a goal. It interacts with tools, APIs, and databases to get real work done.

**Why it matters**

Most AI tools only generate text — they suggest what to do next. Agents actually do it. A chatbot tells you which trains are available; an AI agent books the ticket and emails the confirmation.

**Walkthrough**

Agents have five capabilities that set them apart:

- **Perceive** — sense inputs from the environment.
- **Make decisions autonomously** — reason without needing a human at every step.
- **Take actions** — the most important distinguishing feature over LLMs and chatbots.
- **Interact with tools, APIs, and databases** — call whatever external service the task requires.
- **Learn or adapt based on feedback** — improve when feedback is provided.

The classic analogy is a **personal assistant vs. a chatbot**. A chatbot answers questions — you ask, it replies, nothing else happens. A personal assistant (the agent) reads email, searches the web, and schedules meetings — no permission needed at each step. Give it a goal, and it works autonomously.

**Common mistakes**

- Treating an AI agent as just a smarter chatbot. The defining difference is *action* — an agent performs real tasks, not just text generation.
- Assuming agents work with no structure. Every agent needs all five components (goal, reasoning, memory, tools, actions) to function reliably.

---

### LLMs vs. Chatbots vs. AI Agents

Understanding where agents sit relative to existing tools helps clarify what problem they solve.

**Why it matters**

Teams sometimes pick the wrong tool — deploying a full agent when a chatbot would do, or using a chatbot when the task needs multi-step API execution. Knowing the difference prevents over-engineering and under-engineering.

**Walkthrough**

Consider the task: *book the cheapest flight from Lucknow to Delhi for tomorrow morning.*

Tool | What it does
LLM | Understands the request; suggests going to a website and checking airline options.
Chatbot | Asks for departure time and preferred airline; suggests flights; may provide a booking link; cannot perform the booking itself.
AI agent | Searches flights, compares prices, selects a suitable option, asks for confirmation, books the ticket, and sends the ticket to the user's email.

The full comparison across dimensions:

Dimension | LLM | Chatbot | AI agent
Core role | Understands and generates language | Conversational interface | Autonomous system that reasons, plans, and acts
Decision-making | Limited | Predefined rules | Advanced reasoning; dynamic decisions
Memory | Short-term context window | Short-term or none | Extended; context carried across all steps
Examples | General-purpose LLMs | Banking support, FAQ bots | Autonomous coding agents, ticket-booking agents

**Common mistakes**

- Conflating LLMs with agents. An LLM is the *brain* inside an agent — it is one component, not the whole system.
- Assuming chatbots have no memory. Rule-based chatbots typically have none; even conversational chatbots keep only short-term context.

---

### Five Core Components of an AI Agent

Every working AI agent is built from exactly five components working together.

**Why it matters**

If any component is missing or poorly designed, the agent fails mid-task or produces unreliable results. Knowing all five lets you diagnose weaknesses in an existing agent and fill gaps when designing a new one.

**Walkthrough**

1. **Goal** — A clear task the agent must accomplish (e.g., "find the cheapest flight").
2. **Reasoning and planning** — The agent breaks the goal into steps. For a flight booking: search flights → compare prices → select option → fill form → book.
3. **Memory** — The agent remembers what it has done. Moving from state 1 to state 2, everything accomplished in state 1 is carried forward. Memory enables long-running, multi-step workflows.
4. **Tools** — The agent calls specialised external capabilities as needed: a calculator, a weather API, a calendar API, a code executor, a database.
5. **Actions** — Unlike LLMs and chatbots, which only generate text, agents perform real actions. This is the most important distinguishing feature.

**Common mistakes**

- Designing an agent with no clear goal. Without a concrete, scoped goal, the agent will drift or loop indefinitely.
- Forgetting to wire up memory. An agent that can't carry state between steps is forced to restart the task from scratch at every node.

---

### Tools Used to Build AI Agents

A production AI agent is assembled from several categories of tooling, not just an LLM.

**Why it matters**

A common beginner mistake is treating the LLM as the only component. The LLM is the reasoning core, but retrieval, embeddings, APIs, and a framework like LangGraph are all needed for a robust agent.

**Walkthrough**

Category | Role
LLM | The "brain" — generates text, analyzes input, decides which tool to call
Vector database + RAG | Retrieval-augmented generation when external knowledge is needed
Embedding models | Convert text to vectors for semantic search within RAG pipelines
Agent framework | LangGraph (covered here); other frameworks also exist
Tools and APIs | Weather APIs, email APIs, calendar APIs, database connectors, code-execution tools
Deployment and monitoring | Cloud or containerised deployment; monitoring for hallucinations, latency, and agent improvement

**Common mistakes**

- Skipping the agent framework and wiring everything manually. Frameworks like LangGraph handle state, loops, and routing — reinventing this by hand is fragile and hard to maintain.

---

### LangGraph — What It Is and Why Use It

**LangGraph** is a framework for building AI agents that involve multi-step, stateful workflows. It is part of the LangChain ecosystem — specifically, an extension of LangChain that addresses limitations in the base library.

**Why it matters**

Traditional approaches, including simple LangChain pipelines, suffer from four problems that break down for complex agents:

- Linear flow only — input → process → output, with no way to loop back and improve.
- Limited memory — context is not reliably passed across steps.
- Cannot handle loops and conditional decisions easily.
- Difficult to manage complex workflows.

LangGraph resolves all four.

**Walkthrough**

What LangGraph provides:

- **Multi-step reasoning** with full state persistence across all steps.
- **Cyclic (looping) workflows** — a node can repeat until a condition is satisfied.
- **Conditional routing** — different execution paths based on dynamic conditions.
- **Multi-agent coordination** — multiple specialised agents working together.
- **Maintained agent state** through a graph-based execution model.

```

```

**Common mistakes**

- Using LangGraph for tasks that need only a single LLM call. LangGraph shines for multi-step, stateful, or looping workflows — a one-shot query is simpler with plain LangChain or a direct API call.
- Confusing LangChain with LangGraph. LangChain is the base library; LangGraph is the extension that adds graph-based, stateful, cyclic execution.

---

### Four Key LangGraph Concepts

These four concepts are the design principles that explain *why* LangGraph works the way it does.

### Stateful execution

**Stateful execution** means the system remembers information across all steps while performing a task. Each step is a "state," and when moving to the next step, all accumulated data travels with it.

**Why it matters**

Without state, every node would start fresh with no knowledge of what previous nodes did. A travel agent workflow that forgets the origin city by the time it reaches the booking node is useless.

**Walkthrough**

What gets stored in state: previous outputs, conversational history, retrieved documents, task progress, and decisions made.

Travel agent example: booking a trip from Tokyo to Washington on a ₹80,000 budget. Origin city, destination, budget, and preferred airline are collected in separate steps. State accumulates all of it so later nodes have the full picture.

**Common mistakes**

- Treating state as a global variable with uncontrolled read/write access. State in LangGraph is passed explicitly from node to node — each node reads and writes only its relevant fields.

---

### Cyclic workflows

**Cyclic workflows** allow the agent to loop back and repeat steps until a condition is satisfied, rather than always moving forward in a fixed sequence.

**Why it matters**

Many real tasks require iteration. A coding agent that writes code once, never runs tests, and immediately ships is dangerous. Cycles let the agent self-correct.

**Walkthrough**

AI coding agent example:

1. Agent writes code.
2. Runs tests.
3. Finds errors → fixes errors.
4. Runs tests again.
5. Repeats until all tests pass.
6. Advances to the next phase only when the condition (error-free tests) is met.

Use cases: self-correction, iterative improvement, autonomous agent planning.

**Common mistakes**

- Building cyclic workflows without a termination condition. Always define a clear exit condition (e.g., "all tests pass" or "maximum 5 retries reached") to avoid infinite loops.

---

### Conditional routing

**Conditional routing** means different execution paths are chosen at runtime based on dynamic conditions — analogous to if/else logic in code, but the branches may each call different APIs or agents.

**Why it matters**

Not every query needs the same processing path. Routing ensures each query takes the most efficient path, avoiding unnecessary API calls and latency.

**Walkthrough**

Example 1 — routing by query type:

- User asks about billing → route to the billing support agent.
- User asks a technical question → route to the technical support agent.

Example 2 — routing by confidence level:

- LLM generates an answer with ≥ 90% confidence → return that answer directly.
- Confidence below 90% → search the vector database using RAG and regenerate the answer.

The decision is dynamic: the same agent may take different paths for different inputs in the same session.

Use cases: dynamic decision-making, intelligent branching, efficient workflow management.

**Common mistakes**

- Hard-coding routing logic in the node itself instead of using a dedicated router function. LangGraph's `add_conditional_edges` with a router function keeps routing logic clean and testable.

---

### Multi-agent systems

**Multi-agent systems** are setups where multiple specialised agents — each with its own role, tools, and responsibilities — work together to handle complex tasks that no single agent handles well alone.

**Why it matters**

Complex tasks often involve varied sub-tasks requiring different skills. Splitting responsibilities across specialised agents makes each agent simpler and more reliable than one monolithic agent trying to do everything.

**Walkthrough**

The router function in LangGraph decides which agent receives control at each decision point. A customer support system might have a billing agent, a technical support agent, and an escalation agent. The router directs each query to the right one.

**Common mistakes**

- Over-splitting tasks into too many agents. Each agent boundary adds coordination overhead. Keep specialisation meaningful — split only when the sub-task genuinely requires different tools or reasoning strategies.

---

### The Four LangGraph Building Components

These are the concrete, code-level pieces you assemble to build any LangGraph agent.

### LLM node

The **LLM node** is the brain of the agent — typically the first node to receive user input. It analyzes the query, decides whether to generate a final answer or call a specific tool, and outputs that decision.

**Why it matters**

The LLM node translates a natural-language query into a structured action that the rest of the graph can execute. Without it, the agent has no way to interpret what the user wants.

**Walkthrough**

Example — "what is the temperature in Delhi today?":

1. LLM node analyzes the query.
2. Decides: external weather data is needed.
3. Output: `call weather API, city = Delhi`.

The LLM node produces either a **final answer** (when no external tool is needed) or a **tool-call instruction** (when one is required). It does not always produce the user-facing response.

**Common mistakes**

- Expecting the LLM node to always produce the final user-facing answer. For tool-dependent queries, its output is an intermediate instruction, not the finished response.

---

### State

The **state** is a shared data structure that evolves as the agent moves from node to node. It acts as the agent's working memory for the duration of a task.

**Why it matters**

State is how nodes communicate. Without a well-defined state, nodes cannot pass results to each other, and the agent cannot maintain context across a multi-step workflow.

**Walkthrough**

State evolution for "find AI books":

Stage | State contents
S-0 (start) | user_query: "find AI books",retrieved_docs: [],final_answer: ""
S-1 (after LLM node) | user_query: "find AI books",retrieved_docs: [book1, book2, ...],final_answer: ""
S-2 (after final node) | user_query: "find AI books",retrieved_docs: [book1, book2, ...],final_answer: "Here are AI books: …"

The state is like a shared notebook: a researcher appends retrieved documents; a writer reads them later to generate the answer. Each team member reads and writes only their relevant section.

**Typed state** replaces the generic default dictionary with a custom schema that has explicit fields and types:

```python
# Typed state — define a custom schema with explicit fields
class GraphState(TypedDict):
    query: str
    intent: str
    result: str

```

**Passing state** is the act of taking the current state, updating it with the current node's output, and forwarding the updated version to the next node.

Concept | Meaning
Shared state | All nodes work on the same evolving state object
Passing state | The latest version is passed explicitly from node to node

**Common mistakes**

- Using a plain untyped dictionary for state in production agents. Typed state (`TypedDict`) makes the schema explicit, catches field-name typos early, and avoids carrying irrelevant fields.
- Assuming state is a physical global variable. It is a logical shared object — data is explicitly passed, not broadcast.

---

### Edges

**Edges** define how nodes are connected and in what order execution flows. Every edge links a source node to a destination node.

**Why it matters**

Without edges, nodes are isolated functions with no execution order. Edges are the wiring that turns a set of nodes into a coherent workflow.

**Walkthrough**

Nodes are like cities; edges are the roads connecting them.

```python
# Adding edges to define execution order
graph.add_edge("llm_node", "search_node")
graph.add_edge("search_node", "summary_node")

```

For conditional branching, use **conditional edges**. The router function decides at runtime which outgoing edge to follow:

```python
# Conditional edges — router picks the next node at runtime
graph.add_conditional_edges(
    "llm_node",
    router,
    {
        "weather_tool": "weather_tool_node",
        "general": "general_llm_node"
    }
)

```

Both destination nodes then connect to the end:

```python
graph.add_edge("weather_tool_node", END)
graph.add_edge("general_llm_node", END)

```

**Common mistakes**

- Forgetting to connect terminal nodes to `END`. If a node has no outgoing edge, graph execution will hang or raise an error.
- Using only fixed edges when the workflow needs branching. Conditional edges are the correct tool for dynamic routing.

---

### Graph execution

**Graph execution** controls the order in which nodes run, manages loops, and enforces conditional routing. It acts as the workflow controller that brings the whole graph to life.

**Why it matters**

Defining nodes and edges only creates a blueprint. Graph execution (compile + invoke) is what actually runs the workflow with a real input.

**Walkthrough**

After all nodes, edges, and conditional edges are defined:

```python
# Compile the graph, then invoke it with an input
app = graph.compile()
result = app.invoke({"query": "What is the weather in Delhi today?"})
print(result)

```

Compiling makes the graph ready for execution. Invoking it runs the full workflow with the given input and returns the final state.

**Common mistakes**

- Invoking the graph without compiling it first. Always call `graph.compile()` before `app.invoke(...)`.
- Passing the wrong input keys to `invoke`. The keys must match the field names defined in the typed state schema.

---

### Worked Example — Weather Assistant

Putting all four building components together in a complete, runnable agent.

**Why it matters**

Seeing all components interact end-to-end — typed state, LLM node, tool node, edges, compilation, and invocation — cements how they fit together before you write your own agent.

**Walkthrough**

**Goal:** Build an AI agent that answers weather-related queries.

**User query:** "What is the weather in Delhi today?"

**Workflow:**

```
User input → LLM node → Weather tool node → Final response node → End

```

Step-by-step execution:

1. **LLM node** receives the query combined with the system instruction ("you are a helpful weather assistant"). It decides external data is needed and outputs `action: call_weather_api, city: Delhi`.
2. **Weather tool node** calls the weather API with `city = Delhi`. It returns `temperature: 38°C, condition: sunny`.
3. **Final response node** uses the user query plus the tool output to generate: "The weather in Delhi today is 38 degrees Celsius and it is sunny."

State evolution:

After node | State
Start | user_query: "What is the weather in Delhi today?"
After LLM node | +action: call_weather_api,city: Delhi
After weather tool node | +temperature: 38°C,condition: sunny
After final response node | +final_response: "The weather in Delhi today is 38°C and sunny."

Full code structure:

```python
# Weather assistant — typed state + nodes + graph
class WeatherState(TypedDict):
    user_query: str
    city: str
    weather_data: str
    final_response: str

def llm_node(state): ...           # analyzes query; outputs tool-call instruction
def weather_tool_node(state): ...  # calls weather API; appends temperature/condition
def response_node(state): ...      # generates final natural-language answer

graph = StateGraph(WeatherState)
graph.add_node("llm_node", llm_node)
graph.add_node("weather_tool_node", weather_tool_node)
graph.add_node("response_node", response_node)

graph.set_entry_point("llm_node")
graph.add_edge("llm_node", "weather_tool_node")
graph.add_edge("weather_tool_node", "response_node")
graph.add_edge("response_node", END)

app = graph.compile()
result = app.invoke({"user_query": "What is the weather in Delhi today?"})

```

This example uses a linear flow — no looping, no conditional branching. The execution always proceeds LLM node → weather tool node → response node → end.

**Common mistakes**

- Mixing the LLM node's job with the tool node's job. The LLM node *decides* which tool to call; the tool node *executes* the call. Keeping them separate makes each node easier to test and replace.

---

### Memory Types in LangGraph

**Memory** in LangGraph is maintained through the state. It allows agents to carry context across all nodes in a workflow — and across longer interactions via specialised memory mechanisms.

**Why it matters**

An agent without memory is stateless — it cannot handle multi-turn conversations, long-running tasks, or workflows that span many steps. Choosing the right memory type prevents context loss at scale.

**Walkthrough**

Memory type | Description
State memory | The primary mechanism — the state object passed from node to node
Checkpoint memory | Saves snapshots of state at specific points
Vector database memory | Semantic search over stored knowledge using embeddings
Conversation memory | Retains the full conversation history for chatbot-style agents

Applications: long conversation management, persistent workflows, retrieval-augmented context.

**Common mistakes**

- Relying only on state memory for long-running workflows. For tasks that persist beyond a single invocation, checkpoint memory or vector database memory is needed to avoid losing context.

---

### Retries

**Retries** handle node failures automatically. A retry node monitors execution and re-runs any node that fails, without requiring human intervention.

**Why it matters**

Real-world APIs and services fail. An agent that crashes permanently the first time a weather API times out is not production-ready. Retries give agents robustness and reliability.

**Walkthrough**

Situations where retries apply:

- LLM response times out.
- An API call fails (e.g., the weather API is unreachable).
- A database is temporarily unavailable.
- A tool crashes mid-execution.

The retry mechanism works like `try/catch`: the system watches for exceptions, and when a node fails, the retry node triggers re-execution automatically.

**Common mistakes**

- Retrying indefinitely with no maximum count. Always set an upper bound on retries to avoid infinite loops when a service is permanently down.

---

### Citations

**Citations** are the sources the agent used to generate its output. Including them increases the trustworthiness and reproducibility of the agent's answers.

**Why it matters**

When an agent retrieves documents to answer questions, users and auditors need to know where the answer came from. Without citations, the output is a black box — hard to trust and impossible to audit.

**Walkthrough**

Citations work like a research paper: when an AI agent produces an answer, it accompanies that answer with the documents or URLs it drew from.

Citations are most relevant in:

- RAG-based systems (retrieval-augmented generation).
- Search agents.
- Research assistants.
- Document question-answering systems.

Example state with citations:

```python
# State carries both the answer and its sources
{
    "answer": "LangGraph is a framework for building stateful, multi-step AI agents.",
    "sources": ["langchain docs – langgraph overview", "github.com/langchain-ai/langgraph"]
}

```

**Common mistakes**

- Omitting citations in RAG-based agents. Without sources, users cannot verify the answer, and the agent's credibility suffers in any professional or compliance context.

---

### Environment Setup

Before writing any agent code, three setup steps are required.

**Why it matters**

Skipping setup or misconfiguring API keys is the most common reason a first agent run fails. Getting the environment right once saves hours of debugging later.

**Walkthrough**

1. **Install packages** — LangChain, LangGraph, and the LLM SDK for your chosen provider (e.g., the OpenAI SDK for GPT models or the Anthropic SDK for Claude models).
2. **Configure API keys** — keys are needed to connect to the LLM provider and any external service APIs (weather, calendar, email, etc.).
3. **Set up a Python project environment** — standard virtual environment or similar project structure.

**Common mistakes**

- Hard-coding API keys directly in source files. Use environment variables or a secrets manager — never commit keys to version control.

---

## 3. Key Takeaways

- An AI agent **perceives, decides, acts, and learns** — it performs real tasks rather than only generating text suggestions, which is the fundamental distinction from LLMs and chatbots.
- Every agent needs five components: **goal, reasoning and planning, memory, tools, and actions**. Missing any one of them breaks the agent's ability to complete a multi-step task reliably.
- **LangGraph** solves the four limitations of linear pipelines — no memory, no loops, no conditional routing, no multi-agent coordination — by using a graph-based, stateful execution model.
- The four LangGraph building components (**LLM node, state, edges, graph execution**) map directly to what the agent thinks, remembers, routes, and runs. The weather assistant example shows them working together end-to-end.
- Production agents also need **memory types** (state, checkpoint, vector DB, conversation), **retries** for fault tolerance, and **citations** for auditability — each extending the base agent with persistence, robustness, and transparency.

**Mental model:** Think of a LangGraph agent as a relay team. Each runner (node) picks up the baton (state), completes their leg, and hands it off. The coach (graph execution) decides the route and the baton carries all progress. If a runner stumbles, the team loops back and re-runs that leg.

            .markdown-preview table, 
            .markdown-preview th, 
            .markdown-preview td {
              background-color: white !important;
              color: black !important;
            }
            .markdown-preview pre, 
            .markdown-preview code {
              background-color: inherit !important;
              color: inherit !important;
              box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }