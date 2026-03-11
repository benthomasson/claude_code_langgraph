# How the LangGraph Claude Code Clone Works

**Date:** 2026-03-10
**Time:** 19:47

## Overview

This project is a minimal reimplementation of Claude Code using LangGraph and Langfuse. It exists to make the core agentic tool-use pattern visible and understandable. The same functionality also exists in a raw Anthropic SDK version at `../claude_code_python` — comparing the two shows exactly what LangGraph abstracts away.

## The Graph Structure

The entire agentic loop is expressed as a LangGraph `StateGraph` with two nodes and one conditional edge:

```
[__start__] --> [agent] --tool_calls--> [tools] --results--> [agent] --done--> [__end__]
```

- **`agent` node**: Calls Claude via `ChatAnthropicVertex` with the current conversation messages. Returns the AI response as a new message.
- **`tools` node**: A prebuilt `ToolNode` that reads `tool_calls` from the last AI message, executes each tool by matching its name to the `@tool`-decorated Python functions, and returns `ToolMessage`s with results.
- **`should_continue` (conditional edge)**: After the agent responds, checks if the response contains tool calls. If yes, route to `tools`. If no, route to `END`.
- **Fixed edge**: After `tools`, always go back to `agent` so Claude can see the results.

This is defined in `build_graph()` in `main.py` (lines 29-95). The graph is compiled once at startup and invoked per conversation turn.

## State

LangGraph graphs pass state between nodes. This project uses `MessagesState`, a built-in schema that holds a single key: `messages` (a list of LangChain message objects). Each node receives the full state and returns updates to merge back in.

This is the LangGraph equivalent of the `messages = []` list in the raw version — but formalized as a typed state object that the framework manages.

## Tool Definitions

In the raw Anthropic SDK version, each tool requires two things:
1. A hand-written JSON schema dict (name, description, input_schema with properties and required fields)
2. A manual dispatch function (`if name == 'read_file': return _read_file(input['path'])`)

In LangGraph, both are replaced by a single `@tool`-decorated function:

```python
@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    with open(path, 'r') as f:
        return f.read()
```

LangChain auto-generates the JSON schema from the function signature (parameter names and types) and docstring (used as the tool description). `ToolNode` handles dispatch automatically — no if/elif chain needed.

The six tools are: `read_file`, `write_file`, `edit_file`, `grep`, `glob`, `run_command`. All defined in `tools.py`.

## The Agentic Loop: Raw vs LangGraph

**Raw version** (`claude_code_python/main.py`):
```python
for tool_round in range(MAX_TOOL_ROUNDS):
    response = client.messages.create(model=model, messages=messages, tools=TOOLS)
    messages.append({'role': 'assistant', 'content': response.content})
    if response.stop_reason == 'end_turn':
        break
    # execute tools, append results as user message
    messages.append({'role': 'user', 'content': tool_results})
```

**LangGraph version** (`claude_code_langgraph/main.py`):
```python
result = app.invoke({'messages': messages}, config={'recursion_limit': 25})
messages = result['messages']
```

The manual while/for loop, the stop_reason check, and the tool dispatch are all absorbed into the graph definition. The `recursion_limit` config parameter prevents runaway loops (each agent->tools cycle is 2 steps, so 25 allows ~10 tool rounds).

## Langfuse Integration

Langfuse provides observability — it traces every LLM call and tool execution so you can inspect timing, token usage, and the full message flow in a dashboard.

Integration uses the LangChain callback handler from `langfuse.langchain`:

```python
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()
result = app.invoke({'messages': messages}, config={'callbacks': [langfuse_handler]})
```

The callback handler hooks into LangChain's callback system. Every model invocation and tool execution inside the graph fires callbacks that Langfuse captures. No code changes to the graph or tools are needed — it is purely additive.

Requirements: set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` env vars. Tracing is optional — the app runs fine without it.

**Note**: Langfuse v4 moved the callback handler from `langfuse.callback` to `langfuse.langchain`, and requires the full `langchain` package (not just `langchain-core`).

## Vertex AI Model Routing

An important gotcha: `ChatVertexAI` routes to `publishers/google/models/` which only has Google models (Gemini). For Claude on Vertex AI, you must use `ChatAnthropicVertex` from `langchain_google_vertexai.model_garden`, which routes to `publishers/anthropic/models/`.

## Key Takeaways

1. **LangGraph makes control flow explicit.** The raw version buries the loop logic in a while loop with conditionals. LangGraph makes it a visible graph with named nodes and edges.
2. **LangChain reduces boilerplate.** `@tool` + `ToolNode` eliminates hand-written schemas and dispatch logic.
3. **The core mechanic is the same.** Both versions do the same thing: call the LLM, check for tool calls, execute tools, send results back, repeat. LangGraph just provides structure around it.
4. **Observability comes free with callbacks.** Langfuse integration is a single callback handler passed in config — no instrumentation of the graph or tools required.
