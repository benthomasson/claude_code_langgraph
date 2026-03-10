# CLAUDE.md

## Project overview

LangGraph + Langfuse version of the Claude Code clone. Same functionality as `../claude_code_python` but uses LangGraph for the agentic loop and Langfuse for observability. The goal is educational — understanding how these frameworks work.

## Architecture

- `main.py` — Builds the LangGraph `StateGraph`, defines the agent node and conditional routing, runs the REPL. The graph replaces the manual while loop from the raw version.
- `tools.py` — Same tools as the raw version but using LangChain's `@tool` decorator instead of hand-written JSON schemas.

## Key technical decisions

- Uses `ChatVertexAI` from `langchain-google-vertexai` for Claude on Vertex AI.
- Tools use `@tool` decorator — LangChain auto-generates the JSON schema from the function signature + docstring.
- `ToolNode` from `langgraph.prebuilt` handles tool dispatch automatically (no manual if/elif).
- `MessagesState` is the built-in state schema — just a list of messages that flows through the graph.
- Langfuse tracing is optional — runs fine without it, just set the env vars to enable.
- The graph is compiled once at startup, then invoked per conversation turn.

## Dependencies

- `langchain-google-vertexai` — ChatAnthropicVertex for Claude on Vertex
- `langgraph` — Graph-based agent framework
- `langfuse` — Observability/tracing via callback handler

## Environment variables

- `GOOGLE_CLOUD_PROJECT` — Required. Your GCP project ID.
- `GOOGLE_CLOUD_REGION` — Required. Vertex AI region (e.g. `us-east5`).
- `ANTHROPIC_MODEL` — Optional. Model name (default: `claude-sonnet-4-20250514`).
- `LANGFUSE_PUBLIC_KEY` — Optional. Enables Langfuse tracing.
- `LANGFUSE_SECRET_KEY` — Optional. Langfuse secret key.
- `LANGFUSE_HOST` — Optional. Langfuse host URL.

## Commands

```bash
# Run
python main.py

# Install deps
uv pip install -r requirements.txt
```

## Style

- Same as the raw version: clarity over cleverness.
- Comments should highlight what LangGraph is doing vs what you'd do manually.
- Keep the comparison to the raw version easy — same tools, same behavior, different structure.
