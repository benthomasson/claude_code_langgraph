# claude-code-langgraph

The same minimal Claude Code clone as [claude_code_python](../claude_code_python), rebuilt with [LangGraph](https://langchain-ai.github.io/langgraph/) and [Langfuse](https://langfuse.com/) to understand how those frameworks work.

## Why this exists

The raw Anthropic SDK version has a manual `while True` loop that checks for tool calls, executes them, and sends results back. This version replaces that loop with a **LangGraph graph** — making the control flow explicit and visual:

```
[agent] --tool_calls--> [tools] --results--> [agent] --done--> END
```

Key things to compare between the two versions:

| Concept | Raw SDK (`claude_code_python`) | LangGraph (`claude_code_langgraph`) |
|---|---|---|
| Tool schemas | Hand-written JSON dicts | Auto-generated from `@tool` decorated functions |
| Tool dispatch | Manual `if/elif` in `execute_tool()` | Automatic via `ToolNode` |
| Agentic loop | `while True` + `stop_reason` check | Graph with conditional edge (`should_continue`) |
| Control flow | Implicit in code | Explicit as a graph |
| Observability | Print statements | Langfuse traces every call |

## Prerequisites

- Python 3.11+
- A Google Cloud project with the Vertex AI API enabled
- `gcloud` CLI installed and authenticated (`gcloud auth application-default login`)
- (Optional) A [Langfuse](https://langfuse.com/) account for tracing

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Set your Google Cloud project and region:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_REGION="us-east5"
```

For Langfuse tracing (optional):

```bash
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted URL
```

## Usage

```bash
python main.py
```

If Langfuse is configured, open the Langfuse dashboard to see traces for each conversation turn — including LLM calls, tool executions, token counts, and latencies.

## Project structure

```
main.py           # Graph definition and REPL
tools.py          # Tool functions (@tool decorated)
```
