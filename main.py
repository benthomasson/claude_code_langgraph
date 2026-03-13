"""
A minimal Claude Code clone using LangGraph + Langfuse.

This is the same concept as the raw Anthropic SDK version, but instead of
a manual while loop, the agentic tool loop is expressed as a LangGraph graph:

  [agent] --tool_calls--> [tools] --results--> [agent] --no tools--> END

Key LangGraph concepts demonstrated:
- StateGraph: a graph where each node receives and updates shared state
- MessagesState: built-in state schema that holds a list of messages
- ToolNode: a prebuilt node that executes tool calls from the model's response
- Conditional edges: route to "tools" or END based on the model's response

Langfuse provides observability — every LLM call and tool execution is traced
so you can inspect what happened in the Langfuse dashboard.

Supports three modes:
- Interactive REPL (default): python main.py
- Batch mode: python main.py --batch questions.json --output results.json
- Single query: python main.py --query "What does this code do?"
"""

import argparse
import json
import os
import sys
import time

from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tools import ALL_TOOLS, CORE_TOOLS, EXPERT_TOOLS


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful coding assistant. You are running inside a CLI tool "
    "on the user's machine. Help them with software engineering tasks. "
    "You have tools available to read files, write files, and run shell commands. "
    "Use them when needed to accomplish the user's task."
)


def load_system_prompt(prompt_arg=None):
    """Load system prompt from (in priority order):
    1. --system-prompt CLI argument (if it's a file path, read the file)
    2. SYSTEM_PROMPT_FILE environment variable (path to a file)
    3. SYSTEM_PROMPT environment variable (inline text)
    4. Default prompt
    """
    if prompt_arg:
        if os.path.isfile(prompt_arg):
            with open(prompt_arg, "r") as f:
                return f.read().strip()
        return prompt_arg

    prompt_file = os.environ.get("SYSTEM_PROMPT_FILE")
    if prompt_file and os.path.isfile(prompt_file):
        with open(prompt_file, "r") as f:
            return f.read().strip()

    prompt_env = os.environ.get("SYSTEM_PROMPT")
    if prompt_env:
        return prompt_env

    return DEFAULT_SYSTEM_PROMPT


def select_tools(use_beliefs=True):
    """Select which tools to make available based on configuration."""
    if use_beliefs and os.environ.get("BELIEFS_FILE"):
        return ALL_TOOLS
    if use_beliefs:
        return ALL_TOOLS
    return CORE_TOOLS


def build_graph(tools=None):
    """
    Build the LangGraph agent graph.

    The graph has two nodes:
    - "agent": calls the LLM with the current messages
    - "tools": executes any tool calls the LLM made

    And one conditional edge:
    - After "agent", if the response contains tool calls -> go to "tools"
    - After "agent", if no tool calls -> go to END
    - After "tools", always go back to "agent" (so it can see the results)
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    region = os.environ.get("GOOGLE_CLOUD_REGION", "us-east5")
    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    if not project:
        print("Error: set GOOGLE_CLOUD_PROJECT environment variable")
        sys.exit(1)

    if tools is None:
        tools = ALL_TOOLS

    # Create the model and bind tools to it.
    model = ChatAnthropicVertex(
        model_name=model_name,
        project=project,
        location=region,
    ).bind_tools(tools)

    # --- Define the graph nodes ---

    def agent(state: MessagesState):
        """Call the LLM with the current conversation. Returns updated messages."""
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    # --- Define the conditional routing ---

    def should_continue(state: MessagesState):
        """Decide whether to call tools or finish."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # --- Assemble the graph ---

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


def get_langfuse_handler():
    """Create Langfuse callback handler if configured."""
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    return None


def invoke_agent(app, messages, langfuse_handler=None, recursion_limit=25):
    """Invoke the agent graph and return the result.

    Args:
        app: Compiled LangGraph app
        messages: List of message dicts
        langfuse_handler: Optional Langfuse callback handler
        recursion_limit: Max graph steps (each agent->tools cycle is 2 steps)

    Returns:
        dict with keys:
        - messages: Updated message list
        - response: Final text response from the agent
        - tool_calls: List of tool calls made during this invocation
        - error: Error message if something went wrong, None otherwise
    """
    config = {"recursion_limit": recursion_limit}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    prev_count = len(messages)

    try:
        result = app.invoke({"messages": messages}, config=config)
    except Exception as e:
        if "recursion" in str(e).lower():
            return {
                "messages": messages,
                "response": None,
                "tool_calls": [],
                "error": "Too many tool rounds",
            }
        raise

    new_messages = result["messages"]

    # Extract tool calls and final response from new messages
    tool_calls = []
    response = None
    for msg in new_messages[prev_count:]:
        if msg.type == "ai":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"name": tc["name"], "args": tc["args"]})
            if msg.content:
                response = msg.content

    return {
        "messages": new_messages,
        "response": response,
        "tool_calls": tool_calls,
        "error": None,
    }


def run_repl(app, system_prompt, langfuse_handler=None):
    """Run the interactive REPL."""
    messages = [{"role": "system", "content": system_prompt}]

    if langfuse_handler:
        print("(Langfuse tracing enabled)")
    print("Claude Code (LangGraph) — type 'quit' to exit")
    print()

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.strip().lower() in ("quit", "exit"):
            break

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        prev_count = len(messages)

        result = invoke_agent(app, messages, langfuse_handler)
        messages = result["messages"]

        if result["error"]:
            print(f"\n[stopped: {result['error']}]")
            print()
            continue

        # Print only the new messages from this turn.
        for msg in messages[prev_count:]:
            if msg.type == "ai":
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"\n[tool: {tc['name']}({tc['args']})]")
                if msg.content:
                    print()
                    print(msg.content)
            elif msg.type == "tool":
                content = msg.content
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"[result: {content}]")

        print()


def run_batch(app, system_prompt, questions_file, output_file, langfuse_handler=None):
    """Run batch mode — process questions from a JSON file.

    Input JSON format:
    [
        {"id": "q1", "question": "What does this function do?"},
        {"id": "q2", "question": "How many tests are there?"},
        ...
    ]

    Output JSON format:
    [
        {
            "id": "q1",
            "question": "What does this function do?",
            "response": "The function...",
            "tool_calls": [{"name": "read_file", "args": {"path": "..."}}],
            "error": null,
            "duration_s": 12.3
        },
        ...
    ]
    """
    with open(questions_file, "r") as f:
        questions = json.load(f)

    print(f"Running {len(questions)} questions in batch mode...")
    results = []

    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"q{i}")
        question = q["question"]
        print(f"  [{i}/{len(questions)}] {qid}: {question[:60]}...")

        # Fresh conversation for each question
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        start = time.time()
        result = invoke_agent(app, messages, langfuse_handler)
        duration = time.time() - start

        results.append({
            "id": qid,
            "question": question,
            "response": result["response"],
            "tool_calls": result["tool_calls"],
            "error": result["error"],
            "duration_s": round(duration, 2),
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_file}")

    # Print summary
    errors = sum(1 for r in results if r["error"])
    avg_duration = sum(r["duration_s"] for r in results) / len(results) if results else 0
    total_tools = sum(len(r["tool_calls"]) for r in results)
    print(f"  {len(results)} questions, {errors} errors, {total_tools} tool calls")
    print(f"  avg {avg_duration:.1f}s per question")


def run_single(app, system_prompt, query, langfuse_handler=None):
    """Run a single query and print the response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    result = invoke_agent(app, messages, langfuse_handler)

    if result["error"]:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if result["response"]:
        print(result["response"])


def main():
    parser = argparse.ArgumentParser(description="Claude Code (LangGraph)")
    parser.add_argument("--system-prompt", help="System prompt text or path to a prompt file")
    parser.add_argument("--no-beliefs", action="store_true", help="Disable the beliefs lookup tool")
    parser.add_argument("--batch", metavar="QUESTIONS_FILE", help="Run in batch mode with a JSON questions file")
    parser.add_argument("--output", metavar="RESULTS_FILE", default="results.json", help="Output file for batch results (default: results.json)")
    parser.add_argument("--query", metavar="QUERY", help="Run a single query and exit")
    args = parser.parse_args()

    system_prompt = load_system_prompt(args.system_prompt)
    tools = select_tools(use_beliefs=not args.no_beliefs)
    app = build_graph(tools=tools)
    langfuse_handler = get_langfuse_handler()

    if args.batch:
        run_batch(app, system_prompt, args.batch, args.output, langfuse_handler)
    elif args.query:
        run_single(app, system_prompt, args.query, langfuse_handler)
    else:
        run_repl(app, system_prompt, langfuse_handler)


if __name__ == "__main__":
    main()
