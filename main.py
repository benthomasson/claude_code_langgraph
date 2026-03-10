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
"""

import os
import sys

from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tools import ALL_TOOLS


def build_graph():
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

    # Create the model and bind tools to it.
    # bind_tools() tells the model what tools are available — same as passing
    # the tools list in the raw API, but LangChain handles schema generation.
    # ChatAnthropicVertex routes to publishers/anthropic/models/ on Vertex AI.
    # ChatVertexAI routes to publishers/google/models/ which doesn't have Claude.
    model = ChatAnthropicVertex(
        model_name=model_name,
        project=project,
        location=region,
    ).bind_tools(ALL_TOOLS)

    # --- Define the graph nodes ---

    def agent(state: MessagesState):
        """Call the LLM with the current conversation. Returns updated messages."""
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    # ToolNode is a prebuilt node that:
    # 1. Reads tool_calls from the last AI message
    # 2. Executes each tool by matching the name to our tool functions
    # 3. Returns ToolMessages with the results
    # This replaces the manual tool dispatch loop from the raw version.
    tool_node = ToolNode(ALL_TOOLS)

    # --- Define the conditional routing ---

    def should_continue(state: MessagesState):
        """Decide whether to call tools or finish.
        This is the conditional edge — the LangGraph equivalent of checking
        stop_reason == 'end_turn' in the raw version."""
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
    graph.add_edge("tools", "agent")  # After tools, always go back to agent

    return graph.compile()


def main():
    app = build_graph()

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful coding assistant. You are running inside a CLI tool "
            "on the user's machine. Help them with software engineering tasks. "
            "You have tools available to read files, write files, and run shell commands. "
            "Use them when needed to accomplish the user's task."
        ),
    }

    # Conversation history persists across turns, just like the raw version.
    messages = [system_message]

    # Langfuse tracing — uses OpenTelemetry integration.
    # Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST env vars
    # to enable. Langfuse v4 uses OTEL under the hood, so we initialize it here
    # and it automatically instruments LangChain/LangGraph calls.
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        from langfuse import get_client
        get_client()  # Initializes the Langfuse OTEL instrumentation
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

        # Remember how many messages we had before invoking, so we only
        # print the new ones (not the full conversation history).
        prev_count = len(messages)

        # Invoke the graph. LangGraph runs the agent->tools->agent loop
        # automatically until should_continue returns END.
        # Langfuse traces this automatically if initialized above.
        result = app.invoke({"messages": messages})

        # The graph returns the full message list including new messages.
        messages = result["messages"]

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


if __name__ == "__main__":
    main()
