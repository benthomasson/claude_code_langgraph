"""
Tools for the LangGraph Claude Code clone.

In the raw Anthropic version, tools are JSON schemas + manual dispatch functions.
In LangChain/LangGraph, tools are Python functions decorated with @tool.
LangChain automatically generates the JSON schema from the function signature
and docstring, and LangGraph's ToolNode handles dispatching.
"""

import glob as glob_module
import os
import re
import subprocess

from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path. Use this to examine existing code or files."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path. Creates the file if it doesn't exist, overwrites if it does."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Make a surgical edit to a file by replacing an exact string match.
    The old_string must appear exactly once in the file (including whitespace and indentation).
    Use read_file first to see the current contents.
    Prefer this over write_file when modifying existing files."""
    try:
        with open(path, "r") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}"
        if count > 1:
            return f"Error: old_string appears {count} times in {path} — must be unique"

        new_content = content.replace(old_string, new_string, 1)
        with open(path, "w") as f:
            f.write(new_content)
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing file: {e}"


@tool
def grep(pattern: str, path: str = ".") -> str:
    """Search for a regex pattern across files in a directory.
    Returns matching lines with file paths and line numbers.
    Use this to find where functions, variables, or patterns are used in the codebase."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    matches = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".venv")]
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{filepath}:{i}: {line.rstrip()}")
            except (OSError, IsADirectoryError):
                continue

    if not matches:
        return "No matches found"
    return "\n".join(matches[:100])


@tool
def glob(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern (e.g. '**/*.py' for all Python files).
    Returns a list of matching file paths.
    Use this to discover project structure and find files by name."""
    full_pattern = os.path.join(path, pattern)
    matches = glob_module.glob(full_pattern, recursive=True)
    skip = {".git", ".venv", "node_modules", "__pycache__"}
    filtered = []
    for m in matches:
        parts = m.split(os.sep)
        if any(p in skip or (p.startswith(".") and p != ".") for p in parts):
            continue
        if os.path.isfile(m):
            filtered.append(os.path.relpath(m, path))

    if not filtered:
        return "No files found"
    return "\n".join(sorted(filtered))


@tool
def run_command(command: str) -> str:
    """Run a shell command and return its output.
    Use this for tasks like running tests, installing packages, git operations, etc.
    Prefer grep and glob tools over shell grep/find commands."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {e}"


# --- RMS tools (Reason Maintenance System) ---
# These wrap rms_lib.api functions as LangGraph tools. The api functions
# return JSON-serializable dicts, so we just convert to str for the model.

import json

try:
    from rms_lib import api as rms_api
    _RMS_AVAILABLE = True
except ImportError:
    _RMS_AVAILABLE = False


@tool
def rms_status(db_path: str = "rms.db") -> str:
    """Show all beliefs in the RMS network with truth values (IN or OUT).
    Returns node IDs, text, truth values, and justification counts."""
    result = rms_api.get_status(db_path=db_path)
    return json.dumps(result, indent=2)


@tool
def rms_add(node_id: str, text: str, sl: str = "", unless: str = "",
            label: str = "", source: str = "", db_path: str = "rms.db") -> str:
    """Add a belief to the RMS network.
    Use sl for dependencies (comma-separated node IDs that must be IN).
    Use unless for outlist (comma-separated node IDs that must be OUT).
    Without sl or unless, the node is a premise (IN by default)."""
    result = rms_api.add_node(node_id, text, sl=sl, unless=unless,
                              label=label, source=source, db_path=db_path)
    return json.dumps(result)


@tool
def rms_retract(node_id: str, db_path: str = "rms.db") -> str:
    """Retract a belief and cascade to all dependents.
    Returns the list of all node IDs whose truth value changed."""
    result = rms_api.retract_node(node_id, db_path=db_path)
    return json.dumps(result)


@tool
def rms_assert(node_id: str, db_path: str = "rms.db") -> str:
    """Assert a belief (mark IN) and cascade restoration to dependents.
    Returns the list of all node IDs whose truth value changed."""
    result = rms_api.assert_node(node_id, db_path=db_path)
    return json.dumps(result)


@tool
def rms_explain(node_id: str, db_path: str = "rms.db") -> str:
    """Explain why a belief is IN or OUT by tracing its justification chain.
    Shows the full dependency path back to premises."""
    result = rms_api.explain_node(node_id, db_path=db_path)
    return json.dumps(result, indent=2)


@tool
def rms_show(node_id: str, db_path: str = "rms.db") -> str:
    """Show full details for a belief: text, status, source, justifications, dependents."""
    result = rms_api.show_node(node_id, db_path=db_path)
    return json.dumps(result, indent=2)


@tool
def rms_search(query: str, db_path: str = "rms.db") -> str:
    """Search beliefs by text or ID (case-insensitive substring match)."""
    result = rms_api.search(query, db_path=db_path)
    return json.dumps(result, indent=2)


@tool
def rms_trace(node_id: str, db_path: str = "rms.db") -> str:
    """Trace backward to find all premises (assumptions) a belief rests on."""
    result = rms_api.trace_assumptions(node_id, db_path=db_path)
    return json.dumps(result)


@tool
def rms_challenge(target_id: str, reason: str, db_path: str = "rms.db") -> str:
    """Challenge a belief. Creates a challenge node and the target goes OUT.
    Use when a reviewer or new evidence disputes a belief."""
    result = rms_api.challenge(target_id, reason, db_path=db_path)
    return json.dumps(result)


@tool
def rms_defend(target_id: str, challenge_id: str, reason: str,
               db_path: str = "rms.db") -> str:
    """Defend a belief against a challenge. Neutralises the challenge, target restored."""
    result = rms_api.defend(target_id, challenge_id, reason, db_path=db_path)
    return json.dumps(result)


@tool
def rms_nogood(node_ids: list[str], db_path: str = "rms.db") -> str:
    """Record a contradiction — these beliefs cannot all be true.
    Uses dependency-directed backtracking to find and retract the responsible premise."""
    result = rms_api.add_nogood(node_ids, db_path=db_path)
    return json.dumps(result)


@tool
def rms_compact(budget: int = 500, db_path: str = "rms.db") -> str:
    """Generate a token-budgeted summary of the belief network.
    Priority: nogoods first, then OUT nodes, then IN nodes by importance."""
    return rms_api.compact(budget=budget, db_path=db_path)


RMS_TOOLS = [
    rms_status, rms_add, rms_retract, rms_assert, rms_explain,
    rms_show, rms_search, rms_trace, rms_challenge, rms_defend,
    rms_nogood, rms_compact,
]

# All tools collected in a list — passed to the model and to ToolNode.
BASE_TOOLS = [read_file, write_file, edit_file, grep, glob, run_command]
ALL_TOOLS = BASE_TOOLS + (RMS_TOOLS if _RMS_AVAILABLE else [])
