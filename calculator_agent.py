"""
=============================================================================
ReAct AI Agent: Multi-Tasking & Long-Chain Execution (No Frameworks)
Zero External Frameworks - Pure Python & Google GenAI SDK
=============================================================================
Capabilities:
  1. COMPOSITE MULTI-TASKING : Decomposes complex queries with multiple tasks.
  2. PARALLEL ACTIONS        : Runs independent tool calls simultaneously
                               (via concurrent.futures ThreadPoolExecutor).
  3. DEEP LONG CHAINS        : Chains up to 25+ turns of Reason -> Action -> Observe.
  4. BATCH TASK RUNNER       : Executes queues of independent tasks with progress tracking.
=============================================================================
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
import concurrent.futures
from pathlib import Path

# Ensure UTF-8 output encoding for Windows command line / PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from google import genai
from google.genai import types

# =============================================================================
# 1. API CONFIGURATION
# =============================================================================
# Enter your Gemini API key from AI Studio below, or set the GEMINI_API_KEY environment variable.
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# Default model (fast and quota-friendly)
MODEL_NAME = "gemini-3.5-flash-lite"


# =============================================================================
# 2. TOOL DECLARATIONS (JSON FORMAT WITH REACT REASONING PROPERTY)
# =============================================================================

# Tool 1: Calculator Schema
CALCULATOR_TOOL_SCHEMA = {
    "name": "calculator",
    "description": (
        "Performs basic arithmetic operations: addition, subtraction, multiplication, and division. "
        "Use this tool whenever you need to compute or verify mathematical calculations."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "thought": {
                "type": "STRING",
                "description": (
                    "Reasoning (Thought): Explain why this calculation is required and what "
                    "exact values you are computing."
                ),
            },
            "operation": {
                "type": "STRING",
                "description": "The math operation to perform.",
                "enum": ["add", "subtract", "multiply", "divide"],
            },
            "a": {
                "type": "NUMBER",
                "description": "The first number (left operand).",
            },
            "b": {
                "type": "NUMBER",
                "description": "The second number (right operand).",
            },
        },
        "required": ["thought", "operation", "a", "b"],
    },
}

# Tool 2: Web Search Schema
WEBSEARCH_TOOL_SCHEMA = {
    "name": "websearch",
    "description": (
        "Searches the web for facts, figures, knowledge, or statistics. "
        "Use this tool whenever you need real-world knowledge to answer the question."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "thought": {
                "type": "STRING",
                "description": (
                    "Reasoning (Thought): Explain why this search is needed and what specific "
                    "facts or numbers you are looking for."
                ),
            },
            "query": {
                "type": "STRING",
                "description": "The search query keywords to look up on the web.",
            },
        },
        "required": ["thought", "query"],
    },
}


# =============================================================================
# 3. LOCAL TOOL IMPLEMENTATIONS (PYTHON)
# =============================================================================

def execute_calculator(operation: str, a: float, b: float, **kwargs) -> dict:
    """
    Locally executes the calculator operation requested by the LLM.
    Returns a dictionary result or an error message.
    """
    try:
        op = str(operation).strip().lower()
        val_a = float(a)
        val_b = float(b)

        if op in ("add", "+"):
            result = val_a + val_b
        elif op in ("subtract", "-"):
            result = val_a - val_b
        elif op in ("multiply", "*"):
            result = val_a * val_b
        elif op in ("divide", "/"):
            if val_b == 0:
                return {"error": "Division by zero is undefined."}
            result = val_a / val_b
        else:
            return {"error": f"Unsupported operation '{operation}'. Allowed: add, subtract, multiply, divide"}

        # Format float to int if integer value (e.g., 5.0 -> 5)
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return {"result": result}
    except Exception as exc:
        return {"error": f"Execution error: {str(exc)}"}


def execute_websearch(query: str, **kwargs) -> dict:
    """
    Locally searches the web using DuckDuckGo Instant Answer API and Wikipedia search API.
    Pure Python standard library (no third-party search frameworks required).
    """
    clean_query = str(query).strip()
    if not clean_query:
        return {"error": "Empty search query provided."}

    results = []

    # 1. DuckDuckGo Instant Answer API
    try:
        ddg_url = (
            f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_query)}"
            f"&format=json&no_html=1&skip_disambig=1"
        )
        req = urllib.request.Request(
            ddg_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", "Overview"),
                    "snippet": data.get("AbstractText"),
                    "url": data.get("AbstractURL", ""),
                })
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                        "snippet": topic.get("Text"),
                        "url": topic.get("FirstURL", ""),
                    })
            if results:
                return {"query": clean_query, "results": results}
    except Exception:
        pass

    # 2. Wikipedia Search API fallback
    try:
        wiki_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={urllib.parse.quote(clean_query)}&format=json&utf8=1"
        )
        req_wiki = urllib.request.Request(
            wiki_url,
            headers={"User-Agent": "GeminiReActAgent/1.0 (https://github.com/google; user@domain.com)"},
        )
        with urllib.request.urlopen(req_wiki, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            search_items = data.get("query", {}).get("search", [])
            for item in search_items[:3]:
                snippet = item.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
                results.append({
                    "title": item.get("title", ""),
                    "snippet": snippet,
                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(item.get('title', ''))}",
                })
            if results:
                return {"query": clean_query, "results": results}
    except Exception as exc:
        return {"error": f"Search request failed: {str(exc)}"}

    return {"query": clean_query, "results": [], "message": f"No search results found for '{clean_query}'"}


# Map tool names to their respective local Python functions
TOOL_REGISTRY = {
    "calculator": execute_calculator,
    "websearch": execute_websearch,
    "web_search": execute_websearch,  # Alias
}


def _dispatch_single_tool(call_item):
    """Executes a single tool call item and returns structured results."""
    idx, call = call_item
    tool_name = call.name
    tool_args = call.args or {}
    thought = tool_args.get("thought", "Reasoning step...")
    clean_args = {k: v for k, v in tool_args.items() if k != "thought"}

    if tool_name in TOOL_REGISTRY:
        tool_result = TOOL_REGISTRY[tool_name](**tool_args)
    else:
        tool_result = {"error": f"Tool '{tool_name}' is not registered."}

    return idx, tool_name, thought, clean_args, tool_result


# =============================================================================
# 4. MULTI-TASK & LONG-CHAIN REACT AGENT LOOP
# =============================================================================
def run_agent(
    prompt: str,
    client: genai.Client,
    model: str = MODEL_NAME,
    max_turns: int = 25,
    verbose: bool = True,
) -> str:
    """
    Runs the ReAct (Reason + Action) agentic loop with support for:
      - Deep multi-step sequential reasoning chains (up to max_turns).
      - Parallel action execution when the LLM triggers multiple actions simultaneously.
      - Composite multi-task decomposition.
    """
    calc_decl = types.FunctionDeclaration(**CALCULATOR_TOOL_SCHEMA)
    search_decl = types.FunctionDeclaration(**WEBSEARCH_TOOL_SCHEMA)
    agent_tool = types.Tool(function_declarations=[calc_decl, search_decl])

    react_system_instruction = (
        "You are an advanced multi-tasking AI Agent operating strictly on the ReAct "
        "(Reasoning + Acting) pattern.\n"
        "Your available tools are:\n"
        "  1. 'calculator': Performs basic math (+, -, *, /).\n"
        "  2. 'websearch': Searches the web for facts, figures, or knowledge.\n\n"
        "Core Operating Principles:\n"
        "1. MULTI-TASK HANDLING: When a prompt contains multiple tasks, break them down systematically. "
        "Work through them until all sub-tasks are fully completed.\n"
        "2. PARALLEL ACTIONS: Whenever independent searches or calculations can be done simultaneously, "
        "emit parallel tool calls in the same turn to save time.\n"
        "3. DEEP CHAINING: For dependent tasks, chain step-by-step (e.g. search -> calculate -> compare).\n"
        "4. SEARCH DISCIPLINE (STRICT): Search at most ONCE per topic/entity. Once an approximate or official "
        "number is retrieved (e.g. 14 million for Tokyo, 2.1 million for Paris), DO NOT search again for perfection. "
        "Immediately use those retrieved numbers for your calculations.\n"
        "5. REASONING: Always supply your step-by-step reasoning in the 'thought' parameter.\n"
        "6. FINAL ANSWER: When all sub-tasks are solved, synthesize a comprehensive, well-structured Final Answer."
    )

    config = types.GenerateContentConfig(
        tools=[agent_tool],
        temperature=0.0,  # Zero temperature for deterministic reasoning
        system_instruction=react_system_instruction,
    )

    # Initialize conversation history with user prompt
    conversation_history = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    if verbose:
        print("\n" + "=" * 70)
        print(f"[USER QUERY] {prompt}")
        print("=" * 70)

    for step in range(1, max_turns + 1):
        if verbose:
            print(f"\n--> [Chain Step {step}] Reasoning with LLM ({model})...")

        # LLM Reasoning Call with automatic retry backoff
        response = None
        for attempt in range(4):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=conversation_history,
                    config=config,
                )
                break
            except Exception as exc:
                err_text = str(exc)
                if ("429" in err_text or "RESOURCE_EXHAUSTED" in err_text) and attempt < 3:
                    wait_sec = 10 * (attempt + 1)
                    if verbose:
                        print(f"⏳ [Rate Limit Backoff] Hit 429 quota, waiting {wait_sec}s before retry...")
                    time.sleep(wait_sec)
                else:
                    raise exc

        # Check if the LLM decided to invoke one or more actions
        if response.function_calls:
            calls = response.function_calls
            is_parallel = len(calls) > 1

            if verbose and is_parallel:
                print(f"⚡ [PARALLEL ACTIONS DETECTED] Executing {len(calls)} actions simultaneously:")

            # 1. Record the model's response in conversation history
            conversation_history.append(response.candidates[0].content)

            # 2. Execute all tool calls concurrently using ThreadPoolExecutor
            indexed_calls = list(enumerate(calls))
            execution_results = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(calls), 5)) as executor:
                futures = [executor.submit(_dispatch_single_tool, item) for item in indexed_calls]
                for future in concurrent.futures.as_completed(futures):
                    execution_results.append(future.result())

            # Sort by original call index to preserve call order
            execution_results.sort(key=lambda x: x[0])

            tool_response_parts = []
            for idx, tool_name, thought, clean_args, tool_result in execution_results:
                prefix = f"⚡ [Action {idx + 1}/{len(calls)}]" if is_parallel else "⚡ [ACTION]"

                if verbose:
                    print(f"💭 [THOUGHT]    {thought}")
                    print(f"{prefix}     {tool_name}({json.dumps(clean_args)})")
                    res_str = json.dumps(tool_result, default=str)
                    if len(res_str) > 250:
                        res_str = res_str[:250] + "... [truncated]"
                    print(f"👁️  [OBSERVATION] {res_str}")

                # Prepare function response part
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response=tool_result,
                    )
                )

            # 3. Feed tool results back into conversation history
            conversation_history.append(
                types.Content(
                    role="user",
                    parts=tool_response_parts,
                )
            )

        else:
            # LLM completed all tasks and generated the final answer
            final_text = response.text if response.text else "No text response generated."
            if verbose:
                print("\n🏁 [FINAL ANSWER]")
                print("-" * 70)
                print(final_text)
                print("-" * 70)
            return final_text

    print("! [Warning] Reached maximum allowed turns without final answer.")
    return "Agent loop exceeded maximum turns."


# =============================================================================
# 5. BATCH TASK RUNNER
# =============================================================================
def run_batch_tasks(
    tasks: list[str],
    client: genai.Client,
    model: str = MODEL_NAME,
    max_turns: int = 25,
) -> dict:
    """
    Executes a list/queue of multiple independent tasks one after another,
    each running through its own full ReAct long chain.
    """
    print("\n" + "=" * 70)
    print(f"📦 BATCH EXECUTION: Running {len(tasks)} tasks")
    print("=" * 70)

    results = {}
    for i, task in enumerate(tasks, 1):
        print(f"\n======================================================================")
        print(f"📌 BATCH ITEM [{i}/{len(tasks)}]: {task}")
        print(f"======================================================================")

        answer = run_agent(task, client=client, model=model, max_turns=max_turns, verbose=True)
        results[task] = answer

    print("\n" + "=" * 70)
    print(f"✅ BATCH SUMMARY ({len(tasks)} tasks completed)")
    print("=" * 70)
    for i, (task, answer) in enumerate(results.items(), 1):
        summary_line = answer.strip().split("\n")[0]
        if len(summary_line) > 80:
            summary_line = summary_line[:80] + "..."
        print(f"[{i}] {task} -> {summary_line}")

    return results


# =============================================================================
# 6. CLI & INTERACTIVE SHELL
# =============================================================================
def get_client() -> genai.Client:
    """Initializes the Gemini client using the script API key or environment variable."""
    api_key = GEMINI_API_KEY.strip()

    # Fallback to environment variable if placeholder wasn't replaced
    if api_key == "YOUR_GEMINI_API_KEY_HERE" or not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        print("\n" + "!" * 70)
        print("ERROR: Gemini API Key Not Found!")
        print("!" * 70)
        print("Please configure your AI Studio API key in either of two ways:")
        print("1. Open 'calculator_agent.py' and set:")
        print("     GEMINI_API_KEY = \"AIzaSy...\"")
        print("2. Or set it in your terminal:")
        print("     $env:GEMINI_API_KEY=\"AIzaSy...\" (PowerShell)")
        print("     set GEMINI_API_KEY=\"AIzaSy...\" (CMD)")
        print("!" * 70 + "\n")
        sys.exit(1)

    return genai.Client(api_key=api_key)


def main():
    print("=" * 70)
    print("MULTI-TASK & LONG-CHAIN ReAct AGENT")
    print("Zero External Frameworks - Pure Python & Google GenAI SDK")
    print("=" * 70)
    print("Features:")
    print("  • Composite Multi-Tasking (breaks down complex multiple tasks)")
    print("  • Parallel Actions (concurrent searches/calculations at the same time)")
    print("  • Deep Long Chains (up to 25 reasoning steps)")
    print("  • Batch Runner Mode (type 'batch' to enter multiple tasks)")
    print("\nTry queries like:")
    print("  - Multi-Task at same time: 'At the same time, search for the population of")
    print("    Tokyo and Paris. Calculate their sum, difference, and percentage.'")
    print("  - Long Chain Math: '(45 * 18) -> add 320 -> divide by 4 -> subtract 85'")
    print("  - Batch Mode: Type 'batch'")
    print("=" * 70)

    client = get_client()

    while True:
        try:
            user_input = input("\nEnter your question (or 'batch' / 'exit'): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if user_input.lower() == "batch":
                print("\n--- BATCH MODE ---")
                print("Enter your tasks separated by semicolon ';' (e.g., Task 1; Task 2; Task 3):")
                batch_raw = input("Batch tasks: ").strip()
                tasks = [t.strip() for t in batch_raw.split(";") if t.strip()]
                if tasks:
                    run_batch_tasks(tasks, client=client)
                else:
                    print("No tasks entered.")
                continue

            run_agent(user_input, client=client)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError during execution: {e}")


if __name__ == "__main__":
    main()
