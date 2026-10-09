# LangChain Multi-Tasking Agent (Parallel Actions & Batch Execution)

This agent uses **LangChain**, **LangGraph**, and **Google GenAI** (`langchain-google-genai`) to execute multiple tasks **at the same time** without prolonged long-chain loops.

---

## Key Features

1. **No Long Chain**: Quickly resolves tasks directly in minimal turns (no 20+ step loops).
2. **Parallel Actions at the Same Time**: Within a single prompt, the agent invokes multiple tools simultaneously (e.g., searching two topics or performing multiple math operations concurrently).
3. **LangChain Native Batch**: Executes multiple separate tasks simultaneously using `agent.batch()`.
4. **Tools**:
   - `calculator`: Defined via `@tool` for basic math (`add`, `subtract`, `multiply`, `divide`).
   - `websearch`: Defined via `@tool` with live DuckDuckGo and Wikipedia lookup.

---

## Project Files

```
gemini-calculator-agent/
├── langchain_agent.py     # Main LangChain multi-tasking agent (parallel & batch)
├── calculator_agent.py    # Zero-framework pure Python agent
├── calculator_tool.json   # Calculator JSON tool declaration
├── websearch_tool.json    # WebSearch JSON tool declaration
└── README.md              # Documentation and walkthrough
```

---

## How to Run the LangChain Agent

### 1. In Terminal (PowerShell):
```powershell
cd C:\Users\Admin\.gemini\antigravity\scratch\gemini-calculator-agent
python langchain_agent.py
```

### 2. If API Key is Not Set in the File:
```powershell
$env:GEMINI_API_KEY="AIzaSy..."
python langchain_agent.py
```

---

## Example Queries

### A. Multiple Tasks at the Same Time (Parallel Tool Calling)
```text
Enter your question: At the same time, calculate 15 * 6 and search who created Python.
```
*Output:*
```text
⚡ [PARALLEL TOOL CALLS DETECTED] LangChain invoking 2 tools at the same time:
   [1] calculator({"a": 15, "operation": "multiply", "b": 6})
   [2] websearch({"query": "who created Python"})
👁️  [TOOL OUTPUT] 90
👁️  [TOOL OUTPUT] Guido van Rossum...

🏁 [FINAL ANSWER]
* 15 * 6 = 90
* Python was created by Guido van Rossum.
```

### B. Parallel Math Operations
```text
Enter your question: Calculate 250 * 4 and 1000 / 25 at the same time.
```

### C. LangChain Native Batch Mode (Multiple Independent Tasks)
Type `batch` at the prompt:
```text
Enter your question (or 'batch' / 'exit'): batch
Batch tasks: What is 45 + 55?; What is 12 * 12?; Who founded Google?
```
*LangChain executes all tasks concurrently in parallel!*
