#!/home/ubuntu/llm_pipeline_env/bin/python3
"""
AI-CodeCompass Observer — Autonomous System Monitoring
Location: /home/ubuntu/AI-CodeCompass/observer.py

Roadmap:
  ✅ Structural Intelligence (ParserAgent)
  ✅ Semantic Memory (VectorStore/FAISS)
  ✅ Human-in-the-Loop: Persisted across sessions
  ✅ Persistent Checkpointer: SQLite memory
"""

import os
import sys
import datetime
import sqlite3
import logging
from typing import TypedDict, List, Optional, Annotated, Dict, Any
import operator

# Add the parent directory to sys.path so we can import AI-CodeCompass agents
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Local AI-CodeCompass imports
try:
    from agents.parser_agent import ParserAgent
    from core.vector_store import CodeVectorStore
    AI_COMPASS_READY = True
except ImportError:
    AI_COMPASS_READY = False

# ── Config ────────────────────────────────────────────────────────────────────
AUTO_MODE   = "--auto" in sys.argv
DIGEST_PATH = "/home/ubuntu/PROJECT_DIGEST.md"
DB_PATH     = "/home/ubuntu/observer_memory.db"
VECTOR_DB   = "/home/ubuntu/observer_vectors"
THREAD_ID   = "observer-v2"

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="observer",
    base_url="http://127.0.0.1:1234/v1",
    api_key="not-needed",
    temperature=0.3,
    max_tokens=1000,
    timeout=90,
)

# ── Files to explain (label + path shown to user) ─────────────────────────────
FILES = [
    {
        "label": "Implementation Plan",
        "path": (
            "/home/ubuntu/.gemini/antigravity/brain/"
            "d4d86681-693f-4821-9737-ba80d63894a0/implementation_plan.md"
        ),
    },
    {
        "label": "LangGraph AI Script",
        "path": "/home/ubuntu/observer_crew.py",
    },
    {
        "label": "Filesystem Scanner",
        "path": "/home/ubuntu/observer_scan.sh",
    },
    {
        "label": "AI Server Launcher",
        "path": "/home/ubuntu/start_observer_brain.sh",
    },
    {
        "label": "Observer Security State",
        "path": "/home/ubuntu/UBUNTU_STATE.md",
    },
]

# ── ANSI colours ──────────────────────────────────────────────────────────────
R  = "\033[0m"       # reset
B  = "\033[1m"       # bold
CY = "\033[96m"      # cyan
GR = "\033[92m"      # green
YL = "\033[93m"      # yellow
MA = "\033[95m"      # magenta
DM = "\033[2m"       # dim


# ── Graph State ───────────────────────────────────────────────────────────────
class ObserverState(TypedDict):
    # Accumulated lists
    file_contents:  Annotated[List[dict], operator.add]
    parsed_data:    Dict[str, Any]  # [NEW] Results from ParserAgent
    explanations:   Annotated[List[dict], operator.add]
    human_notes:    str          # User's extra context added during HITL pause
    digest_written: bool
    run_timestamp:  str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _divider(title: str = "", colour: str = CY) -> None:
    w = 64
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"{colour}{B}{'─'*pad} {title} {'─'*pad}{R}")
    else:
        print(f"{colour}{'─'*w}{R}")


def _human_pause(prompt: str, allow_notes: bool = False) -> str:
    """Pause for user input. Returns any notes typed, or '' if auto mode."""
    if AUTO_MODE:
        return ""
    _divider("⏸  HUMAN-IN-THE-LOOP", YL)
    print(f"{YL}{B}{prompt}{R}")
    if allow_notes:
        print(f"{DM}  (Type extra context to add, or press Enter to skip){R}")
        notes = input(f"{YL}  Your notes: {R}").strip()
    else:
        input(f"{YL}  Press Enter to continue… {R}")
        notes = ""
    _divider(colour=YL)
    return notes


# ── Persistent memory via SQLite ──────────────────────────────────────────────
def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT,
            event     TEXT,
            detail    TEXT
        )
    """)
    conn.commit()
    return conn


def _log_event(conn: sqlite3.Connection, event: str, detail: str = "") -> None:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    conn.execute("INSERT INTO session_log (ts, event, detail) VALUES (?,?,?)",
                 (ts, event, detail))
    conn.commit()


def _recall_last_run(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT ts FROM session_log WHERE event='digest_written' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


# ── NODE 1: Read Files ─────────────────────────────────────────────────────────
def read_files(state: ObserverState) -> dict:
    _divider("NODE 1 / 3 — Reading Project Files", CY)
    contents = []
    for item in FILES:
        path  = item["path"]
        label = item["label"]
        exists = os.path.exists(path)
        size   = f"{os.path.getsize(path):,} bytes" if exists else "—"
        status = f"{GR}✅{R}" if exists else f"{YL}⚠️  not found{R}"

        print(f"  {status}  {B}{label}{R}")
        print(f"       {DM}📁 {path}{R}")
        print(f"       {DM}   size: {size}{R}")

        if exists:
            try:
                with open(path) as f:
                    text = f.read(2500)
                if len(text) == 2500:
                    text += "\n...[truncated]"
            except Exception as e:
                text = f"[Error reading: {e}]"
        else:
            text = "[File not found]"

        contents.append({"label": label, "path": path, "content": text})
        print()

    return {
        "file_contents": contents,
        "run_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


# ── NODE 2: Human Review (HITL) ───────────────────────────────────────────────
def human_review(state: ObserverState) -> dict:
    _divider("NODE 2 / 3 — Human Review", MA)
    print(f"{MA}  Files queued for AI explanation:{R}\n")
    for i, item in enumerate(state["file_contents"], 1):
        flag = f"{GR}✅{R}" if item["content"] != "[File not found]" else f"{YL}⚠️ {R}"
        print(f"  {flag} [{i}] {B}{item['label']}{R}")
        print(f"        {DM}📁 {item['path']}{R}")
    print()

    notes = _human_pause(
        "Review the file list above. Add any extra context for the AI? "
        "(e.g. 'This project monitors my server for OpenClaw agents.')",
        allow_notes=True,
    )
    return {"human_notes": notes or ""}


# ── NODE 3: Structural & AI Analysis ──────────────────────────────────────────
def analyse(state: ObserverState) -> dict:
    _divider("NODE 3 / 3 — Structural Intelligence", GR)

    # 1. Structural Parsing (The "Deep Eyes")
    parser = ParserAgent()
    code_files_map = {item["path"]: item["content"] for item in state["file_contents"] if item["content"] != "[File not found]"}
    
    print(f"  {DM}🔍 Analyzing code structure with ParserAgent...{R}")
    parsed_results = parser.parse_files(code_files_map)
    
    # 2. AI Summarization with Structural Context
    extra = f"\n\nUser context: {state['human_notes']}" if state["human_notes"] else ""
    system = SystemMessage(content=(
        "You are an Autonomous System Observer. "
        "Analyze the provided file. Use the 'Structural Insights' to explain exactly what logic changed.\n\n"
        "**PURPOSE:** One clear sentence on what this file does.\n"
        "**STRUCTURAL INSIGHTS:** List key functions, classes, and important logic found.\n"
        "**SECURITY RISK:** High/Med/Low and why.\n"
        "**REMEMBER:** Key ports, file paths, or credentials.\n\n"
        "Be professional and technically precise." + extra
    ))

    explanations = []
    for item in state["file_contents"]:
        label   = item["label"]
        path    = item["path"]
        content = item["content"]

        print(f"\n  {GR}{B}🤖 Explaining:{R} {label}")
        
        # Get structural data for this file
        struct = parsed_results.get(path)
        struct_summary = ""
        if struct and not struct.error:
            funcs = [f["name"] for f in struct.functions]
            classes = [c["name"] for c in struct.classes]
            struct_summary = f"Detected Functions: {', '.join(funcs[:5])}\nDetected Classes: {', '.join(classes[:5])}"
            print(f"     {DM}AST: Found {len(funcs)} functions, {len(classes)} classes.{R}")

        if content == "[File not found]":
            explanation = "File was not found on disk."
        else:
            human = HumanMessage(content=(
                f"Explain '{label}' (`{path}`).\n\n"
                f"Structural Data:\n{struct_summary}\n\n"
                f"Source Snippet:\n```\n{content}\n```"
            ))
            response = llm.invoke([system, human])
            explanation = response.content.strip()

        # Print explanation indented
        for line in explanation.split("\n"):
            print(f"     {line}")

        explanations.append({"label": label, "path": path, "explanation": explanation})

    return {
        "explanations": explanations,
        "parsed_data": {path: (vars(res) if hasattr(res, "__dict__") else res) for path, res in parsed_results.items()}
    }


# ── NODE 4: Human Approval + Write Digest + Semantic Memory ──────────────────
def write_digest(state: ObserverState) -> dict:
    _divider("Writing PROJECT_DIGEST.md & Committing to Memory", GR)

    notes = _human_pause(
        "The AI has finished its analysis above. "
        "Press Enter to save to PROJECT_DIGEST.md and update semantic memory.",
    )

    ts = state.get("run_timestamp", datetime.datetime.now().isoformat())

    # 1. Build the digest sections
    sections = []
    for item in state["explanations"]:
        label = item["label"]
        path  = item["path"]
        expl  = item["explanation"]
        sections.append(
            f"## 📄 {label}\n\n"
            f"> 📁 **File path:** `{path}`\n\n"
            f"{expl}\n"
        )

    body = "\n---\n\n".join(sections)

    # 2. Add structural summary if available
    struct_meta = ""
    if state.get("parsed_data"):
        count = len(state["parsed_data"])
        struct_meta = f"\n> 🧠 **Structural Intelligence:** Analyzed {count} files for logic changes."

    # Human notes section
    notes_section = ""
    if state.get("human_notes"):
        notes_section = (
            f"\n---\n\n## 💬 Your Context Notes\n\n"
            f"> {state['human_notes']}\n"
        )

    full = f"""# 🗂️ AI-CodeCompass Observer — Project Memory Digest
*Auto-generated on {ts} by `/home/ubuntu/AI-CodeCompass/observer.py`*

> Run `python3 /home/ubuntu/AI-CodeCompass/observer.py` anytime to refresh.
> Memory DB: `{DB_PATH}`
> Vector DB: `{VECTOR_DB}`
{struct_meta}

---

{body}{notes_section}
---

## ⚡ Quick Reference

| Command | What it does | Location |
|---------|-------------|----------|
| `python3 observer.py` | Refresh this digest | `/home/ubuntu/AI-CodeCompass/observer.py` |
| `python3 observer.py --auto` | Skip human pauses | — |
| `/home/ubuntu/AI-CodeCompass/scripts/scanner.sh` | Run security scan | — |
| `sqlite3 {DB_PATH} "SELECT * FROM session_log"` | View session history | — |
"""

    with open(DIGEST_PATH, "w") as f:
        f.write(full)

    print(f"\n  {GR}{B}✅ Saved →{R} {DIGEST_PATH}")

    # 3. Persistent Semantic Memory (The "Giant Brain")
    if AI_COMPASS_READY:
        try:
            print(f"  {DM}🧠 Committing security context to FAISS memory...{R}")
            store = CodeVectorStore(device="cpu")
            if os.path.exists(VECTOR_DB):
                store.load(VECTOR_DB)
            
            from langchain_core.documents import Document
            # We index each file's explanation as a searchable document
            docs_to_index = []
            for item in state["explanations"]:
                docs_to_index.append(Document(
                    page_content=item["explanation"],
                    metadata={"source": item["path"], "label": item["label"], "ts": ts}
                ))
            
            store.add_documents(docs_to_index)
            store.save(VECTOR_DB)
            print(f"  {GR}✅ Semantic memory updated.{R}")
        except Exception as e:
            print(f"  {YL}⚠️ Could not update semantic memory: {e}{R}")

    # 4. Log to persistent SQLite DB
    conn = _init_db()
    _log_event(conn, "digest_written", DIGEST_PATH)
    _log_event(conn, "files_explained",
               ", ".join(i["label"] for i in state["explanations"]))
    conn.close()

    return {"digest_written": True}


# ── Build + compile the graph ─────────────────────────────────────────────────
def build_graph():
    g = StateGraph(ObserverState)

    g.add_node("read_files",    read_files)
    g.add_node("human_review",  human_review)
    g.add_node("analyse",       analyse)
    g.add_node("write_digest",  write_digest)

    g.add_edge(START,           "read_files")
    g.add_edge("read_files",    "human_review")
    g.add_edge("human_review",  "analyse")
    g.add_edge("analyse",       "write_digest")
    g.add_edge("write_digest",  END)

    # In-memory checkpointer (swap for SqliteSaver for full persistence)
    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _divider("Ubuntu Observer — Project Memory (LangGraph)", CY)
    
    # Check for Mode
    query_text = ""
    interactive_mode = False
    for i, arg in enumerate(sys.argv):
        if arg in ["--interactive", "-i"]:
            interactive_mode = True
            break
        if arg in ["--query", "-q"] and i + 1 < len(sys.argv):
            query_text = " ".join(sys.argv[i+1:])
            break

    if interactive_mode:
        from observer_chat import start_chat
        start_chat()
        sys.exit(0)

    if query_text:
        print(f"{MA}{B}🕵️‍♂️ Mode: Observer Consultation (Adaptive RAG){R}")
        print(f"{DM}   Consulting memory for: \"{query_text}\"{R}\n")
        
        # Initialize memory and logic
        store = CodeVectorStore(device="cpu")
        if os.path.exists(VECTOR_DB):
            store.load(VECTOR_DB)
        else:
            print(f"{YL}⚠️  Semantic memory not found. Run a scan first.{R}")
            sys.exit(1)

        from engine_adapter import AdaptiveObserverConsultant
        consultant = AdaptiveObserverConsultant(llm, store)
        
        # Run reasoning loop
        result = consultant.ask(query_text)
        
        # Display Intermediate Steps (The "Thinking")
        print(f"{DM}── Agent Reasoning ──{R}")
        for step in result.get("intermediate_steps", []):
            print(f"  {DM}{step}{R}")
        _divider(colour=DM)
        
        # Final Answer
        print(f"\n{GR}{B}🎯 Observer Response:{R}\n")
        print(result.get("final_answer", "I could not formulate an answer."))
        print(f"\n{DM}── Consultation Complete ──{R}\n")
        sys.exit(0)

    # ... otherwise proceed to normal Scan Mode ...
    print(f"{CY}{B}  Script  :{R} /home/ubuntu/observer.py")
    print(f"{CY}{B}  Model   :{R} Llama 3.2 1B Instruct  (llama.cpp · port 1234)")
    print(f"{CY}{B}  Output  :{R} {DIGEST_PATH}")
    print(f"{CY}{B}  Memory  :{R} {DB_PATH}")
    print(f"{CY}{B}  Mode    :{R} {'AUTO (no pauses)' if AUTO_MODE else 'Interactive (Human-in-the-Loop)'}")
    _divider(colour=CY)

    # Check last run from persistent memory
    conn = _init_db()
    last = _recall_last_run(conn)
    if last:
        print(f"\n{DM}  📖 Last digest generated: {last}{R}")
    conn.close()

    app = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    initial = {
        "file_contents":  [],
        "explanations":   [],
        "human_notes":    "",
        "digest_written": False,
        "run_timestamp":  "",
    }

    final = app.invoke(initial, config=config)

    _divider(colour=CY)
    if final.get("digest_written"):
        print(f"\n{GR}{B}✨ Done!{R}  Open {DIGEST_PATH} to read your project summary.")
        print(f"{DM}   Session logged to {DB_PATH}{R}\n")
    else:
        print(f"\n{YL}⚠️  Digest was not written.{R}\n")
