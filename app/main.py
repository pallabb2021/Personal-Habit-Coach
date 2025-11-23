"""
main.py - Multi-Agent Habit Coach (Sequential flow, Gemini Flash optional)
Single-file FastAPI app:
  - SQLite persistence (user-level Memory Bank)
  - Orchestrator -> Memory -> Habit -> Neuroscience -> Evaluator -> Memory store
  - Uses google.generativeai (Gemini) if GEMINI_API_KEY is set
"""

import os
import sqlite3
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try to import Gemini client (google-generativeai). If missing or not configured, fall back to deterministic responses.
USE_GEMINI = False
try:
    import google.generativeai as genai
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        USE_GEMINI = True
except Exception:
    USE_GEMINI = False


# ---------------------------
# Database (SQLite) setup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "habit_coach_multiagent.db")

# Use sqlite3 directly (lightweight & portable)
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize DB and tables if not present
def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS habit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            habit_id INTEGER NOT NULL,
            user_id TEXT,
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            key TEXT,
            value TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# Pydantic models (request/response)
# ---------------------------
class CreateHabit(BaseModel):
    user_id: str
    name: str
    description: Optional[str] = ""

class LogHabit(BaseModel):
    user_id: str
    habit_id: int
    note: Optional[str] = ""

class AskRequest(BaseModel):
    session_id: str
    user_id: str
    question: str

class MessageRequest(BaseModel):
    session_id: str
    user_id: str
    text: str

# ---------------------------
# MemoryBank Service
# ---------------------------
class MemoryBankService:
    def __init__(self, conn):
        self.conn = conn

    def add_memory(self, user_id: str, key: str, value: str):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO memory_items (user_id, key, value) VALUES (?, ?, ?)",
            (user_id, key, value),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_memories(self, user_id: str, limit: int = 50):
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT id, key, value, created_at FROM memory_items WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def find_similar(self, user_id: str, query: str, top_k: int = 5):
        # Simple keyword-based scoring (fast fallback). Later: integrate embeddings for semantic search.
        q = query.lower()
        items = self.get_memories(user_id, limit=200)
        scored = []
        for it in items:
            score = 0
            key = (it.get("key") or "").lower()
            val = (it.get("value") or "").lower()
            if q in key:
                score += 2
            if q in val:
                score += 1
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:top_k] if s[0] > 0]

# ---------------------------
# Session manager
# ---------------------------
class SessionManager:
    def __init__(self, conn):
        self.conn = conn

    def create_session(self, user_id: Optional[str] = None) -> str:
        sid = str(uuid4())
        cur = self.conn.cursor()
        cur.execute("INSERT INTO sessions (session_id, user_id) VALUES (?, ?)", (sid, user_id))
        self.conn.commit()
        return sid

    def session_exists(self, session_id: str) -> bool:
        cur = self.conn.cursor()
        row = cur.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return bool(row)

# ---------------------------
# Agent Implementations (Sequential)
# ---------------------------

# Logging for agents
def log_agent(agent_name, input_text, output_text):
    print("\n" + "="*60)
    print(f"[AGENT CALLED] {agent_name}")
    print("-"*60)
    print(f"Input:\n{input_text}")
    print("-"*60)
    print(f"Output:\n{output_text}")
    print("="*60 + "\n")


# MemoryAgent: retrieves & stores user memories
class MemoryAgent:
    def __init__(self, conn):
        self.service = MemoryBankService(conn)

    def retrieve(self, user_id: str, question: str) -> List[Dict[str, Any]]:
        # returns up to top-k similar memory items (keyword-based)
        return self.service.find_similar(user_id, question, top_k=5)

    def store(self, user_id: str, key: str, value: str):
        return self.service.add_memory(user_id, key, value)

# HabitAgent: extracts habit structure from user text
class HabitAgent:
    def __init__(self):
        pass

    def extract(self, text: str) -> Dict[str, Any]:
        # Lightweight heuristic extraction. For better extraction, call LLM.
        # We'll use a small prompt if Gemini is available to structure habits.
        if USE_GEMINI:
            try:
                model = genai.GenerativeModel("gemini-2.0-flash-lite")
                prompt = (
                    "Extract habits and issues from the following user message.\n"
                    "Return JSON with keys: detected_habits (list), issues (list), intent (string).\n\n"
                    f"Message: {text}\n\n"
                    "Return only valid JSON."
                )
                resp = model.generate_content(prompt)
                # try to parse JSON from model (best-effort)
                raw = resp.text.strip()
                # model may include extra text — attempt to find first JSON block
                json_start = raw.find("{")
                if json_start != -1:
                    raw_json = raw[json_start:]
                    parsed = json.loads(raw_json)
                    return parsed
            except Exception:
                pass

        # Fallback simple heuristic:
        lowered = text.lower()
        detected = []
        issues = []
        for word in ["exercise", "walk", "run", "meditat", "sleep", "reading", "journal", "exercise"]:
            if word in lowered:
                detected.append(word)
        if "procrastin" in lowered or "delay" in lowered:
            issues.append("procrastination")
        if not detected:
            detected = ["general self-care"]
        return {"detected_habits": detected, "issues": issues, "intent": "clarify"}

# NeuroscienceAgent: gives brain-based analysis of habits + memory
class NeuroscienceAgent:
    def __init__(self):
        pass

    def analyze(self, habit_struct: Dict[str, Any], memories: List[Dict[str, Any]]) -> str:
        # Build a prompt summarizing habits + memories
        mem_text = "\n".join([f"- {m['key']}: {m['value']}" for m in memories]) or "(no memories)"
        prompt = (
            "You are a concise neuroscience-aware habit coach. Given the detected habits and the user's memory notes, "
            "produce: (a) a short explanation of likely brain-behavior drivers, (b) 2 quick interventions grounded in behavior change science.\n\n"
            f"Habits: {habit_struct}\n\nMemories:\n{mem_text}\n\nRespond in plain text."
        )

        if USE_GEMINI:
            try:
                model = genai.GenerativeModel("gemini-2.0-flash-lite")
                resp = model.generate_content(prompt)
                return resp.text.strip()
            except Exception:
                pass

        # fallback deterministic answer:
        parts = []
        parts.append("Possible drivers: habit loop (cue-routine-reward), attention & motivation fluctuations, and low immediate reward.")
        parts.append("Interventions:\n1) Reduce friction and set micro-goals (e.g. 2 minutes). 2) Pair habit with an existing daily cue (after breakfast).")
        return "\n\n".join(parts)

# EvaluatorAgent: polishes & synthesizes final user-facing reply
class EvaluatorAgent:
    def __init__(self):
        pass

    def refine(self, question: str, habit_struct: Dict[str, Any], neuro_text: str, memories: List[Dict[str, Any]]) -> str:
        # Compose a concise human-friendly reply, optionally using Gemini to rephrase.
        mem_preview = "\n".join([f"- {m['key']}: {m['value']}" for m in memories]) or "(no memories)"
        base = (
            f"Question: {question}\n\n"
            f"Detected Habits: {habit_struct}\n\n"
            f"Memory notes:\n{mem_preview}\n\n"
            f"Neuroscience insight:\n{neuro_text}\n\n"
            "Summary (3 steps) and 1 short motivational tip:"
        )

        if USE_GEMINI:
            try:
                model = genai.GenerativeModel("gemini-2.0-flash-lite")
                prompt = base + "\n\nPlease output a concise 3-step plan and one motivational sentence."
                resp = model.generate_content(prompt)
                return resp.text.strip()
            except Exception:
                pass

        # deterministic formatting fallback:
        steps = [
            "Start with a tiny version of the habit (2 minutes).",
            "Attach it to an existing daily routine as a cue.",
            "Log immediately and celebrate small wins."
        ]
        return (
            f"{neuro_text}\n\nSuggested steps:\n1) {steps[0]}\n2) {steps[1]}\n3) {steps[2]}\n\n"
            "Motivational tip: Focus on tiny progress consistently — it's compounding."
        )

# OrchestratorAgent: runs the sequential pipeline
class OrchestratorAgent:
    def __init__(self, conn):
        self.conn = conn
        self.memory_agent = MemoryAgent(conn)
        self.habit_agent = HabitAgent()
        self.neuro_agent = NeuroscienceAgent()
        self.eval_agent = EvaluatorAgent()

    def handle(self, session_id: str, user_id: str, text: str) -> Dict[str, Any]:
        # 0. validate session (caller should ensure session exists)
        # 1. retrieve memory relevant to the question
        memories = self.memory_agent.retrieve(user_id, text)
        log_agent("MemoryAgent", text, memories)
        # 2. extract habit structure
        habit_struct = self.habit_agent.extract(text)
        log_agent("HabitAgent", text, habit_struct)
        # 3. neuroscience analysis
        neuro_text = self.neuro_agent.analyze(habit_struct, memories)
        log_agent("NeuroAgent", text, neuro_text)
        # 4. evaluator creates final answer
        final = self.eval_agent.refine(text, habit_struct, neuro_text, memories)
        # 5. store a brief memory note summarizing the interaction
        memory_summary_key = "last_coach_summary"
        summary_value = f"Q: {text} | Summary: {final[:400]}"  # trim for memory
        self.memory_agent.store(user_id, memory_summary_key, summary_value)
        return {"advice": final, "stored_memory_preview": summary_value}

# ---------------------------
# FastAPI app & endpoints
# ---------------------------
app = FastAPI(title="Multi-Agent Personal Habit Coach")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# helper to open DB conn per request (simple approach)
def get_conn():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()

# Session manager instance (shared)
_conn_for_sessions = get_connection()
session_mgr = SessionManager(_conn_for_sessions)

@app.post("/create-session")
def create_session(user_id: Optional[str] = None):
    sid = session_mgr.create_session(user_id)
    return {"session_id": sid}

@app.post("/habits", summary="Create a habit")
def create_habit(payload: CreateHabit):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO habits (user_id, name, description) VALUES (?, ?, ?)",
        (payload.user_id, payload.name, payload.description),
    )
    conn.commit()
    habit_id = cur.lastrowid
    conn.close()
    return {"habit_id": habit_id, "message": "Habit created"}

@app.get("/habits/{user_id}", summary="List habits for a user")
def list_habits(user_id: str):
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute("SELECT id, name, description, created_at FROM habits WHERE user_id = ?", (user_id,)).fetchall()
    conn.close()
    return {"habits": [dict(r) for r in rows]}

@app.post("/habits/{habit_id}/log", summary="Log habit activity")
def log_habit(habit_id: int, payload: LogHabit):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM habits WHERE id = ?", (habit_id,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Habit not found")
    cur.execute("INSERT INTO habit_logs (habit_id, user_id, note) VALUES (?, ?, ?)", (habit_id, payload.user_id, payload.note))
    conn.commit()
    log_id = cur.lastrowid
    conn.close()
    return {"log_id": log_id, "message": "Logged"}

@app.get("/logs/{user_id}", summary="Get logs for a user")
def get_logs(user_id: str):
    conn = get_connection()
    rows = conn.execute("SELECT id, habit_id, note, created_at FROM habit_logs WHERE user_id = ? ORDER BY created_at DESC", (user_id,)).fetchall()
    conn.close()
    return {"logs": [dict(r) for r in rows]}

@app.post("/memory/add", summary="Add a memory item for a user")
def add_memory(user_id: str, key: str, value: str):
    conn = get_connection()
    service = MemoryBankService(conn)
    mem_id = service.add_memory(user_id, key, value)
    conn.close()
    return {"memory_id": mem_id, "message": "Memory added"}

@app.get("/memory/{user_id}", summary="Get memory items for a user")
def get_memory(user_id: str):
    conn = get_connection()
    service = MemoryBankService(conn)
    items = service.get_memories(user_id)
    conn.close()
    return {"memories": items}

@app.post("/message", summary="Send a message to the multi-agent orchestrator")
def message(req: MessageRequest):
    # Validate session
    if not session_mgr.session_exists(req.session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id. Create session with /create-session first.")
    conn = get_connection()
    orchestrator = OrchestratorAgent(conn)
    result = orchestrator.handle(req.session_id, req.user_id, req.text)
    conn.close()
    return result


# ---------------------------
# Quick run
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting Multi-Agent Habit Coach (sequential) on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
