# 🧠 Personal AI Habit Mentor

A personalized AI-powered habit-building assistant using a structured **multi-agent reasoning pipeline**, **persistent long-term memory**, and optional **Gemini Flash-based coaching**.  
This project helps users create, track, and improve habits using behavioral science principles and stored learning from past interactions.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [Database Schema](#database-schema)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [API Usage](#api-usage)
- [Debugging](#debugging)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 📌 Overview

The **Personal AI Habit Mentor** is built using **FastAPI**, **SQLite**, and a modular multi-agent design.

It supports:

- Personalized conversation using **session context**
- Habit tracking and progress journaling
- Continuous “learning” via long-term memory
- Optional LLM reasoning (Gemini Flash), with a rule-based fallback when no API key is provided

---

## ✨ Features

| Feature | Status | Description |
|--------|--------|-------------|
| Multi-agent reasoning | ✅ | Sequential reasoning pipeline |
| Long-term memory | ✅ | SQLite-powered memory bank |
| Session-aware coaching | ✅ | Personalized across multiple conversations |
| Habit tracking | ✅ | Create and log habit progress |
| LLM optional | ✅ | Works without Gemini API |
| Extensible API | ⭐ | Can integrate a UI, notifications, or analytics |

---

## 🧱 Architecture

```
User
  │
  ▼
/message endpoint
  │
  ▼
OrchestratorAgent
  │
  ▼
═══════════════════════ Agent Chain (Sequential) ═══════════════════════

MemoryAgent → HabitAgent → NeuroscienceAgent → EvaluatorAgent → MemoryAgent(store)

═══════════════════════════════════════════════════════════════════════

  │
  ▼
Response back to user
```

---

## 🤖 Agent Pipeline

| Agent | Role |
|-------|------|
| **OrchestratorAgent** | Coordinates execution flow |
| **MemoryAgent** | Retrieves past user context and stores new insights |
| **HabitAgent** | Detects habit patterns, struggles, and goals |
| **NeuroscienceAgent** | Provides behavioral-science-backed reasoning |
| **EvaluatorAgent** | Produces the final polished assistant response |

---

## 🗃️ Database Schema (SQLite)

File: `data/habit_coach_multiagent.db`

| Table | Purpose |
|--------|---------|
| `sessions` | Tracks sessions tied to users |
| `habits` | Stores habit definitions |
| `habit_logs` | Logs user habit behavior |
| `memory_items` | Stores persistent personalized memory |

---

## ⚙ Installation

### 1️⃣ Create a virtual environment

```bash
python -m venv .venv
```

Activate:

```bash
# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

---

### 2️⃣ Install requirements

```bash
pip install fastapi uvicorn google-generativeai
```

---

### 3️⃣ Set Gemini (optional)

```bash
$env:GEMINI_API_KEY="YOUR_KEY"
```

---

## ▶ Running the Application

Start the API:

```bash
uvicorn main:app --reload
```

Open Swagger UI:

➡ http://127.0.0.1:8000/docs

---

## 🧪 API Usage

### Create a session

```json
POST /create-session
```

Response:

```json
{"session_id":"uuid"}
```

---

### Send a message to coach

```json
POST /message
{
  "session_id": "uuid",
  "user_id": "user123",
  "text": "I struggle with consistency."
}
```

---

### Add a habit

```json
POST /habits
{
  "user_id": "user123",
  "name": "Meditation",
  "description": "5 minutes daily"
}
```

---

### Retrieve memory

```http
GET /memory/user123
```

---

## 🛠 Debugging

Console logs show which agent executed:

```
[AGENT] HabitAgent
Input: "I skip workouts"
Output: {"intent":"improve consistency"}
```

---

## 🚀 Future Enhancements

| Feature | Priority |
|--------|----------|
| Vector search memory (FAISS) | 🔥 |
| Mobile app integration | ⭐ |
| Habit streak gamification | ⭐ |
| Real-time notifications | ⭐ |
| Parallel agent execution mode | Optional |

---

## 🧾 License

Free for **research and personal development** use.

---
