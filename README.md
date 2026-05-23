# 🧠 RAG Intelligence

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from **PDFs** and **web pages** using LangChain + Groq + ChromaDB — with a sleek dark web UI.

---

## ✨ Features

- 📄 Upload any PDF and ask questions about it
- 🌐 Paste any webpage URL and query its content
- 🤖 Powered by `llama-3.3-70b-versatile` via Groq
- 🔍 MMR retrieval for diverse, relevant context
- 💾 Persistent ChromaDB vector store
- ⚡ FastAPI backend + clean vanilla JS frontend

---

## 📁 Project Structure

```
rag-project/
├── index.html          # Web UI
├── main.py             # Terminal chatbot (PDF + URL)
├── server.py           # FastAPI backend
├── createdatabase.py   # One-time PDF ingestion script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-project.git
cd rag-project
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create `.env` file
```
GROQ_API_KEY=your_groq_api_key_here
```
> Get your free key at [console.groq.com](https://console.groq.com)

---

## 💻 Running the App

### Start the backend
```bash
uvicorn server:app --reload
```
Server runs at `http://localhost:8000`

### Open the UI
Open `index.html` in your browser — no build step needed.

---

## 🖥️ Terminal Mode
```bash
python main.py
```
Choose PDF or URL, then chat in your terminal.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/status` | Check if document is loaded |
| POST | `/upload-pdf` | Upload a PDF file |
| POST | `/ingest-url` | Ingest a webpage by URL |
| POST | `/chat` | Ask a question |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | `llama-3.3-70b-versatile` via Groq |
| Embeddings | `BAAI/bge-large-en-v1.5` (HuggingFace) |
| Vector DB | ChromaDB |
| Framework | LangChain |
| Backend | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |

---

## ⚠️ Important

- Never commit `.env` — it's in `.gitignore`
- Never commit `.venv/` or `chroma-db/` — also gitignored
- ChromaDB is local only; for production use a hosted vector DB

---

## 📜 License
MIT

# RAG-Intelliegence