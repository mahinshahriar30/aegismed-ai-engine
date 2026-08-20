```markdown
---
title: AegisMed AI Engine
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ AegisMed AI Engine

**AegisMed AI Engine** is a high-performance backend microservice built with **FastAPI**, **ChromaDB**, and **Google Gemini AI**. It powers real-time clinical AI assistance, medical document retrieval, and automated emergency corridor clearing.

---

## 🛠️ Project Structure

```text
├── .github/
│   └── workflows/
│       └── sync_to_hf.yml    # CI/CD pipeline for deployment
├── src/
│   ├── api.py                # FastAPI routes & endpoints
│   ├── rag.py                # RAG pipeline & ChromaDB logic
│   └── utils.py              # Helper utility functions
├── .env.example              # Template for environment variables
├── .gitignore                # Excludes secrets, venv, & temporary files
├── Dockerfile                # Containerization setup
├── README.md                 # Project documentation & configuration
├── app.py                    # Application entry point
└── requirements.txt          # Python dependencies

```

---

## ⚡ Quick Start & Local Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/mahinshahriar30/aegismed-ai-engine.git](https://github.com/mahinshahriar30/aegismed-ai-engine.git)
cd aegismed-ai-engine

```

### 2. Set Up Virtual Environment

* **Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

```


* **Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate

```



### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure Environment Variables

1. Copy the example environment file:
```bash
cp .env.example .env

```


2. Open `.env` and add your **Google Gemini API Key**:
```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8000

```



### 5. Run the Server

```bash
uvicorn src.api:app --reload --port 8000

```

Access the interactive API documentation (Swagger UI) at: **`http://127.0.0.1:8000/docs`**

---

## 🐳 Running with Docker

### Build Image

```bash
docker build -t aegismed-ai-engine .

```

### Run Container

```bash
docker run -d -p 8000:8000 --env-file .env aegismed-ai-engine

```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | System health check and status |
| `POST` | `/query` | Processes medical RAG queries with vector context |
| `POST` | `/clear-corridor` | Triggers emergency route clearing algorithm |

---

## ⚙️ Deployment & Continuous Integration

This repository includes a GitHub Action (`.github/workflows/sync_to_hf.yml`) that automatically syncs code pushes to Hugging Face Spaces or external hosting platforms on every update to the `main` branch.

```

```