# Personal AI Knowledge Assistant

An evidence-grounded Retrieval-Augmented Generation (RAG) system that answers questions about professional portfolios and technical documents using retrieved evidence, source citations, and explicit abstention when evidence is insufficient.

## Architecture

The system supports both a standalone Streamlit application and a decoupled web architecture:
- **Streamlit Web Application (`streamlit_app.py`)**: Standalone local interface calling `src.pipeline.query_pipeline` directly. Does not require FastAPI or Next.js to be running.
- **Backend API (Python / FastAPI)**: Hosts the pipeline (`src.pipeline.query_pipeline`), which calls the **Google Gemini API with the File Search tool**. Google manages chunking, embeddings, indexing, retrieval and grounding; citations come from grounding metadata.
- **Frontend App (Next.js / TypeScript / React)**: Clean web interface designed for deployment to Vercel, querying the backend via REST endpoints.

```text
[ Browser (Next.js) ] ---> HTTP POST /query ---> [ FastAPI Backend (127.0.0.1:8000) ]
                                                            │
                                                            ▼
                                        [ Gemini API + File Search store ]
                                        (managed chunking, embeddings, retrieval)
```

> Configure `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`, then run `python ingest.py` to upload `data/documents/` into the File Search store. Sections below that mention Chroma/OpenAI describe the previous setup and are outdated.

---

## Prerequisites

1. **Python**: Python 3.12+ with virtual environment (`.venv`).
2. **Node.js**: Node.js v20+ or v24+ and `npm`.
3. **Google Gemini API Key**: Set `GEMINI_API_KEY` in root `.env` (and optionally `GEMINI_MODEL=gemini-2.5-flash`).

---

## Streamlit Web Interface (Active Interface)

The local Streamlit application directly calls `src.pipeline.query_pipeline`. It does not require running the FastAPI server or the Next.js frontend.

### 1. Installation
Install project dependencies including Streamlit:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Credentials
Configure `.env` in the project root:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
APP_ENV=development
```
*(Never commit `.env` or `.streamlit/secrets.toml` to version control).*

### 3. Ingest Documents into Google File Search Store
```powershell
.\.venv\Scripts\python.exe ingest.py
```
*(This creates the Gemini File Search store and uploads documents from `data/documents/`).*

### 4. Launch the Local Streamlit App
```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```
Or with virtual environment activated:
```bash
streamlit run streamlit_app.py
```
The application will open in your browser at `http://localhost:8501`.

**Features Provided**:
- **Title**: Personal AI Knowledge Assistant
- **Question Input**: Character-limited to 500 characters
- **Top-K Selector**: Slider from 1 to 10 (defaults to 5)
- **Explicit Submission**: Ask button inside a form prevents accidental queries
- **State Preservation**: Queries are submitted only on explicit Ask actions; results are cached in Streamlit session state across ordinary UI reruns
- **Status Display**: Distinct visual indicators for answered, partial, conflict, abstain, and error statuses
- **Evidence & Citations**: Direct display of grounded answer, missing information, grounding-verified citations, and expandable retrieved evidence chunks
- **Latency Timings**: Live display of generation and total latency breakdown

### 5. Streamlit Community Cloud Settings

For deployment on Streamlit Community Cloud (via the app settings dashboard or `.streamlit/secrets.toml`):

| Variable / Secret | Required in Cloud | Default | Description & Security Boundary |
|---|---|---|---|
| `APP_ENV` | **Yes** | `development` | Set to `production` for cloud deployments. Enforces reviewer authentication and fails closed if access tokens are missing. |
| `GEMINI_API_KEY` | **Yes** | None | **Confidential.** Google Gemini API key for File Search and generation. Populated from Streamlit secrets before config loading; never displayed or logged. |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Operational model identifier override. |
| `REVIEWER_ACCESS_TOKEN` | **Yes** in prod | None | **Confidential.** Shared secret token required in production (`APP_ENV=production`). Streamlit displays a password gate before allowing queries and fails closed if unset. |
| `STREAMLIT_DEMO_BOOTSTRAP` | Optional | `false` | When set to `true`, automatically uploads documents from `data/documents/` into the Gemini File Search store on startup if the store is missing or empty. |

---

## Backend Setup & Execution

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and set your key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
APP_ENV=development
```

### 3. Ingest Documents into Google File Search
```powershell
.\.venv\Scripts\python.exe ingest.py
```

### 4. Run the FastAPI Backend
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Readiness check: `http://127.0.0.1:8000/ready`

---

## Frontend Setup & Execution (Verified — Phase 5B)

### 1. Configuration
In `frontend/`, configure `frontend/.env.local` (or copy from `frontend/.env.example`):
```ini
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```
*(This is public client configuration, NOT a secret).*

### 2. Install & Run Development Server
```bash
cd frontend
npm install
npm run dev
```
*(On Windows PowerShell, use `npm.cmd` if script execution policies block `npm.ps1`).*

The frontend will start at `http://localhost:3000`.

### 3. Frontend Production Build Check
```bash
cd frontend
npm run build
```

---

## Testing (Verified)

Run the complete offline test suite (67 Python tests + 34 frontend tests, zero paid API calls):
```powershell
.\.venv\Scripts\pytest.exe -v
npm.cmd --prefix frontend test -- --run
```

---

## Evaluation Benchmark Runner

The project includes an evaluation runner for the fixed 20-question benchmark (`evaluation/questions.json`):

### Live Evaluation
```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate --live
.\.venv\Scripts\python.exe -m evaluation.evaluate --live --split development
.\.venv\Scripts\python.exe -m evaluation.evaluate --live --split holdout
```
*Requires `GEMINI_API_KEY` and an active File Search store populated with `ingest.py`. Results and metric summaries are saved to `evaluation/results/`.*

### Benchmark Split Discipline
- **Development Split (14 cases)**: Used for iterative development and system diagnostics.
- **Holdout Split (6 cases)**: Kept strictly untouched to measure out-of-sample generalization.
- **Manual Review Rubric (`evaluation/rubric.md`)**: Evaluates factual correctness, groundedness, citation semantic support, and behavioral alignment.

---

## Important Architectural Notes

- **CORS vs. Authentication**: CORS is configured strictly for `http://localhost:3000`. CORS is a browser cross-origin policy, not an authentication mechanism.
- **Top-K Validation**: The API validates `top_k` as an integer between 1 and 10; values outside this range are rejected with `HTTP 422 Unprocessable Entity`, not clamped.
- **Liveness vs. Readiness**: `GET /health` indicates the HTTP server is alive; `GET /ready` confirms the Gemini File Search store has active documents and that `GEMINI_API_KEY` is configured. `gemini_key_configured: true` indicates key presence only, not quota availability or credential validity.
- **Per-Process Rate Limiting**: In-memory sliding-window limits (15 req/min per IP) apply per Python worker process and reset on restart. Behind trusted proxies, set `TRUST_PROXY_HEADERS=true` to parse forwarded IPs; otherwise, direct socket peer IPs are used to prevent header spoofing.