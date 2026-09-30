# fNIRS Fuzzy Rule Explainer

A dissertation project for training and evaluating an explainable fuzzy-rule classifier on fNIRS working-memory data, then presenting its rules and performance through an interactive web application. The web app combines retrieved model rules with an Ollama-hosted language model to explain results in plain language; it also provides channel visualizations and optional live literature lookup.

## Project Overview

The project has two related parts:

1. **Model and analysis pipeline**: `MainFold.py` trains an ExFuzzy classifier and evaluates it with subject-grouped cross-validation. It writes performance summaries and structured rules used by the explainer. `build_channel_atlas.py` builds channel-coordinate data for the visualizations.
2. **Interactive explainer**: a FastAPI service (`backend.py`) loads the rule and channel data; the React/Vite application in `react-frontend/react-frontend/` talks to it over HTTP. The service uses Ollama for generated explanations and can optionally query OpenAlex for additional literature.

The repository also includes a Streamlit dashboard (`dashboard_app.py`), visualization utilities, analysis and latency scripts, and the data/results used by the project. The checked-in rule and atlas files let you run the explainer without retraining the model.

## Requirements

- Python 3.12 recommended (the root `requirements.txt` contains pinned Python dependencies)
- Node.js 18 or newer and npm
- An Ollama API key for model-generated explanations
- Your own OpenAlex API key if you enable live literature lookup; OpenAlex is optional and lookup is off by default

You need your own credentials. Do not copy another person's key or commit API keys to the repository. The Ollama key belongs only in the backend environment, never in the frontend `.env` file.

## Run Locally

### 1. Set up the Python backend

From the repository root, create and activate a virtual environment, then install dependencies.

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set your own Ollama key in the terminal where you will run the backend.

PowerShell:

```powershell
$env:OLLAMA_API_KEY = "YOUR_OLLAMA_API_KEY"
```

macOS/Linux:

```bash
export OLLAMA_API_KEY="YOUR_OLLAMA_API_KEY"
```

The app can start without this key, but generated answers will use the rule-based fallback instead of Ollama.

Start the API from the repository root:

```bash
python -m uvicorn backend:app --reload
```

The API health endpoint is at <http://127.0.0.1:8000/health> and interactive API documentation is at <http://127.0.0.1:8000/docs>.

### 2. Start the React frontend

Open a second terminal and run:

```powershell
Set-Location react-frontend/react-frontend
npm install
Copy-Item .env.example .env
npm run dev
```

On macOS/Linux, replace `Copy-Item .env.example .env` with `cp .env.example .env`.

The example `.env` points to the local backend at `http://127.0.0.1:8000`. Keep the backend URL in this file, but do not put API keys there. Vite serves the frontend at <http://localhost:5173> by default.

The 3D brain view uses `public/fsaverage_brain.glb`, which is included in the frontend. If you change the backend address, update `VITE_BACKEND_URL` in the frontend `.env` and restart the Vite server.

## API Keys and Live Literature

### Ollama

Get an API key from your own Ollama account and set it as `OLLAMA_API_KEY` before starting the backend. The backend calls Ollama Cloud. Keep this key private and out of source control and frontend build settings.

### OpenAlex

Live literature lookup is disabled by default. To enable it, get your own free OpenAlex API key from <https://openalex.org/settings/api> and set these variables before starting the backend.

PowerShell:

```powershell
$env:LIVE_LITERATURE = "1"
$env:OPENALEX_API_KEY = "YOUR_OPENALEX_API_KEY"
```

macOS/Linux:

```bash
export LIVE_LITERATURE=1
export OPENALEX_API_KEY="YOUR_OPENALEX_API_KEY"
```

OpenAlex lookup supplements the project's curated `literature.json`; it does not replace that source. Live results are labelled as unvetted. Without a key, OpenAlex can still serve light use, but its free API key increases the request budget. Live lookup is optional and does not prevent the rest of the application from running.

## Model Pipeline

The main analysis script reads `block_avg_tmb.csv`, groups cross-validation splits by subject to avoid placing one subject in both train and test partitions, and generates rule/performance artifacts. Running it is not required for the web app when the generated artifacts already exist.

From the repository root:

```bash
python MainFold.py
python build_channel_atlas.py
```

These scripts regenerate analysis outputs and channel atlas data, respectively. Model fitting can take substantially longer than starting the web application. Review the script configuration and preserve existing results before regenerating outputs.

## Useful Files

| Path | Purpose |
|---|---|
| `MainFold.py` | Model fitting, subject-grouped cross-validation, and rule extraction |
| `block_avg_tmb.csv` | Block-averaged fNIRS input data |
| `extracted_rules_final_raw.json` | Structured rules consumed by the explainer |
| `grouped_kfold_results_raw.csv` | Cross-validation performance results |
| `backend.py` | FastAPI endpoints and service startup |
| `rag_core.py` | Rule retrieval, answer generation, scope checks, and fallback behavior |
| `literature_live.py` | Optional OpenAlex/Semantic Scholar lookup |
| `viz_tools.py` | Visualization payload generation |
| `channel_atlas.json` | Channel names and coordinates used by visualizations |
| `react-frontend/react-frontend/` | React/Vite web application |
| `dashboard_app.py` | Streamlit model-results dashboard |
| `regression_test.py` | Offline regression checks |
| `deployment.md` | Azure deployment and update instructions |
| `structure.md` | Detailed description of the research/model pipeline |

## Tests and Build

Run the offline backend regression checks from the repository root:

```bash
python regression_test.py
```

Create a production frontend build from the frontend directory:

```bash
npm run build
```

## Deployment

The Azure deployment guide documents the current hosted frontend and backend, configuration, redeployment commands, and verification steps. See [deployment.md](deployment.md). Never place `OLLAMA_API_KEY` or `OPENALEX_API_KEY` in the frontend environment or a public static-hosting configuration.
