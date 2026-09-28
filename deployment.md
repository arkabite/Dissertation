# Deployment Guide

This document records the deployment completed on 2026-09-27 and the commands to publish later backend or frontend changes.

## Deployed Architecture

- **Backend:** FastAPI on Azure App Service, Linux, Python 3.12, Free (F1) plan.
- **Frontend:** React/Vite static files served by Azure Storage static website hosting.
- **Backend URL:** `https://fnirs-rag-backend.azurewebsites.net`
- **Frontend URL:** `https://fnirsragweb7f3a26.z1.web.core.windows.net/`
- **Resource group:** `fnirs-rag-rg`
- **Azure region for the app and storage:** Germany West Central.
- **Storage account:** `fnirsragweb7f3a26` (Standard_LRS).

The repository is arranged as a backend project at its root, with the frontend nested under `react-frontend/react-frontend/`. No folder restructuring was needed.

## Why These Choices Were Made

The Azure for Students subscription has an `Allowed resource deployment regions` policy. It permits Germany West Central, Poland Central, Austria East, Belgium Central, and Italy North. The App Service creation in UK South was rejected by that policy, so the App Service Plan was created in Germany West Central. The resource group itself remains in UK South; a resource group and the resources inside it do not have to use the same location.

Static Web Apps could not be used with this subscription policy. The locations offered by the Static Web Apps provider did not overlap the subscription's permitted locations. Instead, the frontend is hosted by the Storage static website feature in Germany West Central, which is permitted and supported there.

The storage account uses locally redundant storage (`Standard_LRS`) because this is a small demo site and does not need geo-redundant copies. Storage usage is billed according to capacity, operations, and data transfer. A low-traffic site should use little, but charges are not guaranteed to be zero. Deleting the storage account stops future charges for that frontend host.

## Backend Configuration

The App Service startup command is:

```text
uvicorn backend:app --host 0.0.0.0 --port 8000
```

The App Service settings include:

- `DB_PATH=/home/data/interactions.db`, placing the SQLite database under App Service's persistent `/home` storage. `backend.py` creates the parent directory as needed. Local runs default to `interactions.db` in the current directory.
- `OLLAMA_API_KEY`, stored as an App Service environment setting, not in a deployed source file. Set or rotate it through Azure Portal under the App Service's **Environment variables** settings. Do not put it in the frontend `.env`: Vite embeds frontend environment variables into public JavaScript bundles.

FastAPI CORS allows the deployed frontend origin:

```text
https://fnirsragweb7f3a26.z1.web.core.windows.net
```

The backend dependencies are pinned in the root `requirements.txt`, captured from the tested Python 3.12 virtual environment. If dependencies change, update the project environment first and regenerate this file from that environment; do not freeze global Python by accident. In Windows PowerShell 5.1, use `Set-Content -Encoding ASCII` because `>` can write UTF-16 text that pip cannot read as a normal requirements file:

```powershell
& .\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding ASCII requirements.txt
& .\.venv\Scripts\python.exe -m pip check
```

## What Was Deployed

1. Azure CLI was installed and authenticated to the **Azure for Students** subscription.
2. Resource group `fnirs-rag-rg` was created in UK South.
3. Backend preparation made `DB_PATH` environment-configurable while keeping the existing local default. The root `requirements.txt` was pinned from `.venv`.
4. The first App Service Plan attempt in UK South was rejected by subscription policy. The successful Linux Python 3.12 F1 App Service was created in Germany West Central.
5. The App Service startup command and persistent database setting were configured. `OLLAMA_API_KEY` was set in App Service settings and later rotated; keep the replacement there only.
6. The backend health endpoint returned `status: ok`, six loaded rules, and a loaded channel atlas. A live `/ask` request returned a grounded answer with `used_fallback: false`.
7. `Microsoft.Storage` was registered because it was not registered in the subscription. A Standard_LRS StorageV2 account was created in Germany West Central, and static website hosting was enabled with `index.html` as both the index document and the single-page-app fallback.
8. The frontend `.env` was set to the live backend URL, the Vite production build was created, and `dist/` was uploaded to the Storage `$web` container.
9. The Storage website origin was added to FastAPI's CORS allowlist and the backend was redeployed. Live preflight returned HTTP 200 with the expected `Access-Control-Allow-Origin` value.
10. The deployed page rendered in a browser, and a browser-submitted question returned an assistant answer.

The deployment checklist mentioned a 14-question live test. No 14-question list was found in the workspace. `questions.txt` contains an 11-question transcript; the full 14-case test has not been run.

## Deploy Backend Changes

Run these commands from the repository root (`C:\Users\AADITIYA\Desktop\Dissertation`). Azure CLI commands should be run after `az login` and with the intended subscription selected.

For edits to `backend.py`, `rag_core.py`, `viz_tools.py`, or backend data files in the repository root, redeploy the same App Service:

```powershell
az webapp up --name fnirs-rag-backend --resource-group fnirs-rag-rg --location germanywestcentral --runtime "PYTHON:3.12" --sku F1
```

Wait for Azure to report **Site started successfully**. This command packages and deploys the current repository root. It is deprecated by Azure CLI in favor of `webapp create` and `webapp deploy`, but it was the command used successfully for this deployment.

If you change dependencies, update the active `.venv`, regenerate `requirements.txt` with the commands above, check it with `pip check`, and then redeploy. If you change an App Service setting, save it in Azure and restart the app:

```powershell
az webapp restart --resource-group fnirs-rag-rg --name fnirs-rag-backend
```

If the frontend's public origin changes, add the new exact HTTPS origin to `allow_origins` in `backend.py` and redeploy the backend. Do not use `allow_origins=["*"]` with credentialed CORS.

## Deploy Frontend Changes

The frontend is in `react-frontend/react-frontend/`. Its `.env` contains:

```text
VITE_BACKEND_URL=https://fnirs-rag-backend.azurewebsites.net
```

Vite bakes this value into the production bundle at build time. Rebuild after changing frontend code or this URL. From PowerShell:

```powershell
Set-Location C:\Users\AADITIYA\Desktop\Dissertation\react-frontend\react-frontend
npm.cmd ci
npm.cmd run build
```

If PowerShell cannot resolve `npm.cmd`, use its installed path, commonly:

```powershell
& "$env:ProgramFiles\nodejs\npm.cmd" run build
```

Upload the new build to the existing static website:

```powershell
az storage blob upload-batch --account-name fnirsragweb7f3a26 --auth-mode key --destination '$web' --source .\dist --overwrite true
```

The signed-in identity did not have the Storage Blob data role, so the successful upload used `--auth-mode key`; Azure CLI retrieves the account key for the operation. Do not print, paste, or commit the key. If an administrator grants `Storage Blob Data Contributor` on this storage account to your identity, you can instead use `--auth-mode login`.

After upload, refresh the frontend with Ctrl+F5 to avoid a cached bundle. Frontend-only edits do not require a backend redeployment unless you also change the API contract or frontend origin.

## Verify the Live Deployment

Check backend health:

```powershell
Invoke-RestMethod https://fnirs-rag-backend.azurewebsites.net/health | ConvertTo-Json -Depth 5
```

Check a live question:

```powershell
$body = @{ question = "What does AF7 tell us?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "https://fnirs-rag-backend.azurewebsites.net/ask" -ContentType "application/json" -Body $body
```

Open the frontend at `https://fnirsragweb7f3a26.z1.web.core.windows.net/` and submit a question. The first request after the free App Service has been idle may take longer due to cold start.

## Stop or Remove Resources

To remove only the frontend host and stop its future Storage charges:

```powershell
az storage account delete --name fnirsragweb7f3a26 --resource-group fnirs-rag-rg --yes
```

To remove the entire deployment, including the backend, its App Service Plan, storage, and persistent database, delete the resource group:

```powershell
az group delete --name fnirs-rag-rg --yes --no-wait
```

Resource-group deletion is destructive and cannot be undone. Use it only when you intend to remove the complete deployment.