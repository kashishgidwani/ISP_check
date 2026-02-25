# ISP & Footfall Research Monitor

Web app to log campus network quality and people footfall into Google Sheets, with YOLO-based people counting.

---

## Local development

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Configure:**

- Put `credentials.json` (Google service account key) in the project root.
- Create `.env` with:

```bash
SPREADSHEET_ID=your_google_sheet_id
CF_WORKER_URL=https://your.worker.dev   # optional
CF_SECRET_TOKEN=your_secret_token       # optional
```

---

## Deploy to Streamlit Community Cloud

### 1. Push the app to GitHub

From the project folder:

```bash
cd /Users/scarlett/Documents/eda_project

git init
git add app.py config.py requirements.txt README.md .gitignore
# Do NOT add .env or credentials.json (they are in .gitignore)
git commit -m "Initial commit"
```

Create a new repository on GitHub (e.g. `isp-footfall-monitor`), then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### 2. Create the app on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **“New app”**.
3. Choose:
   - **Repository:** `YOUR_USERNAME/YOUR_REPO_NAME`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **“Advanced settings”** and add the secrets in the next step.

### 3. Add secrets (required for cloud)

In the app’s **“Secrets”** (or **“Advanced settings” → “Secrets”**), paste:

```toml
SPREADSHEET_ID = "your_google_sheet_id_from_url"
CF_WORKER_URL = "https://your-worker.workers.dev"
CF_SECRET_TOKEN = "your_secret_token"

# Paste the *entire contents* of credentials.json as one JSON string (no line breaks):
GOOGLE_CREDENTIALS_JSON = '{"type": "service_account", "project_id": "...", ...}'
```

**How to get `GOOGLE_CREDENTIALS_JSON`:**

- Open your local `credentials.json`.
- Copy the whole file (one JSON object).
- In the Secrets box, use one of these formats:
  - **TOML:**  
    `GOOGLE_CREDENTIALS_JSON = '''{"type": "service_account", ...}'''`  
    (triple quotes so the JSON can span lines.)
  - Or use the Streamlit Cloud UI: add a key `GOOGLE_CREDENTIALS_JSON` and paste the JSON string as the value.

### 4. Deploy

Click **“Deploy”**. Streamlit will install from `requirements.txt` and run `app.py`. When it’s ready, you get a public URL like:

`https://YOUR_APP_NAME.streamlit.app`

Open it in Chrome on your phone or any device to use the app.

### 5. Share your Google Sheet with the service account

In Google Cloud, the service account has an email like `something@project-id.iam.gserviceaccount.com`. In your Google Sheet: **Share** → add that email with **Editor** access. Otherwise the app will get permission errors when writing.

---

## Summary

| Step | Action |
|------|--------|
| 1 | Push code to GitHub (no `.env`, no `credentials.json`) |
| 2 | Create new app on share.streamlit.io → connect repo, main file `app.py` |
| 3 | Set Secrets: `SPREADSHEET_ID`, `CF_WORKER_URL`, `CF_SECRET_TOKEN`, `GOOGLE_CREDENTIALS_JSON` |
| 4 | Deploy and open the URL |
| 5 | Share the Google Sheet with the service account email |
