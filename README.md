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

