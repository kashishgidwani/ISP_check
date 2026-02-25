# config.py
# ─────────────────────────────────────────────────────────────
# FILL IN ALL VALUES BELOW BEFORE RUNNING
# Never commit this file to git. Add to .gitignore immediately.
# ─────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
load_dotenv()

# ── Cloudflare Worker URL (your deployed worker endpoint) ─────
CF_WORKER_URL = os.getenv("CF_WORKER_URL", "https://your-worker.your-subdomain.workers.dev")

# ── Secret token shared between Streamlit app & CF Worker ────
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
CF_SECRET_TOKEN = os.getenv("CF_SECRET_TOKEN", "REPLACE_WITH_YOUR_SECRET_TOKEN")

# ── Google Sheets ─────────────────────────────────────────────
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "YOUR_GOOGLE_SHEET_ID_FROM_URL")
CREDENTIALS_FILE = "credentials.json"  # Service account JSON key

# ── 6 Key Locations (name, lat, lon) ─────────────────────────
# Edit these to your 6 survey locations
LOCATIONS = [
    {"name": "Admin Block",      "lat": 12.840781369406425, "lon": 80.15388067267332},  # A
    {"name": "Academic Block 1", "lat": 12.843791545367528, "lon": 80.15341169966594},  # B
    {"name": "Academic Block 2", "lat": 12.842892230021507, "lon": 80.15644974037480},  # C
    {"name": "Gazebo",           "lat": 12.841666470031175, "lon": 80.15454470158947},  # D
    {"name": "Student Parking",  "lat": 12.842005560382493, "lon": 80.15192150614196},  # E
    {"name": "Academic Block 3", "lat": 12.844138596358814, "lon": 80.15500320500311},  # F
]

# ── ISPs to track ─────────────────────────────────────────────
ISPS = ["Airtel", "Jio", "BSNL", "ACT", "Vi", "Other"]

# ── Time frames for 3 readings per day ───────────────────────
TIME_FRAMES = ["Morning (6–10 AM)", "Afternoon (12–4 PM)", "Evening (6–10 PM)"]

# ── YOLO model (auto-downloads on first run) ─────────────────
YOLO_MODEL = "yolov8n.pt"   # nano = fastest; swap to yolov8s.pt for better accuracy

# ── Google Sheets tab names ───────────────────────────────────
SHEET_NETWORK_TAB = "Network_Data"
SHEET_PEOPLE_TAB  = "People_Count"