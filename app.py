# app.py — ISP & Footfall Research Monitor
# Run: streamlit run app.py
#
# Fixed:
#   1. GPS coordinates now persisted in st.session_state so People Counter
#      tab retains location after Streamlit reruns (tab switches, button clicks).
#   2. All-NaN RuntimeWarning suppressed cleanly — NaN columns filled with NaN
#      (not dropped) before background_gradient so styling never sees an
#      all-NaN slice.
#   3. People-counter sheet columns aligned with actual save_row payload
#      (added DayOfWeek, DayType, Weather, Density, Group, Event columns).
#   4. PPL_HEADERS updated to match the extended row.
#   5. safe_net numeric coercions cover every optional column defensively.
#   6. Plotly figures use use_container_width=True (width="stretch" is not a
#      valid kwarg for st.plotly_chart — was silently ignored before).
#   7. st.dataframe width="stretch" replaced with use_container_width=True.
#   8. Minor: load_data() result unpacked safely even when sheets are empty.

import warnings
import math
import datetime
import os
import json
import statistics
import time

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import gspread
import streamlit as st
import streamlit.components.v1 as components
from google.oauth2.service_account import Credentials
from PIL import Image

# Suppress the All-NaN background_gradient RuntimeWarning cleanly
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="All-NaN slice encountered",
)

from config import (
    CF_WORKER_URL,
    CF_SECRET_TOKEN,
    SPREADSHEET_ID,
    CREDENTIALS_FILE,
    LOCATIONS,
    ISPS,
    TIME_FRAMES,
    YOLO_MODEL,
    SHEET_NETWORK_TAB,
    SHEET_PEOPLE_TAB,
)

# Override from Streamlit Secrets when deployed (e.g. Community Cloud)
if hasattr(st, "secrets") and st.secrets:
    SPREADSHEET_ID = st.secrets.get("SPREADSHEET_ID", SPREADSHEET_ID)
    CF_WORKER_URL = st.secrets.get("CF_WORKER_URL", CF_WORKER_URL)
    CF_SECRET_TOKEN = st.secrets.get("CF_SECRET_TOKEN", CF_SECRET_TOKEN)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ISP & Footfall Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"]        { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3                        { font-family: 'Space Mono', monospace; letter-spacing: -0.03em; }
  .stTabs [data-baseweb="tab"]      { font-family: 'Space Mono', monospace; font-size: 0.8rem; }
  .metric-card {
    background: #0f172a; color: #e2e8f0; border-radius: 12px;
    padding: 1rem 1.2rem; margin: 0.3rem 0; border-left: 4px solid #38bdf8;
  }
  .metric-card .val  { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
  .metric-card .lbl  { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
  .result-box {
    background: #0f172a; border-radius: 12px; padding: 1.5rem;
    border: 1px solid #1e293b; margin: 0.5rem 0; text-align: center; color: #64748b;
  }
  @media (max-width: 640px) {
    .metric-card .val { font-size: 1.3rem; }
    .stButton>button { width: 100%; }
  }
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════
# CONSTANTS — column names
# ══════════════════════════════════════════════════════════════
LOC_DIST_COLS = [f"DistTo_{l['name'].replace(' ', '_')}_m" for l in LOCATIONS]

NET_HEADERS = (
    [
        "Timestamp", "Date", "DayOfWeek", "WeekdayOrWeekend", "TimeFrame",
        "ReadingNo", "ZoneLabel", "ActualLat", "ActualLon", "ISP",
        "DownloadMbps", "UploadMbps", "PingMs", "LatencyMs",
        "JitterMs", "PacketLossPct", "RTT_Min_ms", "RTT_Avg_ms", "RTT_Max_ms",
    ]
    + LOC_DIST_COLS
)

# Extended people headers — must match the row built in Tab 2 exactly
PPL_HEADERS = (
    [
        "Timestamp", "Date", "DayOfWeek", "WeekdayOrWeekend", "TimeFrame",
        "ReadingNo", "ZoneLabel", "ActualLat", "ActualLon",
        "PeopleCount", "Weather", "CrowdDensity", "DominantGroup",
        "EventNearby", "EventType",
    ]
    + LOC_DIST_COLS
)

# ══════════════════════════════════════════════════════════════
# PURE HELPERS
# ══════════════════════════════════════════════════════════════

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between two GPS coordinates."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)


def distances_to_all(lat: float, lon: float) -> list[float]:
    """Distances in metres from (lat, lon) to every configured epicenter."""
    return [haversine_m(lat, lon, loc["lat"], loc["lon"]) for loc in LOCATIONS]


def nearest_location_idx(lat: float, lon: float) -> int:
    dists = distances_to_all(lat, lon)
    return int(min(range(len(dists)), key=lambda i: dists[i]))


def now_meta() -> tuple[str, str, str, str]:
    """Returns (iso_timestamp, date_str, day_name, day_type)."""
    now = datetime.datetime.now()
    day_type = "Weekend" if now.weekday() >= 5 else "Weekday"
    return now.isoformat(), now.strftime("%Y-%m-%d"), now.strftime("%A"), day_type

# ══════════════════════════════════════════════════════════════
# GOOGLE SHEETS LAYER
# ══════════════════════════════════════════════════════════════

def _get_credentials():
    """Load Google credentials from Streamlit secrets (cloud) or credentials.json (local)."""
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    # Cloud deploy: use secrets if set (paste full JSON from credentials.json)
    if hasattr(st, "secrets") and st.secrets:
        raw = st.secrets.get("GOOGLE_CREDENTIALS_JSON")
        if raw:
            try:
                info = raw if isinstance(raw, dict) else json.loads(raw)
                return Credentials.from_service_account_info(info, scopes=scopes), None
            except Exception as exc:
                return None, f"Invalid GOOGLE_CREDENTIALS_JSON in secrets: {exc}"
    # Local: use file
    if not os.path.exists(CREDENTIALS_FILE):
        return None, f"credentials.json not found at {os.path.abspath(CREDENTIALS_FILE)}"
    try:
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
        return creds, None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource
def get_sheet_client():
    creds, err = _get_credentials()
    if err:
        return None, err
    try:
        return gspread.authorize(creds), None
    except Exception as exc:
        return None, str(exc)


def get_worksheet(tab_name: str):
    client, err = get_sheet_client()
    if err:
        return None, err
    if not SPREADSHEET_ID or SPREADSHEET_ID == "YOUR_GOOGLE_SHEET_ID_FROM_URL":
        return None, "SPREADSHEET_ID not set in config.py / .env"
    try:
        sh = client.open_by_key(SPREADSHEET_ID)
        try:
            return sh.worksheet(tab_name), None
        except gspread.WorksheetNotFound:
            return sh.add_worksheet(title=tab_name, rows=5000, cols=40), None
    except gspread.exceptions.APIError as exc:
        return None, f"API error: {exc} — is the sheet shared with your service account?"
    except Exception as exc:
        return None, str(exc)


def save_row(tab_name: str, row: list) -> tuple[bool, str]:
    ws, err = get_worksheet(tab_name)
    if err:
        return False, err
    try:
        ws.append_row(row, value_input_option="USER_ENTERED")
        return True, "Saved to Google Sheets"
    except Exception as exc:
        return False, str(exc)


def push_row(row_type: str, row: list) -> tuple[bool, str]:
    """Try Cloudflare Worker first; fall back to direct Sheets write."""
    tab = SHEET_NETWORK_TAB if row_type == "network" else SHEET_PEOPLE_TAB
    cf_ready = (
        CF_WORKER_URL
        and "your-worker" not in CF_WORKER_URL
        and CF_SECRET_TOKEN != "REPLACE_WITH_YOUR_SECRET_TOKEN"
    )
    if cf_ready:
        try:
            resp = requests.post(
                CF_WORKER_URL,
                json={"type": row_type, "row": row},
                headers={"X-Secret-Token": CF_SECRET_TOKEN},
                timeout=10,
            )
            if resp.status_code == 200:
                return True, "Saved via Cloudflare Worker"
        except Exception:
            pass  # fall through to direct write
    return save_row(tab, row)


def read_sheet(tab_name: str) -> pd.DataFrame:
    ws, err = get_worksheet(tab_name)
    if err:
        return pd.DataFrame()
    try:
        data = ws.get_all_records()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def ensure_headers():
    for tab, headers in [(SHEET_NETWORK_TAB, NET_HEADERS), (SHEET_PEOPLE_TAB, PPL_HEADERS)]:
        ws, err = get_worksheet(tab)
        if err:
            st.error(f"❌ {tab}: {err}")
            continue
        if not ws.row_values(1):
            ws.append_row(headers)
            st.success(f"✅ Headers written to {tab}")
        else:
            st.info(f"ℹ️ {tab} already has headers")

# ══════════════════════════════════════════════════════════════
# SPEED TEST
# ══════════════════════════════════════════════════════════════

def run_cf_latency_probes(n: int = 10) -> tuple:
    """
    HTTP-based latency probes to Cloudflare edge.
    Works on networks that block ICMP ping (campus / corporate WiFi).
    Returns (avg_ms, jitter_ms, packet_loss_pct, rtt_min_ms, rtt_max_ms).
    All values are None if every probe failed.
    """
    rtts: list[float] = []
    failed = 0
    for _ in range(n):
        try:
            t0 = time.perf_counter()
            requests.get(
                "https://speed.cloudflare.com/__down?bytes=1",
                timeout=3,
                stream=False,
            )
            rtts.append((time.perf_counter() - t0) * 1000)
        except Exception:
            failed += 1
        time.sleep(0.05)

    if not rtts:
        return None, None, None, None, None

    avg = round(statistics.mean(rtts), 1)
    jitter = round(statistics.stdev(rtts), 2) if len(rtts) > 1 else 0.0
    rtt_min = round(min(rtts), 1)
    rtt_max = round(max(rtts), 1)
    loss = round((failed / n) * 100, 1)
    return avg, jitter, loss, rtt_min, rtt_max


def _http_download_speed(seconds: float = 5.0) -> tuple[float | None, str | None]:
    """
    Approximate download Mbps using a timed HTTPS download from Cloudflare.
    Returns (mbps, error_message). error_message is set only on failure.
    """
    # Use 25 MB max — Cloudflare docs use similar sizes; huge requests can timeout or be blocked
    url = "https://speed.cloudflare.com/__down?bytes=26214400"
    t0 = time.perf_counter()
    bytes_downloaded = 0
    try:
        with requests.get(url, stream=True, timeout=seconds + 15) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=64_000):
                if not chunk:
                    break
                bytes_downloaded += len(chunk)
                if time.perf_counter() - t0 > seconds:
                    break
    except requests.exceptions.HTTPError as e:
        return None, f"Download: server returned {e.response.status_code} ({e.response.reason})"
    except requests.exceptions.Timeout:
        return None, "Download: request timed out (slow or blocked network)"
    except requests.exceptions.ConnectionError as e:
        return None, f"Download: connection failed — {type(e).__name__}: {e}"
    except Exception as e:
        return None, f"Download: {type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0
    if elapsed <= 0 or bytes_downloaded == 0:
        return None, "Download: no data received (empty or blocked response)"
    return round((bytes_downloaded * 8) / (elapsed * 1e6), 2), None  # Mbps


def _http_upload_speed(megabytes: int = 5) -> tuple[float | None, str | None]:
    """
    Approximate upload Mbps using a single HTTPS POST to Cloudflare.
    Returns (mbps, error_message). error_message is set only on failure.
    """
    url = "https://speed.cloudflare.com/__up"
    size_bytes = megabytes * 1024 * 1024
    payload = os.urandom(size_bytes)
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, data=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return None, f"Upload: server returned {e.response.status_code} ({e.response.reason})"
    except requests.exceptions.Timeout:
        return None, "Upload: request timed out (slow or blocked network)"
    except requests.exceptions.ConnectionError as e:
        return None, f"Upload: connection failed — {type(e).__name__}: {e}"
    except Exception as e:
        return None, f"Upload: {type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0
    if elapsed <= 0:
        return None, "Upload: no timing"
    return round((size_bytes * 8) / (elapsed * 1e6), 2), None  # Mbps


def run_speedtest() -> tuple[dict | None, str | None]:
    """
    Measures download / upload via pure HTTPS requests to Cloudflare and
    latency metrics via HTTP probes (also Cloudflare).
    Returns (results_dict, error_string_or_None).
    """
    dl, err = _http_download_speed(seconds=5.0)
    if err or dl is None:
        return None, err or "HTTP download test failed — network or Cloudflare blocked."

    ul, err = _http_upload_speed(megabytes=5)
    if err or ul is None:
        return None, err or "HTTP upload test failed — network or Cloudflare blocked."

    latency, jitter, loss, rtt_min, rtt_max = run_cf_latency_probes(n=10)
    return {
        "download": dl,
        "upload": ul,
        "ping": latency if latency is not None else None,
        "latency": latency,
        "jitter_ms": jitter,
        "packet_loss_pct": loss,
        "rtt_min_ms": rtt_min,
        "rtt_avg_ms": latency,
        "rtt_max_ms": rtt_max,
    }, None

# ══════════════════════════════════════════════════════════════
# YOLO
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_yolo():
    try:
        from ultralytics import YOLO
        return YOLO(YOLO_MODEL)
    except Exception:
        return None


def count_people(model, image: Image.Image) -> tuple[int, Image.Image]:
    arr = np.array(image)
    results = model(arr, classes=[0], verbose=False)
    count = int(results[0].boxes.shape[0]) if results[0].boxes is not None else 0
    return count, Image.fromarray(results[0].plot())

# ══════════════════════════════════════════════════════════════
# GPS PANEL  (shared between Tab 1 & Tab 2)
# ══════════════════════════════════════════════════════════════

# JavaScript: reloads the page with ?gps_lat=&gps_lon= appended to the URL.
GPS_JS = """
<script>
function grabGPS() {
    var btn = document.getElementById('gpsbtn');
    btn.innerText = '📡 Detecting…';
    btn.disabled = true;
    if (!navigator.geolocation) {
        btn.innerText = '❌ Not supported';
        return;
    }
    navigator.geolocation.getCurrentPosition(
        function(p) {
            var url = new URL(window.location.href);
            url.searchParams.set('gps_lat', p.coords.latitude.toFixed(8));
            url.searchParams.set('gps_lon', p.coords.longitude.toFixed(8));
            window.location.href = url.toString();
        },
        function() { btn.innerText = '❌ Denied — enter manually'; btn.disabled = false; },
        { enableHighAccuracy: true, timeout: 10000 }
    );
}
</script>
<button id="gpsbtn" onclick="grabGPS()"
  style="background:#0ea5e9;color:#fff;border:none;border-radius:8px;
         padding:0.55rem 1.1rem;font-size:0.9rem;cursor:pointer;width:100%;max-width:100%;">
  📡 Capture My GPS Coordinates
</button>
"""


def gps_location_panel(key_prefix: str = "net"):
    """
    Renders GPS button + distance table + zone selectbox.

    GPS coordinates are read from query-params when first captured (page reload),
    then immediately persisted in st.session_state["gps_lat"] / ["gps_lon"] so
    they survive all subsequent Streamlit reruns (tab switches, button clicks, etc.).

    Returns: (actual_lat, actual_lon, zone_label, dists_list)
    """
    st.markdown("**📍 Your Position**")
    components.html(GPS_JS, height=48)

    params = st.query_params
    gps_lat_qp = params.get("gps_lat")
    gps_lon_qp = params.get("gps_lon")

    loc_names = [loc["name"] for loc in LOCATIONS]

    # Priority: query-param (fresh capture) → session_state (persisted) → manual input
    if gps_lat_qp and gps_lon_qp:
        actual_lat = float(gps_lat_qp)
        actual_lon = float(gps_lon_qp)
        # Persist so other tabs and future reruns can use them
        st.session_state["gps_lat"] = actual_lat
        st.session_state["gps_lon"] = actual_lon
        st.success(f"📍 GPS locked: **{actual_lat:.7f}, {actual_lon:.7f}** ← saved to sheet")
    elif "gps_lat" in st.session_state and "gps_lon" in st.session_state:
        actual_lat = st.session_state["gps_lat"]
        actual_lon = st.session_state["gps_lon"]
        st.success(f"📍 GPS (session): **{actual_lat:.7f}, {actual_lon:.7f}**")
    else:
        st.caption("No GPS yet — or enter manually:")
        actual_lat = st.number_input(
            "Latitude", value=12.8420000, format="%.7f", key=f"{key_prefix}_lat"
        )
        actual_lon = st.number_input(
            "Longitude", value=80.1540000, format="%.7f", key=f"{key_prefix}_lon"
        )

    dists = distances_to_all(actual_lat, actual_lon)
    nearest_idx = int(min(range(len(dists)), key=lambda i: dists[i]))

    st.markdown("**📏 Distance to each epicenter:**")
    for i, loc in enumerate(LOCATIONS):
        tag = " ◀ nearest" if i == nearest_idx else ""
        st.caption(f"• {loc['name']}: {dists[i]:.1f} m{tag}")

    zone = st.selectbox(
        "Zone label (nearest auto-selected, override if wrong)",
        loc_names,
        index=nearest_idx,
        key=f"{key_prefix}_zone",
    )
    return actual_lat, actual_lon, zone, dists

# ══════════════════════════════════════════════════════════════
# CLOUDFLARE STATUS
# ══════════════════════════════════════════════════════════════
cf_ready = (
    CF_WORKER_URL
    and "your-worker" not in CF_WORKER_URL
    and CF_SECRET_TOKEN != "REPLACE_WITH_YOUR_SECRET_TOKEN"
)

# ══════════════════════════════════════════════════════════════
# BANNER
# ══════════════════════════════════════════════════════════════
st.title("📡 ISP & Footfall Research Monitor")
st.caption("SRM Campus • 6 epicenters • Auto speed test • YOLO people counter")
if not cf_ready:
    st.info("ℹ️ Cloudflare Worker not configured — writing directly to Google Sheets.", icon="🔁")

tabs = st.tabs(["📥 Log Network & People", "📊 Analysis", "🗺️ Map", "⚙️ Setup"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — NETWORK + PEOPLE (MERGED)
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Network & People Log")
    st.caption("Use one location & time entry to log both network and footfall readings.")

    # ── Shared location & session meta ─────────────────────────────
    st.markdown("#### 📍 Location & Session")
    actual_lat, actual_lon, zone_label, dists = gps_location_panel("shared")

    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        time_frame = st.selectbox("🕐 Time Frame", TIME_FRAMES, key="session_time_frame")
    with meta_col2:
        reading_no = st.selectbox("Reading # (1–3)", [1, 2, 3], key="session_reading_no")

    st.markdown("---")
    col_net, col_ppl = st.columns([1, 1])

    # ── Network speed test & log ──────────────────────────────────
    with col_net:
        st.markdown("#### 🚀 Auto Speed Test")
        st.caption("Measures download / upload via Cloudflare HTTPS — no manual typing needed")
        with st.expander("Why can an HTTPS speed test fail?"):
            st.markdown(
                "HTTPS is just **encrypted HTTP**. The request can still fail because:\n"
                "- **Firewall/proxy** (e.g. campus Wi‑Fi) blocks or inspects traffic to speed test domains\n"
                "- **DNS** can’t resolve `speed.cloudflare.com`\n"
                "- **Timeout** — network too slow or the path is blocked\n"
                "- **Server response** — e.g. 403 Forbidden or 429 Too Many Requests if the server limits or blocks automated tests\n\n"
                "When it fails, the app now shows the **actual error** (e.g. status code or timeout) so you can see the cause."
            )

        if "speed_results" not in st.session_state:
            st.session_state.speed_results = None

        isp = st.selectbox("📶 ISP", ISPS)

        if st.button("▶️ Start Speed Test", type="primary", use_container_width=True):
            st.session_state.speed_results = None
            with st.spinner("Running… ~20–30 seconds ⏱️"):
                results, err = run_speedtest()
            if err:
                st.error(f"Speed test failed: {err}")
            else:
                st.session_state.speed_results = results

        r = st.session_state.speed_results
        if r:
            m1, m2 = st.columns(2)
            m1.metric("⬇️ Download", f"{r['download']} Mbps")
            m2.metric("⬆️ Upload", f"{r['upload']} Mbps")

            m3, m4 = st.columns(2)
            m3.metric("📶 Ping", f"{r['ping']} ms")
            m4.metric("⏱️ Latency", f"{r['latency']} ms")

            m5, m6 = st.columns(2)
            jitter_val = r.get("jitter_ms")
            loss_val = r.get("packet_loss_pct")
            m5.metric("〰️ Jitter", f"{jitter_val} ms" if jitter_val is not None else "—")
            m6.metric("📦 Packet Loss", f"{loss_val} %" if loss_val is not None else "—")

            m7, m8 = st.columns(2)
            m7.metric(
                "RTT Min",
                f"{r.get('rtt_min_ms')} ms" if r.get("rtt_min_ms") is not None else "—",
            )
            m8.metric(
                "RTT Max",
                f"{r.get('rtt_max_ms')} ms" if r.get("rtt_max_ms") is not None else "—",
            )

            if jitter_val is None:
                st.caption(
                    "⚠️ ICMP ping blocked on this network (common on campus/corporate WiFi) — "
                    "jitter & packet loss saved as empty. Download / upload / ping are unaffected."
                )

            st.markdown("---")
            if st.button("💾 Save Network Reading", use_container_width=True):
                ts, date_str, day_name, day_type = now_meta()
                row = (
                    [
                        ts, date_str, day_name, day_type,
                        time_frame, reading_no,
                        zone_label, actual_lat, actual_lon, isp,
                        r["download"], r["upload"], r["ping"], r["latency"],
                        r.get("jitter_ms", ""),
                        r.get("packet_loss_pct", ""),
                        r.get("rtt_min_ms", ""),
                        r.get("rtt_avg_ms", ""),
                        r.get("rtt_max_ms", ""),
                    ]
                    + dists
                )
                with st.spinner("Saving…"):
                    ok, msg = push_row("network", row)
                if ok:
                    st.success(
                        f"✅ {msg} — {isp} @ {zone_label} | "
                        f"↓{r['download']} ↑{r['upload']} Mbps"
                    )
                    st.session_state.speed_results = None
                else:
                    st.error(f"❌ {msg}")
        else:
            st.markdown(
                "<div class='result-box'><br>Press <b>▶️ Start Speed Test</b> to auto-measure<br>"
                "Download · Upload · Ping · Latency · Jitter · Packet Loss<br><br></div>",
                unsafe_allow_html=True,
            )

    # ── People counter (YOLO) ────────────────────────────────────
    with col_ppl:
        st.markdown("#### 👥 People Counter via YOLO")
        st.caption("YOLOv8 nano — loads on first use, cached after")

        st.markdown("##### 🌤️ Crowd Context *(for ML)*")
        weather = st.selectbox(
            "Weather", ["Sunny", "Cloudy", "Overcast", "Rainy"], key="p_weather"
        )
        density = st.selectbox(
            "Crowd Density", ["Sparse", "Moderate", "Dense", "Packed"], key="p_density"
        )
        group = st.selectbox(
            "Dominant Group", ["Students", "Staff", "Mixed", "Unknown"], key="p_group"
        )
        event = st.radio("Event Nearby?", ["No", "Yes"], horizontal=True, key="p_event")
        event_type = st.text_input(
            "Event type",
            placeholder="e.g. exam, fest, sports",
            key="p_event_type",
            disabled=(event == "No"),
        )

        st.markdown("##### 📷 Image Capture")
        src = st.radio(
            "Source", ["Upload file", "Camera (mobile/webcam)"], key="p_src"
        )
        uploaded = (
            st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
            if src == "Upload file"
            else st.camera_input("Take photo")
        )

        if uploaded:
            if st.button("🔍 Detect & Log", type="primary", use_container_width=True):
                with st.spinner("Loading YOLO (first time ~5 s)…"):
                    model = load_yolo()
                if model is None:
                    st.error("YOLO unavailable — run: pip install ultralytics")
                else:
                    img = Image.open(uploaded).convert("RGB")
                    with st.spinner("Running detection…"):
                        count, annotated = count_people(model, img)
                    st.image(
                        annotated,
                        caption=f"Detected: {count} people",
                        use_container_width=True,
                    )

                    ts, date_str, day_name, day_type = now_meta()
                    row = (
                        [
                            ts, date_str, day_name, day_type, time_frame, reading_no,
                            zone_label, actual_lat, actual_lon,
                            count, weather, density, group,
                            event,
                            event_type if event == "Yes" else "",
                        ]
                        + dists
                    )
                    with st.spinner("Saving…"):
                        ok, msg = push_row("people", row)
                    if ok:
                        st.success(
                            f"✅ {msg} — {count} people @ {zone_label} | {density} | {weather}"
                        )
                    else:
                        st.error(f"❌ {msg}")

# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Statistical Analysis")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    @st.cache_data(ttl=120)
    def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        return read_sheet(SHEET_NETWORK_TAB), read_sheet(SHEET_PEOPLE_TAB)

    net_df, ppl_df = load_data()

    # ── NETWORK ANALYSIS ─────────────────────────────────────
    if net_df.empty:
        st.info("No network data yet — run a speed test and save a reading.")
    else:
        # Coerce all numeric columns defensively — missing cols get NaN column
        numeric_net_cols = [
            "DownloadMbps", "UploadMbps", "PingMs", "LatencyMs",
            "JitterMs", "PacketLossPct",
            "RTT_Min_ms", "RTT_Avg_ms", "RTT_Max_ms",
        ]
        for col in numeric_net_cols:
            if col not in net_df.columns:
                net_df[col] = np.nan
            net_df[col] = pd.to_numeric(net_df[col], errors="coerce")

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, val in [
            (c1, "Avg Download",   f"{net_df['DownloadMbps'].mean():.1f} Mbps"),
            (c2, "Avg Upload",     f"{net_df['UploadMbps'].mean():.1f} Mbps"),
            (c3, "Avg Ping",       f"{net_df['PingMs'].mean():.1f} ms"),
            (c4, "Total Readings", str(len(net_df))),
        ]:
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        has_jitter = net_df["JitterMs"].notna().any()
        if has_jitter:
            j1, j2 = st.columns(2)
            j1.metric("Avg Jitter", f"{net_df['JitterMs'].mean():.1f} ms")
            j2.metric("Avg Packet Loss", f"{net_df['PacketLossPct'].mean():.1f} %")

        # ── ISP comparison chart
        isp_agg_cols = ["DownloadMbps", "UploadMbps", "PingMs"]
        if has_jitter:
            isp_agg_cols.append("JitterMs")
        isp_grp = net_df.groupby("ISP")[isp_agg_cols].mean().reset_index()

        fig1 = px.bar(
            isp_grp, x="ISP", y=["DownloadMbps", "UploadMbps"],
            barmode="group", title="Download vs Upload by ISP",
            color_discrete_sequence=["#38bdf8", "#818cf8"],
        )
        fig1.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="#e2e8f0"
        )
        st.plotly_chart(fig1, use_container_width=True)

        if has_jitter:
            fig_j = px.bar(
                isp_grp, x="ISP", y="JitterMs",
                title="Avg Jitter by ISP",
                color_discrete_sequence=["#f472b6"],
            )
            fig_j.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="#e2e8f0"
            )
            st.plotly_chart(fig_j, use_container_width=True)

        # ── Zone comparison chart
        zone_col = "ZoneLabel" if "ZoneLabel" in net_df.columns else "Location"
        zone_agg_cols = ["DownloadMbps", "PingMs"]
        if has_jitter:
            zone_agg_cols.append("JitterMs")
        zone_grp = (
            net_df.groupby(zone_col)[zone_agg_cols]
            .mean()
            .reset_index()
            .rename(columns={zone_col: "ZoneLabel"})
        )
        fig2 = px.bar(
            zone_grp, x="ZoneLabel", y="DownloadMbps",
            color="DownloadMbps", color_continuous_scale="Blues",
            title="Avg Download by Zone",
        )
        fig2.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="#e2e8f0"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Time of day chart
        tf_agg_cols = ["DownloadMbps", "PingMs"]
        if has_jitter:
            tf_agg_cols.append("JitterMs")
        tf_grp = net_df.groupby("TimeFrame")[tf_agg_cols].mean().reset_index()
        fig3 = px.line(
            tf_grp, x="TimeFrame", y="DownloadMbps",
            markers=True, title="Download by Time of Day",
            color_discrete_sequence=["#34d399"],
        )
        fig3.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="#e2e8f0"
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── Distance vs latency scatter
        dist_cols = [c for c in net_df.columns if c.startswith("DistTo_")]
        if dist_cols:
            chosen_dist = st.selectbox("Distance column for scatter", dist_cols)
            scatter_y = "LatencyMs"
            scatter_cols = [chosen_dist, scatter_y, "ISP"]
            if has_jitter:
                scatter_cols.append("JitterMs")
            melt = net_df[scatter_cols].dropna(subset=[chosen_dist, scatter_y])
            if len(melt) > 1:
                try:
                    import statsmodels.api as sm  # noqa: F401
                    trendline_opt = "ols"
                except ImportError:
                    trendline_opt = None
                fig4 = px.scatter(
                    melt, x=chosen_dist, y=scatter_y, color="ISP",
                    trendline=trendline_opt,
                    title=f"{chosen_dist} vs Latency",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig4.update_layout(
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="#e2e8f0"
                )
                st.plotly_chart(fig4, use_container_width=True)

        # ── Descriptive stats table
        # Fill NaN so background_gradient never encounters an all-NaN column
        with st.expander("Full Descriptive Statistics"):
            desc = net_df[numeric_net_cols].describe().T
            # Fill NaN cells with 0 for display only — keeps rows intact
            desc_display = desc.fillna(0)
            st.dataframe(
                desc_display.style.background_gradient(cmap="Blues"),
                use_container_width=True,
            )

    # ── PEOPLE ANALYSIS ───────────────────────────────────────
    if not ppl_df.empty:
        st.markdown("---")
        ppl_df["PeopleCount"] = pd.to_numeric(
            ppl_df.get("PeopleCount", 0), errors="coerce"
        ).fillna(0)

        ppl_zone = "ZoneLabel" if "ZoneLabel" in ppl_df.columns else "Location"
        ppl_grp = (
            ppl_df.groupby([ppl_zone, "TimeFrame"])["PeopleCount"]
            .mean()
            .reset_index()
            .rename(columns={ppl_zone: "ZoneLabel"})
        )
        fig5 = px.bar(
            ppl_grp, x="ZoneLabel", y="PeopleCount", color="TimeFrame",
            barmode="group", title="Avg Footfall by Zone & Time",
            color_discrete_sequence=["#fb923c", "#a78bfa", "#34d399"],
        )
        fig5.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="#e2e8f0"
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Distance vs people count
        p_dist_cols = [c for c in ppl_df.columns if c.startswith("DistTo_")]
        if p_dist_cols:
            for c in p_dist_cols:
                ppl_df[c] = pd.to_numeric(ppl_df[c], errors="coerce")
            chosen_p = st.selectbox("Epicenter distance for footfall scatter", p_dist_cols)
            pm = ppl_df[[chosen_p, "PeopleCount", "ZoneLabel"]].dropna(
                subset=[chosen_p, "PeopleCount"]
            )
            if len(pm) > 1:
                try:
                    import statsmodels.api as sm  # noqa: F401
                    trendline_opt_p = "ols"
                except ImportError:
                    trendline_opt_p = None
                figp = px.scatter(
                    pm, x=chosen_p, y="PeopleCount", color="ZoneLabel",
                    trendline=trendline_opt_p,
                    title=f"{chosen_p} vs People Count",
                )
                figp.update_layout(
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="#e2e8f0"
                )
                st.plotly_chart(figp, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — MAP
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Campus Epicenter Map")

    loc_df = pd.DataFrame(LOCATIONS)
    fig_map = px.scatter_map(
        loc_df, lat="lat", lon="lon", text="name",
        zoom=16, height=520,
        color_discrete_sequence=["#38bdf8"],
    )
    fig_map.update_traces(marker=dict(size=14), textposition="top center")
    fig_map.update_layout(
        map_style="open-street-map",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        margin={"r": 0, "t": 10, "l": 0, "b": 0},
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("#### Inter-Epicenter Distance Matrix (metres)")
    names = [loc["name"] for loc in LOCATIONS]
    dm = pd.DataFrame(
        [
            [
                haversine_m(
                    LOCATIONS[i]["lat"], LOCATIONS[i]["lon"],
                    LOCATIONS[j]["lat"], LOCATIONS[j]["lon"],
                )
                for j in range(len(LOCATIONS))
            ]
            for i in range(len(LOCATIONS))
        ],
        index=names,
        columns=names,
    )
    st.dataframe(dm.style.background_gradient(cmap="YlOrRd"), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — SETUP
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("⚙️ Setup & Diagnostics")

    creds_ok = os.path.exists(CREDENTIALS_FILE)
    sheet_ok = SPREADSHEET_ID not in ("", "YOUR_GOOGLE_SHEET_ID_FROM_URL", None)

    s1, s2, s3 = st.columns(3)
    s1.metric("credentials.json", "✅ Found"      if creds_ok else "❌ Missing")
    s2.metric("Spreadsheet ID",   "✅ Set"         if sheet_ok else "❌ Not set")
    s3.metric("Cloudflare",       "✅ Configured"  if cf_ready else "⚠️ Direct mode")

    st.markdown("---")

    if st.button("🧪 Test Google Sheets Connection", use_container_width=True):
        client, err = get_sheet_client()
        if err:
            st.error(f"❌ Auth failed: {err}")
        else:
            st.success("✅ Authenticated")
            if not sheet_ok:
                st.error("❌ SPREADSHEET_ID not set")
            else:
                try:
                    sh = client.open_by_key(SPREADSHEET_ID)
                    st.success(f"✅ Opened: '{sh.title}'")
                    ws, werr = get_worksheet("_test")
                    if werr:
                        st.error(f"❌ {werr}")
                    else:
                        ws.append_row(["CONNECTION TEST", datetime.datetime.now().isoformat()])
                        st.success("✅ Test write succeeded — saves will work!")
                        try:
                            sh.del_worksheet(ws)
                        except Exception:
                            pass
                except Exception as exc:
                    st.error(f"❌ {exc}")

    st.markdown("---")

    if st.button("🔧 Initialise Sheet Headers (run once)", use_container_width=True):
        with st.spinner("Writing headers…"):
            ensure_headers()

    if creds_ok:
        try:
            sa = json.load(open(CREDENTIALS_FILE))
            st.markdown("**Share your Google Sheet with this service account email:**")
            st.code(sa.get("client_email", ""))
        except Exception:
            pass

    st.markdown("---")
    st.markdown("**Install all dependencies:**")
    st.code(
        "pip install streamlit gspread google-auth "
        "pandas plotly pillow requests python-dotenv ultralytics"
    )
    st.markdown("**`.env` template:**")
    st.code(
        "SPREADSHEET_ID=your_sheet_id\n"
        "CF_WORKER_URL=https://your.workers.dev\n"
        "CF_SECRET_TOKEN=your_token"
    )