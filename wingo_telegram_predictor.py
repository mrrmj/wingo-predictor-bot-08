#!/usr/bin/env python3
import os
import time
import json
import sqlite3
from collections import defaultdict
from typing import List, Tuple, Optional

import requests

# ==========================
# CONFIG
# ==========================

# 👉 PUT YOUR REAL VALUES HERE LOCALLY (do NOT share publicly)
TELEGRAM_BOT_TOKEN = "8237694201:AAGCy8nfrz9g6IKh0VrOLt1SC6bdR_dY7kM"
TELEGRAM_CHAT_ID   = "7284648472"  # keep as string

# Game API endpoints
GAME_URL    = "https://draw.ar-lottery01.com/WinGo/WinGo_1M.json?"
HISTORY_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?"

# Local DB file this script owns
DB_FILE = "wingo.db"

# Pattern library (lengths 13–19, we use 14–19)
PATTERN_FILE = "patterns_len13to19.json"

# k-lock thresholds
THRESHOLDS = {
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
}

# Approx historical accuracies by pattern length
L_ACCURACY = {
    14: 0.816,
    15: 0.849,
    16: 0.896,
    17: 0.939,
    18: 0.971,
    19: 0.988,
}

# Poll interval in seconds
POLL_INTERVAL = 5

# ==========================
# GLOBAL STATS (for /stats)
# ==========================

TOTAL_PREDS = 0
TOTAL_HITS = 0
STATS_BY_L = {L: {"total": 0, "hits": 0} for L in range(14, 20)}

CURRENT_DAY = None     # 'YYYY-MM-DD'
PREDS_DAY = 0
HITS_DAY = 0

LAST_UPDATE_ID = None  # for Telegram getUpdates offset


# ==========================
# TELEGRAM – RAW API
# ==========================

def tg_send_message(text: str):
    """Send a Telegram message using raw Bot API."""
    if TELEGRAM_BOT_TOKEN == "PUT_YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("[WARN] Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the script.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        resp = requests.post(url, data=data, timeout=10)
        if not resp.ok:
            print("[Telegram] Error:", resp.text)
    except Exception as e:
        print("[Telegram] Exception:", e)


def poll_telegram_updates_and_handle_stats():
    """
    Poll Telegram updates and respond to /stats command from TELEGRAM_CHAT_ID.
    Uses raw getUpdates and a global LAST_UPDATE_ID.
    """
    global LAST_UPDATE_ID, TOTAL_PREDS, TOTAL_HITS, CURRENT_DAY, PREDS_DAY, HITS_DAY, STATS_BY_L

    if TELEGRAM_BOT_TOKEN == "PUT_YOUR_TELEGRAM_BOT_TOKEN_HERE":
        return  # not configured

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {}
    if LAST_UPDATE_ID is not None:
        params["offset"] = LAST_UPDATE_ID + 1
    try:
        resp = requests.get(url, params=params, timeout=3)
        if not resp.ok:
            return
        data = resp.json()
        results = data.get("result", [])
        for upd in results:
            LAST_UPDATE_ID = upd.get("update_id", LAST_UPDATE_ID)
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            chat = msg.get("chat", {})
            chat_id = str(chat.get("id", ""))
            if chat_id != str(TELEGRAM_CHAT_ID):
                continue
            text = msg.get("text", "") or ""
            if text.strip().lower().startswith("/stats"):
                # Build stats reply
                if TOTAL_PREDS > 0:
                    overall_acc = round(100 * TOTAL_HITS / TOTAL_PREDS, 2)
                else:
                    overall_acc = 0.0
                if PREDS_DAY > 0:
                    day_acc = round(100 * HITS_DAY / PREDS_DAY, 2)
                else:
                    day_acc = 0.0

                lines = []
                lines.append("📊 *Forecast Stats*")
                lines.append(f"Date: `{CURRENT_DAY or 'N/A'}`")
                lines.append("")
                lines.append(f"Today: *{day_acc}%*  ({HITS_DAY}/{PREDS_DAY})")
                lines.append(f"Total (since start): *{overall_acc}%*  ({TOTAL_HITS}/{TOTAL_PREDS})")
                lines.append("")
                lines.append("*By Length L (total/hits/acc%)*")
                for L in range(14, 20):
                    t = STATS_BY_L[L]["total"]
                    h = STATS_BY_L[L]["hits"]
                    if t > 0:
                        accL = round(100 * h / t, 2)
                    else:
                        accL = 0.0
                    lines.append(f"L={L}: {h}/{t}  → {accL}%")
                tg_send_message("\n".join(lines))
    except Exception:
        # swallow any polling errors silently, not critical
        return


# ==========================
# DB SETUP & HELPERS
# ==========================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rounds (
            issue_number TEXT PRIMARY KEY,
            number       INTEGER NOT NULL,
            color        TEXT,
            category     TEXT,
            timestamp    TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_rounds(rows: List[Tuple[str, int, str, str, str]]):
    """
    rows = [(issue_number, number, color, category, timestamp), ...]
    """
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    for issue, num, color, cat, ts in rows:
        try:
            cur.execute(
                "INSERT OR IGNORE INTO rounds(issue_number, number, color, category, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (issue, num, color, cat, ts)
            )
        except Exception as e:
            print("[DB] Insert error:", e)
    conn.commit()
    conn.close()


def load_history_bs() -> List[str]:
    """
    Load full B/S history from DB ordered by timestamp.
    """
    if not os.path.exists(DB_FILE):
        return []
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT number FROM rounds ORDER BY timestamp ASC")
    rows = cur.fetchall()
    conn.close()

    history = []
    for (num,) in rows:
        bs = "B" if num >= 5 else "S"
        history.append(bs)
    return history


def get_last_issue_number() -> Optional[str]:
    if not os.path.exists(DB_FILE):
        return None
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT issue_number FROM rounds ORDER BY timestamp DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if row:
        return row[0]
    return None


# ==========================
# FETCHER (HISTORY ENDPOINT)
# ==========================

def fetch_new_rounds_from_server(last_issue: Optional[str]) -> List[Tuple[str, int, str, str, str]]:
    """
    Returns list of NEW rounds as:
        (issue_number, number, color, category, timestamp)
    Only issue_number > last_issue are returned.
    """
    rows: List[Tuple[str, int, str, str, str]] = []

    try:
        params = {
            "pageSize": 20,
            "pageIndex": 1
        }
        resp = requests.get(HISTORY_URL, params=params, timeout=10)
        if not resp.ok:
            print("[Fetcher] HTTP error:", resp.status_code, resp.text)
            return rows

        data = resp.json()

        lst = []
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], dict) and "list" in data["data"]:
                lst = data["data"]["list"]
            elif "Data" in data and isinstance(data["Data"], dict) and "List" in data["Data"]:
                lst = data["Data"]["List"]
            else:
                print("[Fetcher] Unknown JSON structure keys:", list(data.keys()))
                return rows

        for item in lst:
            try:
                issue = str(item.get("issueNumber") or item.get("IssueNumber") or "")
                num = int(item.get("number") or item.get("Number") or 0)
                color = str(item.get("color") or item.get("Color") or "")
                category = "Small" if num <= 4 else "Big"
                ts = str(
                    item.get("openTime") or item.get("OpenTime") or
                    item.get("time") or item.get("Time") or
                    time.strftime("%Y-%m-%d %H:%M:%S")
                )
                if not issue:
                    continue
                if last_issue is not None and issue <= last_issue:
                    continue
                rows.append((issue, num, color, category, ts))
            except Exception as e:
                print("[Fetcher] Item parse error:", e)
                continue

        rows.sort(key=lambda r: r[0])

    except Exception as e:
        print("[Fetcher] Exception:", e)

    return rows


# ==========================
# COMBINED PATTERN PREDICTOR
# ==========================

class CombinedPredictor:
    """
    Combined pattern mode predictor:
    uses patterns of length 14–19 with per-length k-lock thresholds.
    """

    def __init__(self, pattern_file: str):
        with open(pattern_file, "r") as f:
            self.patterns = json.load(f)
        self.prefix_maps = self._build_prefix_maps()

    def _build_prefix_maps(self):
        """
        prefix_maps[L][k][prefix] -> list of patterns with that prefix of length k
        """
        pmaps: dict[int, dict[int, dict[str, list[str]]]] = {}
        for L in range(14, 20):
            pats_dict = self.patterns.get(str(L), {})
            if not pats_dict:
                continue
            pats = list(pats_dict.keys())
            maps = {k: defaultdict(list) for k in range(1, L)}
            for pat in pats:
                for k in range(1, L):
                    maps[k][pat[:k]].append(pat)
            pmaps[L] = maps
        return pmaps

    def predict_next(self, history_bs: List[str]):
        """
        Use last k values in history to see if there's a unique pattern lock-in.
        Returns (prediction, L, k_lock) only when:
          - EXACTLY ONE matching pattern for prefix of length k
          - k >= THRESHOLDS[L]
        If no lock-in → (None, None, None)
        """
        seq = "".join(history_bs)
        n = len(seq)

        for L in range(14, 20):
            if L not in self.prefix_maps:
                continue
            maps = self.prefix_maps[L]

            for k in range(THRESHOLDS[L], L):
                if n < k:
                    break
                prefix = seq[-k:]
                cands = maps[k].get(prefix)
                if not cands:
                    break
                if len(cands) == 1:
                    pat = cands[0]
                    prediction = pat[k]
                    return prediction, L, k

        return None, None, None


def estimate_confidence(L: int, k_lock: int) -> float:
    base_acc = L_ACCURACY.get(L, 0.8)
    score = base_acc * (k_lock / (L - 1))
    return round(score * 100, 2)


# ==========================
# MAIN LOOP WITH DAY + TOTAL ACCURACY
# ==========================

def main():
    global TOTAL_PREDS, TOTAL_HITS, STATS_BY_L
    global CURRENT_DAY, PREDS_DAY, HITS_DAY

    if TELEGRAM_BOT_TOKEN == "PUT_YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("⚠️ Please edit TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the script before running.")
        return

    if not os.path.exists(PATTERN_FILE):
        print(f"❌ Pattern file not found: {PATTERN_FILE}")
        return

    print("✅ Initializing DB & predictor...")
    init_db()
    predictor = CombinedPredictor(PATTERN_FILE)

    history = load_history_bs()
    last_issue = get_last_issue_number()

    print(f"📜 Loaded {len(history)} historical rounds.")
    print(f"🆔 Last known issue: {last_issue}")

    pending_pred = None  # {"pred": 'B'/'S', "L":L, "k":k}

    while True:
        try:
            # 1) Fetch new rounds
            new_rows = fetch_new_rounds_from_server(last_issue)
            if new_rows:
                insert_rounds(new_rows)
                last_issue = new_rows[-1][0]

                for issue, num, color, cat, ts in new_rows:
                    bs = "B" if num >= 5 else "S"

                    # --- determine day string ---
                    day_str = ts.split(" ")[0] if " " in ts else ts

                    # --- detect new day and reset daily stats ---
                    if CURRENT_DAY is None:
                        CURRENT_DAY = day_str
                    elif day_str != CURRENT_DAY:
                        CURRENT_DAY = day_str
                        PREDS_DAY = 0
                        HITS_DAY = 0

                    # 2) Evaluate previous prediction (if any)
                    if pending_pred is not None:
                        TOTAL_PREDS += 1
                        PREDS_DAY += 1

                        Lp = pending_pred["L"]
                        STATS_BY_L[Lp]["total"] += 1

                        if bs == pending_pred["pred"]:
                            TOTAL_HITS += 1
                            HITS_DAY += 1
                            outcome = "✅ HIT"
                        else:
                            outcome = "❌ MISS"

                        overall_acc = round(100 * TOTAL_HITS / TOTAL_PREDS, 2)
                        day_acc = round(100 * HITS_DAY / PREDS_DAY, 2) if PREDS_DAY > 0 else 0.0

                        L_acc = 0.0
                        if STATS_BY_L[Lp]["total"] > 0:
                            L_acc = round(
                                100 * STATS_BY_L[Lp]["hits"] / STATS_BY_L[Lp]["total"], 2
                            )

                        result_msg = (
                            f"{outcome}\n"
                            f"Issue: `{issue}`\n"
                            f"Date: `{CURRENT_DAY}`\n"
                            f"Actual: `{bs}`  | Predicted: `{pending_pred['pred']}`\n"
                            f"Length L={Lp}, k={pending_pred['k']}\n\n"
                            f"Today accuracy: *{day_acc}%*  "
                            f"({HITS_DAY}/{PREDS_DAY})\n"
                            f"Total accuracy (since start): *{overall_acc}%*  "
                            f"({TOTAL_HITS}/{TOTAL_PREDS})\n"
                            f"Accuracy for L={Lp}: *{L_acc}%* "
                            f"({STATS_BY_L[Lp]['hits']}/{STATS_BY_L[Lp]['total']})"
                        )
                        print("[RESULT]", result_msg.replace("\n", " | "))
                        tg_send_message(result_msg)

                        pending_pred = None

                    # 3) Append current round to history
                    history.append(bs)
                    if len(history) < 13:
                        continue

                    # 4) Try new lock-in prediction for NEXT round
                    pred, L, k = predictor.predict_next(history)
                    if pred is None:
                        continue  # no lock-in → no signal

                    conf = estimate_confidence(L, k)
                    pending_pred = {"pred": pred, "L": L, "k": k}

                    overall_acc_now = round(100 * TOTAL_HITS / TOTAL_PREDS, 2) if TOTAL_PREDS > 0 else 0.0
                    day_acc_now = round(100 * HITS_DAY / PREDS_DAY, 2) if PREDS_DAY > 0 else 0.0

                    signal_msg = (
                        "📢 *Prediction Signal*\n"
                        f"Latest issue: `{issue}`\n"
                        f"Time: `{ts}`\n"
                        f"Date: `{CURRENT_DAY}`\n\n"
                        f"👉 *Bet on next round*: `{pred}`  (B=Big, S=Small)\n"
                        f"Pattern length: *{L}*\n"
                        f"k_lock: *{k}*\n"
                        f"Est. confidence: *{conf}%*\n\n"
                        f"Today accuracy (so far): *{day_acc_now}%*\n"
                        f"Total accuracy (since start): *{overall_acc_now}%*"
                    )
                    print("[SIGNAL]", signal_msg.replace("\n", " | "))
                    tg_send_message(signal_msg)

            else:
                print("🔁 No new rounds fetched...")

            # 5) Poll Telegram for /stats command
            poll_telegram_updates_and_handle_stats()

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            break
        except Exception as e:
            print("[MAIN] Exception:", e)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
