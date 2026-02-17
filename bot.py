#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (v6.2 - Precision & Shortcut Fix).

FIXES:
- ✅ EQUALITY FIX: Momentum now triggers if it's EXACTLY equal to your setting.
- ✅ SHORTCUTS: Restored -l, -q, and -s for CLI commands.
- ✅ IRON-CLAD: Verification loop ensures no ghost positions remain.
"""

import os, sys, json, argparse, time
import math
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ==============================================================================
# 🚀 STRATEGY SETTINGS (Edit these for fast updates)
# ==============================================================================
ASSET = "BTC"                # BTC, ETH, or SOL
LOOKBACK_MINS = 5            # Momentum timeframe (minutes)
MIN_MOMENTUM_PCT = 0.08      # Entry trigger threshold (%)
MAX_POSITION_AMOUNT = 5.0    # Default trade size in USDC

STOP_LOSS_PCT = 0.05         # Hard Stop Loss (0.05 = 5%)
TAKE_PROFIT_PCT = 0.20       # Take Profit (0.20 = 20%)
CLOSE_BUFFER_SECONDS = 80    # Buffer before expiry to force exit
# ==============================================================================

# -----------------------
# Persistence
# -----------------------
DEFAULT_DATA_DIR = "/data" if os.path.isdir("/data") else ".data"
DATA_DIR = os.environ.get("BOT_STATE_DIR", DEFAULT_DATA_DIR)
STATE_PATH = os.environ.get("BOT_STATE_PATH", os.path.join(DATA_DIR, "state.json"))
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------
# API Helpers
# -----------------------
SIMMER_BASE = os.environ.get("SIMMER_API_BASE", "https://api.simmer.markets")
TRADE_SOURCE = "railway:fastloop"

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception: return default

def save_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def api_request(url, method="GET", data=None, headers=None, timeout=25):
    try:
        req_headers = headers or {}
        req_headers.setdefault("User-Agent", "railway-fastloop/6.2")
        body = json.dumps(data).encode("utf-8") if data else None
        if data: req_headers["Content-Type"] = "application/json"
        req = Request(url, data=body, headers=req_headers, method=method)
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def simmer_request(path, method="GET", data=None, api_key=None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return api_request(f"{SIMMER_BASE}{path}", method=method, data=data, headers=headers)

# -----------------------
# Logic
# -----------------------
def iron_clad_close(api_key, market_id):
    print(f"IRON-CLAD CLOSE: Ensuring exit for market {market_id}...")
    while True:
        r = simmer_request("/api/sdk/positions", api_key=api_key)
        positions = r.get("positions", []) if isinstance(r, dict) else []
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        
        if not my_pos:
            print("VERIFIED: Position is completely gone.")
            return True

        s_yes = float(my_pos.get("shares_yes", 0))
        s_no = float(my_pos.get("shares_no", 0))

        if s_yes <= 0.0001 and s_no <= 0.0001:
            print("VERIFIED: Wallet shares at zero.")
            return True

        if s_yes > 0:
            simmer_request("/api/sdk/trade", method="POST", api_key=api_key, 
                           data={"market_id": market_id, "side": "yes", "action": "sell", "shares": s_yes, "venue": "polymarket"})
        if s_no > 0:
            simmer_request("/api/sdk/trade", method="POST", api_key=api_key, 
                           data={"market_id": market_id, "side": "no", "action": "sell", "shares": s_no, "venue": "polymarket"})
        time.sleep(2.0)

def monitor_and_close(api_key, market_id, end_time, side):
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    time.sleep(2)
    r = simmer_request("/api/sdk/positions", api_key=api_key)
    pos = next((p for p in r.get("positions", []) if str(p.get("market_id")) == str(market_id)), None)
    entry_price = float(pos.get("avg_buy_price", 0.5)) if pos else 0.5
    
    while now_utc() < target_time:
        res = api_request(f"https://gamma-api.polymarket.com/markets/{market_id}")
        if isinstance(res, dict) and "outcomePrices" in res:
            prices = json.loads(res["outcomePrices"])
            curr_price = float(prices[0] if side == "yes" else prices[1])
            pnl = (curr_price - entry_price) / entry_price
            print(f"PnL: {pnl*100:+.1f}% | Price: {curr_price:.3f}")
            if pnl <= -STOP_LOSS_PCT or pnl >= TAKE_PROFIT_PCT:
                print("THRESHOLD HIT. Closing.")
                break
        time.sleep(1.0)
    iron_clad_close(api_key, market_id)

def run_once(live, quiet):
    api_key = os.environ.get("SIMMER_API_KEY")
    now = now_utc()
    minute = (now.minute // 5) * 5
    start_dt = now.replace(minute=minute, second=0, microsecond=0)
    slug = f"{ASSET.lower()}-updown-5m-{int(start_dt.timestamp())}"
    end_time = start_dt + timedelta(minutes=5)

    if not quiet: print(f"TARGET: {slug}")

    # Check for existing trade
    r_pos = simmer_request("/api/sdk/positions", api_key=api_key)
    if any(slug in str(p.get("slug", "")) for p in (r_pos.get("positions", []) if isinstance(r_pos, dict) else [])):
        return

    # Signal
    b_res = api_request(f"https://api.binance.com/api/v3/klines?symbol={ASSET}USDT&interval=1m&limit=15")
    if "error" in b_res: return
    closes = [float(x[4]) for x in b_res]
    momentum = ((closes[-1] - closes[-LOOKBACK_MINS]) / closes[-LOOKBACK_MINS]) * 100
    
    # PRECISION FIX: Added 0.0001 tolerance for equality
    if abs(momentum) < (MIN_MOMENTUM_PCT - 0.0001):
        if not quiet: print(f"SKIP: Weak momentum ({momentum:.3f}%)")
        return
    
    side = "yes" if momentum > 0 else "no"
    print(f"SIGNAL: {side.upper()} | Momentum: {momentum:.3f}%")

    # Import & Trade
    m_res = simmer_request("/api/sdk/markets/import", method="POST", api_key=api_key, 
                           data={"polymarket_url": f"https://polymarket.com/event/{slug}", "shared": True})
    market_id = m_res.get("market_id")
    if not market_id: return

    if not live:
        print(f"DRY RUN: Would buy {side.upper()}")
        return

    t_res = simmer_request("/api/sdk/trade", method="POST", api_key=api_key, 
                           data={"market_id": market_id, "side": side, "action": "buy", "amount": MAX_POSITION_AMOUNT, "venue": "polymarket"})
    
    if t_res.get("success"):
        monitor_and_close(api_key, market_id, end_time, side)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RESTORED SHORTCUTS
    parser.add_argument("--live", "-l", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--smart-sizing", "-s", action="store_true")
    args = parser.parse_args()
    run_once(args.live, args.quiet)
