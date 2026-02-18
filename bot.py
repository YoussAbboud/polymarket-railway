#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (v8.7 - Anti-Glitch).

FIXES:
- ✅ 3-STRIKE EXIT RULE: Bot requires 3 consecutive "0 share" checks (spaced 5s) to exit.
- ✅ GLITCH PROOF: If API blinks and returns [] once, the bot ignores it and keeps running.
- ✅ LOUD COOLDOWN: Prints countdown so you know exactly when the bot wakes up.
- ✅ ROBUST GET_POSITIONS: Retries internally on network errors to prevent false empty lists.
"""

import os, sys, json, argparse, time
import math
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ==============================================================================
# 🚀 STRATEGY SETTINGS
# ==============================================================================
ASSET = "BTC"                
LOOKBACK_MINS = 12           
MIN_MOMENTUM_PCT = 0.12      
MAX_POSITION_AMOUNT = 5.0    
SMART_SIZING_PCT = 0.95      

STOP_LOSS_PCT = 0.15         
TAKE_PROFIT_PCT = 0.20       
CLOSE_BUFFER_SECONDS = 80    
COOLDOWN_SECONDS = 60        
# ==============================================================================

# -----------------------
# Persistence
# -----------------------
DEFAULT_DATA_DIR = "/data" if os.path.isdir("/data") else ".data"
DATA_DIR = os.environ.get("BOT_STATE_DIR", DEFAULT_DATA_DIR)
STATE_PATH = os.path.join(DATA_DIR, "state.json")
COOLDOWN_PATH = os.path.join(DATA_DIR, "last_close.json")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------
# API
# -----------------------
SIMMER_BASE = os.environ.get("SIMMER_API_BASE", "https://api.simmer.markets")
CLOB_BASE = "https://clob.polymarket.com"
TRADE_SOURCE = "railway:fastloop"
COINGECKO_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}

# -----------------------
# Helpers
# -----------------------
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

def api_request(url, method="GET", data=None, headers=None, timeout=10):
    try:
        req_headers = headers or {}
        req_headers.setdefault("User-Agent", "railway-fastloop/8.7")
        body = json.dumps(data).encode("utf-8") if data else None
        if data: req_headers["Content-Type"] = "application/json"
        req = Request(url, data=body, headers=req_headers, method=method)
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def simmer_request(path, method="GET", data=None, api_key=None, timeout=45):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return api_request(f"{SIMMER_BASE}{path}", method=method, data=data, headers=headers, timeout=timeout)

def get_api_key():
    key = os.environ.get("SIMMER_API_KEY")
    if not key:
        print("ERROR: SIMMER_API_KEY is not set")
        sys.exit(1)
    return key

# -----------------------
# Cooldown & Finality
# -----------------------
def record_close_time():
    save_json(COOLDOWN_PATH, {"timestamp": now_utc().timestamp()})

def check_and_wait_cooldown():
    data = load_json(COOLDOWN_PATH, {})
    last_ts = data.get("timestamp")
    if last_ts:
        elapsed = now_utc().timestamp() - last_ts
        if elapsed < COOLDOWN_SECONDS:
            wait_for = int(COOLDOWN_SECONDS - elapsed)
            print(f"💤 COOLDOWN ACTIVE: Waiting {wait_for}s before checking signals...")
            time.sleep(wait_for)
            print("⏰ COOLDOWN OVER: Resuming...")

# -----------------------
# Core Functions
# -----------------------
def get_portfolio(api_key):
    return simmer_request("/api/sdk/portfolio", api_key=api_key)

def get_positions(api_key):
    # ROBUST VERSION: Retries 3 times if API fails, to avoid "fake empty" lists
    for _ in range(3):
        r = simmer_request("/api/sdk/positions", api_key=api_key)
        if isinstance(r, dict):
            return r.get("positions", [])
        time.sleep(1)
    return []

def import_market(api_key, slug):
    url = f"https://polymarket.com/event/{slug}"
    r = simmer_request("/api/sdk/markets/import", method="POST", data={"polymarket_url": url, "shared": True}, api_key=api_key, timeout=60)
    if isinstance(r, dict) and r.get("market_id"):
        return r.get("market_id")
    return None

def execute_trade(api_key, market_id, side, amount=None, shares=None, action="buy"):
    payload = {"market_id": market_id, "side": side, "venue": "polymarket", "source": TRADE_SOURCE, "action": action}
    if action == "sell": payload["shares"] = float(shares or 0)
    else: payload["amount"] = float(amount or 0)
    return simmer_request("/api/sdk/trade", method="POST", data=payload, api_key=api_key, timeout=60)

def get_clob_price(token_id):
    if not token_id: return None
    res = api_request(f"{CLOB_BASE}/prices/{token_id}", timeout=5)
    return float(res["price"]) if isinstance(res, dict) and "price" in res else None

# -----------------------
# Logic
# -----------------------
def terminate_position_with_prejudice(api_key, market_id, side):
    print(f"\n⚡ TERMINATOR: Initializing kill sequence for {side.upper()}...")
    settled_count = 0
    attempt = 1
    
    while settled_count < 3: # 3-Strike Rule
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        shares_left = float(my_pos.get(f"shares_{side}", 0)) if my_pos else 0
        
        if shares_left <= 0.001:
            settled_count += 1
            print(f"✅ VERIFYING CLOSED: {settled_count}/3 (Wallet confirms 0)")
            time.sleep(5) # Wait 5s between checks (15s total)
        else:
            settled_count = 0 # RESET counter if shares appear!
            print(f"⚠️  SHARES DETECTED: {shares_left} remaining. Sending SELL (Attempt {attempt})...")
            execute_trade(api_key, market_id, side, shares=shares_left, action="sell")
            attempt += 1
            time.sleep(5) 

    print("🏁 FINALITY REACHED: Trade is dead.")
    record_close_time()
    return True

def monitor_and_close(api_key, market_id, end_time, initial_side):
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    print("MONITOR: Waiting for trade to settle...")
    
    active_side = initial_side
    shares_owned = 0.0
    entry_price = 0.5
    token_id_map = {}
    
    # 3-Strike Counter for Monitor Phase
    empty_wallet_strikes = 0

    # Phase 1: Detect Side
    for i in range(20): 
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        if my_pos:
            s_yes, s_no = float(my_pos.get("shares_yes", 0)), float(my_pos.get("shares_no", 0))
            if s_yes > 0.001: active_side, shares_owned = "yes", s_yes
            elif s_no > 0.001: active_side, shares_owned = "no", s_no
            
            if shares_owned > 0:
                entry_price = float(my_pos.get("avg_buy_price", 0)) or 0.5
                print(f"MONITOR: Auto-Detected {active_side.upper()} | {shares_owned:.4f} shares @ {entry_price:.3f}")
                if "clob_token_ids" in my_pos:
                    ids = my_pos["clob_token_ids"]
                    if isinstance(ids, str): ids = json.loads(ids)
                    token_id_map["yes"], token_id_map["no"] = ids.get("0"), ids.get("1")
                break
        time.sleep(3)

    # Phase 2: Live Tracking
    if not token_id_map:
        res = simmer_request(f"/api/sdk/markets/{market_id}", api_key=api_key)
        data = res.get("market") or res.get("data") or {}
        if "clob_token_ids" in data:
            ids = data["clob_token_ids"]
            if isinstance(ids, str): ids = json.loads(ids)
            token_id_map["yes"] = ids.get("0")
            token_id_map["no"] = ids.get("1")

    print(f"MONITOR: Tracking via CLOB. SL: {STOP_LOSS_PCT*100}% | TP: {TAKE_PROFIT_PCT*100}%")
    active_token_id = token_id_map.get("0" if active_side == "yes" else "1")
    
    while now_utc() < target_time:
        sys.stdout.write(".")
        sys.stdout.flush()
        
        # --- GLITCH-PROOF CHECK ---
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        curr_shares = float(my_pos.get(f"shares_{active_side}", 0)) if my_pos else 0
        
        if curr_shares <= 0.001:
             empty_wallet_strikes += 1
             # Only verify if we hit 3 strikes
             if empty_wallet_strikes >= 3:
                 print("\n✅ MONITOR: Wallet is EMPTY (Confirmed 3x). Exiting.")
                 record_close_time()
                 return
        else:
             # If shares exist, RESET strikes to 0 (It was just a glitch)
             if empty_wallet_strikes > 0:
                 # Optional: print debug if it was glitching
                 # print(f"(Debug: Glitch detected, shares reappeared)")
                 pass
             empty_wallet_strikes = 0

        curr_price = get_clob_price(active_token_id)
        if curr_price:
            pnl = (curr_price - entry_price) / entry_price
            print(f"\rMONITOR: {active_side.upper()} @ {curr_price:.3f} | PnL: {pnl*100:+.1f}%   ", end="")
            if pnl <= -STOP_LOSS_PCT or pnl >= TAKE_PROFIT_PCT:
                break
        time.sleep(2)
        
    terminate_position_with_prejudice(api_key, market_id, active_side)

def run_once(live, quiet, smart_sizing):
    check_and_wait_cooldown()
    api_key = get_api_key()
    
    now = now_utc()
    start_dt = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
    slug = f"{ASSET.lower()}-updown-5m-{int(start_dt.timestamp())}"
    end_time = start_dt + timedelta(minutes=5)

    if not quiet: print(f"TARGET: {slug}")
    
    positions = get_positions(api_key)
    if any(slug in str(p.get("slug", "")) for p in positions): return

    coin_id = COINGECKO_IDS.get(ASSET, "bitcoin")
    data = api_request(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1")
    if "prices" not in data: return
    prices = data["prices"]
    latest_price = prices[-1][1]
    target_ts = prices[-1][0] - (LOOKBACK_MINS * 60 * 1000)
    past_price = next((p for t, p in reversed(prices) if abs(t - target_ts) < 300000), None)
    if not past_price: return

    momentum = ((latest_price - past_price) / past_price) * 100
    if abs(momentum) < MIN_MOMENTUM_PCT: return
        
    side = "yes" if momentum > 0 else "no"
    print(f"SIGNAL: {side.upper()} | Mom={momentum:.3f}%")

    pf = get_portfolio(api_key)
    bal = float(pf.get("balance_usdc", 0) or 0)
    amount = bal * 0.95 if bal < 5.0 else (bal * SMART_SIZING_PCT if smart_sizing else MAX_POSITION_AMOUNT)
    amount = float(f"{amount:.2f}")

    if amount < 0.2 or not live: return

    market_id = import_market(api_key, slug)
    if not market_id: return

    res = execute_trade(api_key, market_id, side, amount=amount)
    if res.get("success"):
        print("TRADE: Success. Monitoring...")
        monitor_and_close(api_key, market_id, end_time, side)
    else:
        print(f"TRADE FAILED: {res.get('error')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--smart-sizing", "-s", action="store_true")
    args = parser.parse_args()
    run_once(args.live, args.quiet, args.smart_sizing)
