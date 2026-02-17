#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (v8.5 - Persistent Cooldown).

FIXES:
- ✅ PERSISTENT COOLDOWN: Uses a file in /data to enforce a 60s wait between trades.
- ✅ RESTART PROOF: Even if Railway reruns the bot, it will wait if it traded recently.
- ✅ CLOB SYNC: Keeps the real-time pricing and auto-detection logic.
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
MAX_POSITION_AMOUNT = 3.0    
SMART_SIZING_PCT = 0.95      

STOP_LOSS_PCT = 0.15         # Adjusted to your log (15%)
TAKE_PROFIT_PCT = 0.20       # Adjusted to your log (20%)
CLOSE_BUFFER_SECONDS = 120    
COOLDOWN_SECONDS = 60        # Wait 60s after a trade before allowing a new one
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
        req_headers.setdefault("User-Agent", "railway-fastloop/8.5")
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
# Cooldown Management
# -----------------------
def record_close_time():
    """Saves the current time as the last trade close."""
    save_json(COOLDOWN_PATH, {"timestamp": now_utc().timestamp()})

def check_and_wait_cooldown():
    """Reads the last close time and sleeps if within cooldown window."""
    data = load_json(COOLDOWN_PATH, {})
    last_ts = data.get("timestamp")
    if not last_ts:
        return

    elapsed = now_utc().timestamp() - last_ts
    if elapsed < COOLDOWN_SECONDS:
        wait_for = int(COOLDOWN_SECONDS - elapsed)
        print(f"💤 COOLDOWN: Last trade was {int(elapsed)}s ago. Waiting {wait_for}s for API reset...")
        time.sleep(wait_for)
        print("🚀 COOLDOWN FINISHED: Resuming discovery.")

# -----------------------
# Core Functions
# -----------------------
def get_portfolio(api_key):
    return simmer_request("/api/sdk/portfolio", api_key=api_key)

def get_positions(api_key):
    r = simmer_request("/api/sdk/positions", api_key=api_key)
    return r.get("positions", []) if isinstance(r, dict) else []

def find_simmer_market_broad(api_key, slug):
    r = simmer_request(f"/api/sdk/markets?limit=100&search={slug}", api_key=api_key)
    if isinstance(r, dict) and "markets" in r:
        for m in r["markets"]:
            if slug in str(m.get("slug", "")) or slug in str(m.get("polymarket_url", "")):
                return m.get("id")
    return None

def import_market(api_key, slug):
    existing_id = find_simmer_market_broad(api_key, slug)
    if existing_id: return existing_id, None, True
    
    url = f"https://polymarket.com/event/{slug}"
    for _ in range(3):
        r = simmer_request("/api/sdk/markets/import", method="POST", data={"polymarket_url": url, "shared": True}, api_key=api_key, timeout=60)
        if isinstance(r, dict) and r.get("status") in ["imported", "already_exists"]:
            return r.get("market_id"), None, True
        time.sleep(2)
    return None, "import_failed", False

def execute_trade(api_key, market_id, side, amount=None, shares=None, action="buy"):
    payload = {"market_id": market_id, "side": side, "venue": "polymarket", "source": TRADE_SOURCE, "action": action}
    if action == "sell": payload["shares"] = float(shares or 0)
    else: payload["amount"] = float(amount or 0)
    
    for _ in range(3):
        res = simmer_request("/api/sdk/trade", method="POST", data=payload, api_key=api_key, timeout=60)
        if res and res.get("success"): return res
        time.sleep(1)
    return {"error": "max_retries"}

def get_clob_price(token_id):
    if not token_id: return None
    url = f"{CLOB_BASE}/prices/{token_id}"
    res = api_request(url, timeout=5)
    if isinstance(res, dict) and "price" in res:
        try: return float(res["price"])
        except: pass
    return None

# -----------------------
# Logic
# -----------------------
def terminate_position_with_prejudice(api_key, market_id, side):
    print(f"\n⚡ TERMINATOR ENGAGED: Checking wallet for {side.upper()} shares...")
    attempt = 1
    while True:
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        
        if not my_pos:
            print("✅ TERMINATOR: Position is completely gone.")
            record_close_time() # Save the time!
            return True
            
        shares_left = float(my_pos.get(f"shares_{side}", 0))
        if shares_left <= 0.001:
             print("✅ TERMINATOR: Shares at 0.")
             record_close_time() # Save the time!
             return True

        print(f"⚠️  SHARES DETECTED: {shares_left} remaining. Sending SELL (Attempt {attempt})...")
        execute_trade(api_key, market_id, side, shares=shares_left, action="sell")
        attempt += 1
        time.sleep(2.0) 

def monitor_and_close(api_key, market_id, end_time, initial_side):
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    print("MONITOR: Waiting for trade to settle...")
    
    active_side = initial_side
    shares_owned = 0.0
    entry_price = 0.5
    token_id_map = {}

    for i in range(20): 
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        
        if my_pos:
            s_yes = float(my_pos.get("shares_yes", 0))
            s_no = float(my_pos.get("shares_no", 0))
            
            if s_yes > 0.001:
                active_side = "yes"
                shares_owned = s_yes
                entry_price = float(my_pos.get("avg_buy_price", 0)) or 0.5
            elif s_no > 0.001:
                active_side = "no"
                shares_owned = s_no
                entry_price = float(my_pos.get("avg_buy_price", 0)) or 0.5
            
            if shares_owned > 0:
                print(f"MONITOR: Auto-Detected {active_side.upper()} position: {shares_owned:.4f} shares @ {entry_price:.3f}")
                if "clob_token_ids" in my_pos:
                    ids = my_pos["clob_token_ids"]
                    if isinstance(ids, str): ids = json.loads(ids)
                    token_id_map["yes"] = ids.get("0")
                    token_id_map["no"] = ids.get("1")
                break
        
        print(f"MONITOR: Waiting for shares... ({i+1}/20)")
        time.sleep(3)

    if shares_owned <= 0.001:
        print(f"✅ MONITOR: No shares found. Trade ALREADY CLOSED.")
        record_close_time()
        return

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
        
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        curr_shares = float(my_pos.get(f"shares_{active_side}", 0)) if my_pos else 0
        if curr_shares <= 0.001:
             print("\n✅ MONITOR: Shares dropped to 0. Trade Closed.")
             record_close_time()
             return

        curr_price = get_clob_price(active_token_id)
        if curr_price is not None:
            pnl = (curr_price - entry_price) / entry_price
            print(f"\rMONITOR: {active_side.upper()} @ {curr_price:.3f} | PnL: {pnl*100:+.1f}%   ", end="")
            if pnl <= -STOP_LOSS_PCT or pnl >= TAKE_PROFIT_PCT:
                break
        time.sleep(2)
        
    print("") 
    terminate_position_with_prejudice(api_key, market_id, active_side)

def run_once(live, quiet, smart_sizing):
    # --- CHECK COOLDOWN FIRST ---
    check_and_wait_cooldown()

    api_key = get_api_key()
    
    # 1. Discovery
    now = now_utc()
    minute = (now.minute // 5) * 5
    start_dt = now.replace(minute=minute, second=0, microsecond=0)
    slug = f"{ASSET.lower()}-updown-5m-{int(start_dt.timestamp())}"
    end_time = start_dt + timedelta(minutes=5)

    if not quiet: print(f"TARGET: {slug} (Ends {end_time.strftime('%H:%M')})")
    
    # 2. Existing Check
    positions = get_positions(api_key)
    for p in positions:
        if slug in str(p.get("slug", "")) or slug in str(p.get("polymarket_url", "")):
            if not quiet: print("SKIP: Already in market.")
            return

    # 3. Signal
    coin_id = COINGECKO_IDS.get(ASSET, "bitcoin")
    data = api_request(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1")
    if "prices" not in data: return
    prices = data["prices"]
    latest_ts, latest_price = prices[-1]
    target_ts = latest_ts - (LOOKBACK_MINS * 60 * 1000)
    past_price = next((p for t, p in reversed(prices) if abs(t - target_ts) < 300000), None)
    if not past_price: return

    momentum = ((latest_price - past_price) / past_price) * 100
    if abs(momentum) < MIN_MOMENTUM_PCT:
        if not quiet: print(f"SKIP: Weak momentum {momentum:.3f}%")
        return
        
    side = "yes" if momentum > 0 else "no"
    print(f"SIGNAL: {side.upper()} | Mom={momentum:.3f}% | Price={latest_price:.2f}")

    # 4. Sizing
    pf = get_portfolio(api_key)
    bal = float(pf.get("balance_usdc", 0) or 0)
    amount = bal * 0.95 if bal < 5.0 else (bal * SMART_SIZING_PCT if smart_sizing else MAX_POSITION_AMOUNT)
    amount = float(f"{amount:.2f}")
    if amount < 0.2: return

    # 5. Execute
    market_id, _, _ = import_market(api_key, slug)
    if not market_id: return
    if not live: return

    res = execute_trade(api_key, market_id, side, amount=amount)
    if res.get("success"):
        print("TRADE: Success. Monitoring...")
        monitor_and_close(api_key, market_id, end_time, side)
        print("CYCLE COMPLETE: Closed.")
    else:
        print(f"TRADE FAILED: {res.get('error')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--smart-sizing", "-s", action="store_true")
    args = parser.parse_args()
    run_once(args.live, args.quiet, args.smart_sizing)
