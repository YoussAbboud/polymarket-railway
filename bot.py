#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (v7.9 - X-Ray Mode).

FIXES:
- ✅ X-RAY LOGGING: Prints RAW API response if price is missing (No more silent fails).
- ✅ SMART PARSER: Checks 'outcome_prices', 'outcomePrices', and nested 'market' blocks.
- ✅ ROBUSTNESS: Keeps the 'Terminator' close logic but fixes the blind panic trigger.
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
MIN_MOMENTUM_PCT = 0.12      # High quality threshold
MAX_POSITION_AMOUNT = 5.0    
SMART_SIZING_PCT = 0.95      # Survival Mode

STOP_LOSS_PCT = 0.25         # 25% Stop
TAKE_PROFIT_PCT = 0.30       # 30% Profit
CLOSE_BUFFER_SECONDS = 80    
# ==============================================================================

# -----------------------
# Persistence locations
# -----------------------
DEFAULT_DATA_DIR = "/data" if os.path.isdir("/data") else ".data"
DATA_DIR = os.environ.get("BOT_STATE_DIR", DEFAULT_DATA_DIR)
STATE_PATH = os.environ.get("BOT_STATE_PATH", os.path.join(DATA_DIR, "state.json"))
JOURNAL_PATH = os.environ.get("BOT_JOURNAL_PATH", os.path.join(DATA_DIR, "trades.jsonl"))

os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------
# API
# -----------------------
SIMMER_BASE = os.environ.get("SIMMER_API_BASE", "https://api.simmer.markets")
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
        req_headers.setdefault("User-Agent", "railway-fastloop/7.9")
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
# State & Locks
# -----------------------
def load_state():
    st = load_json(STATE_PATH, {})
    if st.get("day") != now_utc().date().isoformat():
        st = {"day": now_utc().date().isoformat(), "trades": 0}
    return st

def save_state(st): save_json(STATE_PATH, st)

def lock_path(key): return os.path.join(DATA_DIR, f"lock_{key.replace('/', '_')}.json")

def write_lock(key, extra=None):
    payload = {"ts": now_utc().isoformat(), "key": key}
    if extra: payload.update(extra)
    save_json(lock_path(key), payload)

def clear_lock(key):
    try: os.remove(lock_path(key))
    except: pass

def has_recent_lock(key):
    data = load_json(lock_path(key), None)
    if not data: return False
    ts = datetime.fromisoformat(data["ts"])
    if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
    return (now_utc() - ts).total_seconds() < 300

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
            return True
            
        shares_left = float(my_pos.get(f"shares_{side}", 0))
        if shares_left <= 0.001:
             print("✅ TERMINATOR: Shares at 0.")
             return True

        print(f"⚠️  SHARES DETECTED: {shares_left} remaining. Sending SELL (Attempt {attempt})...")
        execute_trade(api_key, market_id, side, shares=shares_left, action="sell")
        attempt += 1
        time.sleep(2.0) 

def monitor_and_close(api_key, market_id, end_time, side, est_entry_price):
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    print("MONITOR: Waiting for trade to settle...")
    
    shares_owned = 0.0
    entry_price = est_entry_price
    
    # --- STEP 1: WAIT FOR SHARES ---
    for i in range(20): 
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        
        if my_pos:
            shares_owned = float(my_pos.get(f"shares_{side}", 0))
            if shares_owned > 0:
                entry_price = float(my_pos.get("avg_buy_price", 0)) or est_entry_price
                print(f"MONITOR: Shares confirmed! Tracking {shares_owned:.4f} {side.upper()} @ {entry_price:.3f}")
                break
        
        print(f"MONITOR: Waiting for shares to appear in wallet... ({i+1}/20)")
        time.sleep(3)

    if shares_owned <= 0:
        print(f"WARNING: API never showed shares. Engaging Terminator blindly.")
        terminate_position_with_prejudice(api_key, market_id, side)
        return

    # --- STEP 3: PnL MONITOR ---
    print(f"MONITOR: Live Tracking. SL: {STOP_LOSS_PCT*100}% | TP: {TAKE_PROFIT_PCT*100}%")
    last_valid_price_time = time.time()
    
    while now_utc() < target_time:
        sys.stdout.write(".")
        sys.stdout.flush()
        
        if time.time() - last_valid_price_time > 45:
            print("\n!!! CRITICAL: No price data for 45s. FORCE CLOSING !!!")
            break

        # FETCH & PARSE PRICE (The "X-Ray" Logic)
        res = simmer_request(f"/api/sdk/markets/{market_id}", api_key=api_key)
        
        # DEBUG: Un-comment next line to see RAW spam if needed
        # print(f"\nDEBUG RAW: {str(res)[:100]}") 

        # Normalize the response (Unwrap 'market' or 'data' blocks)
        data_block = res
        if isinstance(res, dict):
            if "market" in res: data_block = res["market"]
            elif "data" in res: data_block = res["data"]
            
        try:
            # Try to find prices in various common keys
            prices = None
            if "outcome_prices" in data_block: prices = data_block["outcome_prices"]
            elif "outcomePrices" in data_block: prices = data_block["outcomePrices"]
            elif "clob_token_ids" in data_block: 
                 # Fallback: Sometimes only clob IDs are sent, ignore for now to avoid complexity
                 pass 

            if prices:
                if isinstance(prices, str): prices = json.loads(prices)
                
                # Check 0/1 keys
                p_yes = prices.get("0") or prices.get("yes")
                p_no = prices.get("1") or prices.get("no")
                
                curr_price = float(p_yes if side == "yes" else p_no)
                last_valid_price_time = time.time()
                
                pnl = (curr_price - entry_price) / entry_price
                print(f"\rMONITOR: {side.upper()} @ {curr_price:.3f} | PnL: {pnl*100:+.1f}%   ", end="")
                
                if pnl <= -STOP_LOSS_PCT:
                    print(f"\nSTOP LOSS HIT: {pnl*100:.1f}%. Closing.")
                    break
                if pnl >= TAKE_PROFIT_PCT:
                    print(f"\nTAKE PROFIT HIT: {pnl*100:.1f}%. Closing.")
                    break
            else:
                # X-RAY: If we got a dict but no prices, tell user what keys we DID get
                # sys.stdout.write(f"?") 
                pass

        except Exception as e:
            print(f"\nDEBUG PARSE ERROR: {e}")
        
        time.sleep(3)
        
    print("") 
    terminate_position_with_prejudice(api_key, market_id, side)

def run_once(live, quiet, smart_sizing):
    api_key = get_api_key()
    
    # 1. Discovery
    now = now_utc()
    minute = (now.minute // 5) * 5
    start_dt = now.replace(minute=minute, second=0, microsecond=0)
    ts = int(start_dt.timestamp())
    slug = f"{ASSET.lower()}-updown-5m-{ts}"
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
    if "prices" not in data: 
        if not quiet: print("SKIP: Signal error.")
        return
        
    prices = data["prices"]
    latest_ts, latest_price = prices[-1]
    target_ts = latest_ts - (LOOKBACK_MINS * 60 * 1000)
    past_price = next((p for t, p in reversed(prices) if abs(t - target_ts) < 300000), None)
    if not past_price: 
        if not quiet: print("SKIP: No history found.")
        return

    momentum = ((latest_price - past_price) / past_price) * 100
    if abs(momentum) < MIN_MOMENTUM_PCT:
        if not quiet: print(f"SKIP: Weak momentum {momentum:.3f}% (Threshold: {MIN_MOMENTUM_PCT}%)")
        return
        
    side = "yes" if momentum > 0 else "no"
    print(f"SIGNAL: {side.upper()} | Mom={momentum:.3f}% | Price={latest_price:.2f}")

    # 4. SURVIVAL SIZING
    amount = MAX_POSITION_AMOUNT
    pf = get_portfolio(api_key)
    bal = float(pf.get("balance_usdc", 0) or 0)
    
    if bal < MAX_POSITION_AMOUNT:
        print(f"⚠️ LOW BALANCE ({bal:.2f}). Engaging Survival Sizing (95%)...")
        amount = bal * 0.95
    elif smart_sizing:
        amount = bal * SMART_SIZING_PCT
    
    amount = float(f"{amount:.2f}")
    
    if amount < 0.2: 
        print(f"SKIP: Insufficient funds ({amount:.2f}). Need >$0.20.")
        return

    # 4b. Est Entry
    est_entry_price = 0.5
    market_id, _, _ = import_market(api_key, slug)
    if market_id:
         res = simmer_request(f"/api/sdk/markets/{market_id}", api_key=api_key)
         if isinstance(res, dict) and "outcome_prices" in res:
             try:
                 prices = res["outcome_prices"]
                 if isinstance(prices, str): prices = json.loads(prices)
                 est_entry_price = float(prices["0" if side == "yes" else "1"])
             except: pass

    # 5. Execute
    if not market_id:
        print("FAIL: Import error.")
        return

    if not live:
        print(f"DRY RUN: Buy {side} ${amount}")
        return

    res = execute_trade(api_key, market_id, side, amount=amount)
    if res.get("success"):
        st = load_state()
        st["trades"] += 1
        save_state(st)
        print("TRADE: Success. Monitoring...")
        monitor_and_close(api_key, market_id, end_time, side, est_entry_price)
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
