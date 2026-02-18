#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (v9.2 - Strict Verification).

CHANGES:
- ✅ ENTRY VERIFICATION: Bot waits up to 60s to CONFIRM shares are in wallet before monitoring.
- ✅ EXIT VERIFICATION: Bot loops until wallet confirms 0 shares (retries sell if needed).
- ✅ PER-SECOND DEBUG: Prints PnL and Price every second.
- ✅ SAFETY: No more "assuming" trades worked. We verify everything.
"""

import os, sys, json, argparse, time
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request

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
COOLDOWN_PATH = os.path.join(DATA_DIR, "last_close.json")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------
# API Config
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
        req_headers.setdefault("User-Agent", "railway-fastloop/9.2")
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
# Cooldown
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
            print(f"💤 COOLDOWN: Waiting {wait_for}s...")
            time.sleep(wait_for)

# -----------------------
# Core Logic
# -----------------------
def get_portfolio(api_key):
    return simmer_request("/api/sdk/portfolio", api_key=api_key)

def get_positions(api_key):
    # Retry logic to avoid network blips causing "0 shares" errors
    for _ in range(3):
        r = simmer_request("/api/sdk/positions", api_key=api_key)
        if isinstance(r, dict): return r.get("positions", [])
        time.sleep(1)
    return []

def import_market(api_key, slug):
    url = f"https://polymarket.com/event/{slug}"
    r = simmer_request("/api/sdk/markets/import", method="POST", data={"polymarket_url": url, "shared": True}, api_key=api_key, timeout=60)
    if isinstance(r, dict) and r.get("market_id"): return r.get("market_id")
    return None

def execute_trade(api_key, market_id, side, amount=None, shares=None, action="buy"):
    payload = {"market_id": market_id, "side": side, "venue": "polymarket", "source": TRADE_SOURCE, "action": action}
    if action == "sell": payload["shares"] = float(shares or 0)
    else: payload["amount"] = float(amount or 0)
    return simmer_request("/api/sdk/trade", method="POST", data=payload, api_key=api_key, timeout=60)

def get_clob_price(token_id):
    if not token_id: return None
    res = api_request(f"{CLOB_BASE}/prices/{token_id}", timeout=2)
    return float(res["price"]) if isinstance(res, dict) and "price" in res else None

# -----------------------
# Verification Functions
# -----------------------
def wait_for_entry(api_key, market_id, side):
    """Loops until shares appear in the wallet."""
    print("🕵️ VERIFYING ENTRY: Checking wallet for shares...")
    for i in range(20): # Check for 60 seconds (20 * 3s)
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        
        if my_pos:
            s_yes = float(my_pos.get("shares_yes", 0))
            s_no = float(my_pos.get("shares_no", 0))
            
            # Check if we have shares for the requested side
            shares_found = s_yes if side == "yes" else s_no
            if shares_found > 0.001:
                entry_price = float(my_pos.get("avg_buy_price", 0)) or 0.5
                
                # Get Token ID for pricing
                token_id = None
                if "clob_token_ids" in my_pos:
                    ids = my_pos["clob_token_ids"]
                    if isinstance(ids, str): ids = json.loads(ids)
                    token_id = ids.get("0" if side == "yes" else "1")
                
                # Fallback Token ID check
                if not token_id:
                     res = simmer_request(f"/api/sdk/markets/{market_id}", api_key=api_key)
                     data = res.get("market") or res.get("data") or {}
                     if "clob_token_ids" in data:
                        ids = data["clob_token_ids"]
                        if isinstance(ids, str): ids = json.loads(ids)
                        token_id = ids.get("0" if side == "yes" else "1")

                print(f"✅ ENTRY CONFIRMED: {shares_found} {side.upper()} shares @ {entry_price:.3f}")
                return shares_found, entry_price, token_id
                
        print(f"⏳ Waiting for shares to arrive... ({i+1}/20)")
        time.sleep(3)
    
    return None, None, None

def ensure_exit(api_key, market_id, side, known_shares):
    """Loops until shares are gone. Retries sell if they stick around."""
    print(f"\n⚡ VERIFYING EXIT: Closing {known_shares} {side.upper()} shares...")
    
    # 1. Send Initial Sell
    execute_trade(api_key, market_id, side, shares=known_shares, action="sell")
    
    # 2. Monitor for Exit (Terminator Loop)
    attempt = 1
    zero_count = 0 # We want 3 consecutive '0' readings to be sure
    
    while zero_count < 3:
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        
        current_shares = float(my_pos.get(f"shares_{side}", 0)) if my_pos else 0
        
        if current_shares <= 0.001:
            zero_count += 1
            print(f"✅ CONFIRMING EXIT: {zero_count}/3 (Wallet Empty)")
            time.sleep(2)
        else:
            zero_count = 0 # Reset count if shares reappear
            print(f"⚠️  SHARES REMAIN: {current_shares}. Retrying SELL (Attempt {attempt})...")
            execute_trade(api_key, market_id, side, shares=current_shares, action="sell")
            attempt += 1
            time.sleep(3) # Give it time to process
            
    print("🏁 TRADE CLOSED: Confirmed.")
    record_close_time()

def monitor_position(api_key, market_id, end_time, side, shares, entry, token_id):
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    print(f"📊 MONITORING: SL {STOP_LOSS_PCT*100}% | TP {TAKE_PROFIT_PCT*100}%")
    
    while now_utc() < target_time:
        curr_price = get_clob_price(token_id)
        
        if curr_price:
            pnl = (curr_price - entry) / entry
            print(f"⏱️ {side.upper()} | Price: {curr_price:.3f} | PnL: {pnl*100:+.1f}%")
            
            if pnl <= -STOP_LOSS_PCT:
                print(f"🛑 STOP LOSS: {pnl*100:.1f}%")
                break
            if pnl >= TAKE_PROFIT_PCT:
                print(f"💰 TAKE PROFIT: {pnl*100:.1f}%")
                break
        else:
            # If CLOB fails, we just print a dot and keep waiting
            sys.stdout.write(".")
            sys.stdout.flush()
        
        time.sleep(1)
        
    # Exit Phase
    ensure_exit(api_key, market_id, side, shares)

def run_once(live, quiet, smart_sizing):
    check_and_wait_cooldown()
    api_key = get_api_key()
    
    now = now_utc()
    start_dt = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
    slug = f"{ASSET.lower()}-updown-5m-{int(start_dt.timestamp())}"
    end_time = start_dt + timedelta(minutes=5)

    if not quiet: print(f"TARGET: {slug}")
    
    # Pre-check
    positions = get_positions(api_key)
    if any(slug in str(p.get("slug", "")) for p in positions): 
        print("SKIP: Already in market.")
        return

    # Signal
    coin_id = COINGECKO_IDS.get(ASSET, "bitcoin")
    data = api_request(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1")
    if "prices" not in data: return
    prices = data["prices"]
    latest_price = prices[-1][1]
    target_ts = prices[-1][0] - (LOOKBACK_MINS * 60 * 1000)
    past_price = next((p for t, p in reversed(prices) if abs(t - target_ts) < 300000), None)
    if not past_price: return

    momentum = ((latest_price - past_price) / past_price) * 100
    if abs(momentum) < MIN_MOMENTUM_PCT: 
        print(f"SKIP: Low Mom {momentum:.3f}%")
        return
        
    side = "yes" if momentum > 0 else "no"
    print(f"SIGNAL: {side.upper()} | Mom={momentum:.3f}%")

    # Sizing
    pf = get_portfolio(api_key)
    bal = float(pf.get("balance_usdc", 0) or 0)
    amount = bal * 0.95
    if amount < 0.2: 
        print("SKIP: Low Balance")
        return

    if not live: return

    # Execute
    market_id = import_market(api_key, slug)
    if not market_id: return

    print(f"🚀 SENDING ORDER: {side.upper()} ${amount}...")
    res = execute_trade(api_key, market_id, side, amount=amount)
    
    if res.get("success"):
        # STRICT VERIFICATION: Wait for shares to appear
        shares, entry, token_id = wait_for_entry(api_key, market_id, side)
        
        if shares:
            monitor_position(api_key, market_id, end_time, side, shares, entry, token_id)
        else:
            print("❌ ERROR: Trade sent but no shares appeared after 60s. Aborting.")
    else:
        print(f"❌ ORDER FAILED: {res.get('error')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--smart-sizing", "-s", action="store_true")
    args = parser.parse_args()
    run_once(args.live, args.quiet, args.smart_sizing)
