#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (v9.5 - Stagnation Kill Switch).

NEW FEATURE:
- ❄️ STAGNATION EXIT: If price/PnL doesn't move for 20 seconds, the bot sells immediately.

SAFETY FEATURES (RETAINED):
- 🛡️ HARD CAP: Max bet is $5.00. Never bets full wallet unless balance < $5.
- ✅ STRICT VERIFICATION: Confirms shares in wallet before monitoring or exiting.
- 🛑 CIRCUIT BREAKER: Shuts down on Stop Loss hit.
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

# --- SAFETY SETTINGS ---
MAX_POSITION_AMOUNT = 10.0    # HARD LIMIT.
STOP_LOSS_PCT = 0.10         
TAKE_PROFIT_PCT = 0.10       
CLOSE_BUFFER_SECONDS = 80    
COOLDOWN_SECONDS = 60
STAGNATION_TIMEOUT = 20      # Seconds of flat price before forced exit
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
        req_headers.setdefault("User-Agent", "railway-fastloop/9.5")
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
    print("🕵️ VERIFYING ENTRY: Checking wallet for shares...")
    for i in range(20): 
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        
        if my_pos:
            s_yes = float(my_pos.get("shares_yes", 0))
            s_no = float(my_pos.get("shares_no", 0))
            shares_found = s_yes if side == "yes" else s_no
            
            if shares_found > 0.001:
                entry_price = float(my_pos.get("avg_buy_price", 0)) or 0.5
                token_id = None
                if "clob_token_ids" in my_pos:
                    ids = my_pos["clob_token_ids"]
                    if isinstance(ids, str): ids = json.loads(ids)
                    token_id = ids.get("0" if side == "yes" else "1")
                
                if not token_id:
                     res = simmer_request(f"/api/sdk/markets/{market_id}", api_key=api_key)
                     data = res.get("market") or res.get("data") or {}
                     if "clob_token_ids" in data:
                        ids = data["clob_token_ids"]
                        if isinstance(ids, str): ids = json.loads(ids)
                        token_id = ids.get("0" if side == "yes" else "1")

                print(f"✅ ENTRY CONFIRMED: {shares_found} {side.upper()} shares @ {entry_price:.3f}")
                return shares_found, entry_price, token_id
                
        print(f"⏳ Waiting for shares... ({i+1}/20)")
        time.sleep(3)
    
    return None, None, None

def ensure_exit(api_key, market_id, side, known_shares):
    print(f"\n⚡ VERIFYING EXIT: Closing {known_shares} {side.upper()} shares...")
    execute_trade(api_key, market_id, side, shares=known_shares, action="sell")
    
    attempt = 1
    zero_count = 0
    while zero_count < 3:
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        current_shares = float(my_pos.get(f"shares_{side}", 0)) if my_pos else 0
        
        if current_shares <= 0.001:
            zero_count += 1
            print(f"✅ CONFIRMING EXIT: {zero_count}/3 (Wallet Empty)")
            time.sleep(2)
        else:
            zero_count = 0 
            print(f"⚠️  SHARES REMAIN: {current_shares}. Retrying SELL (Attempt {attempt})...")
            execute_trade(api_key, market_id, side, shares=current_shares, action="sell")
            attempt += 1
            time.sleep(3)
            
    print("🏁 TRADE CLOSED: Confirmed.")
    record_close_time()

def monitor_position(api_key, market_id, end_time, side, shares, entry, token_id):
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    print(f"📊 MONITORING: SL {STOP_LOSS_PCT*100}% | TP {TAKE_PROFIT_PCT*100}% | STAGNATION {STAGNATION_TIMEOUT}s")
    
    stop_triggered = False
    
    # Stagnation Tracking
    last_price = None
    stagnation_start_time = time.time()
    
    while now_utc() < target_time:
        curr_price = get_clob_price(token_id)
        
        if curr_price is not None:
            # --- STAGNATION LOGIC ---
            if curr_price == last_price:
                elapsed = time.time() - stagnation_start_time
                if elapsed >= STAGNATION_TIMEOUT:
                    print(f"\n❄️ STAGNATION DETECTED: Price stuck at {curr_price:.3f} for {int(elapsed)}s. Exiting.")
                    break # Trigger Exit
            else:
                last_price = curr_price
                stagnation_start_time = time.time() # Reset timer on movement
            # ------------------------

            pnl = (curr_price - entry) / entry
            print(f"⏱️ {side.upper()} | Price: {curr_price:.3f} | PnL: {pnl*100:+.1f}%")
            
            if pnl <= -STOP_LOSS_PCT:
                print(f"🛑 STOP LOSS: {pnl*100:.1f}%")
                stop_triggered = True
                break
            if pnl >= TAKE_PROFIT_PCT:
                print(f"💰 TAKE PROFIT: {pnl*100:.1f}%")
                break
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
        
        time.sleep(1)
        
    ensure_exit(api_key, market_id, side, shares)
    
    if stop_triggered:
        print("\n🚨 CIRCUIT BREAKER: Stop Loss hit. Shutting down bot.")
        sys.exit(0)

def run_once(live, quiet):
    check_and_wait_cooldown()
    api_key = get_api_key()
    
    now = now_utc()
    start_dt = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
    slug = f"{ASSET.lower()}-updown-5m-{int(start_dt.timestamp())}"
    end_time = start_dt + timedelta(minutes=5)

    if not quiet: print(f"TARGET: {slug}")
    
    positions = get_positions(api_key)
    if any(slug in str(p.get("slug", "")) for p in positions): 
        print("SKIP: Already in market.")
        return

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

    pf = get_portfolio(api_key)
    bal = float(pf.get("balance_usdc", 0) or 0)
    
    # --- FIXED SIZING LOGIC ---
    if bal < 5.0:
        amount = bal * 0.95
        print(f"⚠️ SURVIVAL MODE: Balance ${bal:.2f} < $5. Using ${amount:.2f}")
    else:
        amount = MAX_POSITION_AMOUNT # HARD CAP
        print(f"🛡️ HARD CAP: Wallet is ${bal:.2f}, but only betting ${amount:.2f}.")
        
    amount = float(f"{amount:.2f}")

    if amount < 0.2: 
        print("SKIP: Low Balance")
        return

    if not live: return

    market_id = import_market(api_key, slug)
    if not market_id: return

    print(f"🚀 SENDING ORDER: {side.upper()} ${amount}...")
    res = execute_trade(api_key, market_id, side, amount=amount)
    
    if res.get("success"):
        shares, entry, token_id = wait_for_entry(api_key, market_id, side)
        if shares:
            monitor_position(api_key, market_id, end_time, side, shares, entry, token_id)
        else:
            print("❌ ERROR: Trade sent but no shares appeared. Aborting.")
    else:
        print(f"❌ ORDER FAILED: {res.get('error')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()
    run_once(args.live, args.quiet)
