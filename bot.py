#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (v6.6 - Fix & Simple Close).

FIXES:
- ✅ CRITICAL FIX: Restored the missing `import_market` function.
- ✅ SIMPLE LOGIC: Removed complex wallet checking.
  -> Now monitors PnL -> Hits Threshold -> Retries Sell Order until Success.
- ✅ STRICT MONITOR: Tracks the specific side you bought.
"""

import os, sys, json, argparse, time
import math
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

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

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

TRADE_SOURCE = "railway:fastloop"
LOCK_TTL_SECONDS = int(os.environ.get("LOCK_TTL_SECONDS", "300"))

# --- STRATEGY SETTINGS ---
CLOSE_BUFFER_SECONDS = 80   # Close 80s before end
STOP_LOSS_PCT = 0.05        # -5% Loss
TAKE_PROFIT_PCT = 0.15      # +15% Profit

# -----------------------
# Helpers
# -----------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def api_request(url, method="GET", data=None, headers=None, timeout=25):
    try:
        req_headers = headers or {}
        req_headers.setdefault("User-Agent", "railway-fastloop/6.6")
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        
        req = Request(url, data=body, headers=req_headers, method=method)
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return json.loads(raw)
            except Exception:
                return {"error": "non_json_response", "raw": raw[:500]}
    except HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8"))
            return {"error": err.get("detail", str(e)), "status_code": e.code, "body": err}
        except Exception:
            return {"error": str(e), "status_code": e.code}
    except URLError as e:
        return {"error": f"Connection error: {getattr(e, 'reason', e)}"}
    except Exception as e:
        return {"error": str(e)}

def simmer_request(path, method="GET", data=None, api_key=None, timeout=45):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return api_request(
        f"{SIMMER_BASE}{path}",
        method=method,
        data=data,
        headers=headers,
        timeout=timeout
    )

def get_api_key():
    key = os.environ.get("SIMMER_API_KEY")
    if not key:
        print("ERROR: SIMMER_API_KEY is not set")
        sys.exit(1)
    return key

# -----------------------
# Lock helpers
# -----------------------
def lock_path_for(key: str) -> str:
    safe = (key or "").replace("/", "_").replace(" ", "_").replace(":", "_")
    return os.path.join(DATA_DIR, f"lock_{safe}.json")

def read_lock(key: str):
    path = lock_path_for(key)
    data = load_json(path, None)
    if not isinstance(data, dict) or "ts" not in data: return None
    try:
        ts = datetime.fromisoformat(data["ts"])
        if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
        if (now_utc() - ts).total_seconds() < LOCK_TTL_SECONDS: return data
    except Exception: return None
    return None

def has_recent_lock(key: str) -> bool: return read_lock(key) is not None

def write_lock(key: str, extra: dict = None):
    path = lock_path_for(key)
    payload = {"ts": now_utc().isoformat(), "key": key}
    if isinstance(extra, dict): payload.update(extra)
    save_json(path, payload)

def clear_lock(key: str):
    path = lock_path_for(key)
    try:
        if os.path.exists(path): os.remove(path)
    except Exception: pass

# -----------------------
# State helpers
# -----------------------
def load_state():
    st = load_json(STATE_PATH, {})
    if not isinstance(st, dict): st = {}
    today = now_utc().date().isoformat()
    if st.get("day") != today:
        st = {"day": today, "trades": 0, "imports": 0, "last_trade_ts": None}
    return st

def save_state(st: dict):
    save_json(STATE_PATH, st)

# -----------------------
# Simmer actions
# -----------------------
def get_portfolio(api_key: str):
    return simmer_request("/api/sdk/portfolio", api_key=api_key, timeout=45)

def get_positions(api_key: str):
    r = simmer_request("/api/sdk/positions", api_key=api_key, timeout=45)
    if isinstance(r, dict) and "positions" in r: return r["positions"]
    if isinstance(r, list): return r
    return []

# --- THE MISSING FUNCTION RESTORED ---
def find_simmer_market_broad(api_key: str, slug: str):
    r = simmer_request(f"/api/sdk/markets?limit=100&search={slug}", api_key=api_key)
    if isinstance(r, dict) and "markets" in r:
        for m in r["markets"]:
            if slug in str(m.get("slug", "")) or slug in str(m.get("polymarket_url", "")):
                return m.get("id")
    return None

def import_market(api_key: str, slug: str):
    existing_id = find_simmer_market_broad(api_key, slug)
    if existing_id: return existing_id, None, True

    url = f"https://polymarket.com/event/{slug}"
    for attempt in range(3):
        r = simmer_request("/api/sdk/markets/import", method="POST", data={"polymarket_url": url, "shared": True}, api_key=api_key, timeout=90)
        if isinstance(r, dict) and r.get("status") in ["imported", "already_exists"]:
            return r.get("market_id"), None, True
        err = r.get("error") if isinstance(r, dict) else str(r)
        if "internal server error" in str(err).lower():
            time.sleep(2)
            fallback_id = find_simmer_market_broad(api_key, slug)
            if fallback_id: return fallback_id, None, True
        time.sleep(2)
    return None, "import_failed_after_retries", False
# -------------------------------------

def execute_trade(api_key: str, market_id: str, side: str, amount: float = None, shares: float = None, action: str = "buy"):
    payload = {
        "market_id": market_id,
        "side": side,
        "venue": "polymarket",
        "source": TRADE_SOURCE,
        "action": action,
    }
    if action == "sell": payload["shares"] = float(shares or 0)
    else: payload["amount"] = float(amount or 0)

    for attempt in range(3):
        res = simmer_request("/api/sdk/trade", method="POST", data=payload, api_key=api_key, timeout=120)
        if res and res.get("success"): return res
        err_msg = str(res.get("error") if isinstance(res, dict) else res).lower()
        if "market not found" in err_msg or "timeout" in err_msg:
            time.sleep(2)
            continue
        return res
    return {"error": "max_retries_exceeded"}

# -----------------------
# NEW: AGGRESSIVE RETRY CLOSE (No Wallet Checks)
# -----------------------
def retry_close_until_success(api_key: str, market_id: str, side: str, shares: float):
    """
    Continually sends the sell order until the API returns success.
    """
    print(f"EXIT TRIGGERED: Selling {shares} shares of {side.upper()}...")
    
    attempt = 1
    while True:
        res = execute_trade(api_key, market_id, side, shares=shares, action="sell")
        
        if res.get("success"):
            print(f"SELL SUCCESS (Attempt {attempt}): Order accepted.")
            return True
        else:
            print(f"SELL FAILED (Attempt {attempt}): {res.get('error')}. Retrying in 1s...")
            time.sleep(1.0)
            attempt += 1
            if attempt > 10:
                print("WARNING: 10 failed sell attempts. Still trying...")

def calc_position_size(api_key: str, max_size: float, smart_pct: float, smart_sizing: bool):
    if not smart_sizing: return max_size
    pf = get_portfolio(api_key)
    if not isinstance(pf, dict) or pf.get("error"): return max_size
    bal = float(pf.get("balance_usdc", 0) or 0)
    if bal <= 0: return max_size
    return min(max_size, bal * smart_pct)

# -----------------------
# Logic: Close & Monitor
# -----------------------
def get_market_end_time_from_slug(slug: str):
    try:
        parts = slug.split("-")
        ts = int(parts[-1])
        return datetime.fromtimestamp(ts + 300, tz=timezone.utc)
    except:
        return None

def get_market_price(market_id: str):
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    res = api_request(url, timeout=5)
    if isinstance(res, dict) and "outcomePrices" in res:
        try:
            return json.loads(res["outcomePrices"])
        except:
            pass
    return None

def monitor_and_close(api_key: str, market_id: str, end_time: datetime, side: str):
    target_close_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    
    print("MONITOR: Fetching initial position to determine shares...")
    time.sleep(2) # Wait for trade to settle
    
    # We only check wallet ONCE at start to know how many shares we have to sell later
    positions = get_positions(api_key)
    my_pos = next((p for p in positions if str(p.get("market_id") or p.get("id")) == str(market_id)), None)
    
    shares_owned = 0.0
    entry_price = 0.5

    if my_pos:
        shares_owned = float(my_pos.get(f"shares_{side}", 0) or 0)
        entry_price = float(my_pos.get("avg_buy_price", 0) or my_pos.get("avg_price", 0) or 0)
    
    if shares_owned <= 0:
        print(f"MONITOR ERROR: Wallet says 0 shares of {side}. Assuming failed entry.")
        return

    if entry_price <= 0: entry_price = 0.5 # Fallback

    print(f"MONITOR: Tracking {shares_owned:.4f} shares of {side.upper()}. Entry: {entry_price:.3f}. Buffer: {CLOSE_BUFFER_SECONDS}s")
    
    # --- PnL MONITOR LOOP ---
    while True:
        now = now_utc()
        if now >= target_close_time:
            print(f"MONITOR: Time limit reached ({CLOSE_BUFFER_SECONDS}s buffer). Closing.")
            break
            
        prices = get_market_price(market_id)
        if prices and entry_price > 0:
            idx = 0 if side == "yes" else 1
            curr_price = float(prices[idx])
            
            pnl = (curr_price - entry_price) / entry_price
            
            print(f"MONITOR: {side.upper()} @ {curr_price:.3f} | PnL: {pnl*100:+.1f}% | Time left: {(target_close_time - now).total_seconds():.0f}s")
            
            if pnl <= -STOP_LOSS_PCT:
                print(f"!!! STOP LOSS TRIGGERED ({pnl*100:.1f}%) !!! Closing...")
                break
            if pnl >= TAKE_PROFIT_PCT:
                print(f"!!! TAKE PROFIT TRIGGERED ({pnl*100:.1f}%) !!! Closing...")
                break
        
        time.sleep(3)

    # --- AGGRESSIVE CLOSE ---
    retry_close_until_success(api_key, market_id, side, shares_owned)

def already_in_this_market(positions, slug: str, market_id: str = None):
    target_slug = (slug or "").strip().lower()
    for p in positions or []:
        # Check by ID
        if market_id:
            pid = str(p.get("market_id") or p.get("id") or "")
            if pid == str(market_id):
                if float(p.get("shares_yes", 0)) > 0 or float(p.get("shares_no", 0)) > 0:
                    return True
        # Check by Slug
        p_slug = str(p.get("slug") or "").lower()
        p_url = str(p.get("polymarket_url") or "").lower()
        if target_slug and (target_slug in p_slug or target_slug in p_url):
            if float(p.get("shares_yes", 0)) > 0 or float(p.get("shares_no", 0)) > 0:
                return True
    return False

def run_once(cfg, live: bool, quiet: bool, smart_sizing: bool):
    def log(msg, force=False):
        if not quiet or force: print(msg)

    api_key = get_api_key()
    st = load_state()

    # 1. Market Discovery
    markets = discover_fast_markets(cfg["asset"], cfg["window"])
    best = select_best_market(markets, int(cfg["min_time_remaining"]))
    
    if not best:
        log("SKIP: Time window too close to end.", force=True)
        return

    log(f"TARGET: {best['slug']} (End: {best['end_time'].strftime('%H:%M')})", force=True)
    
    # 2. Duplicate Check
    positions = get_positions(api_key)
    if already_in_this_market(positions, best["slug"]):
        log("SKIP: Already in this market.", force=True)
        return

    # 3. Signal
    sig = get_coingecko_momentum(cfg["asset"], int(cfg["lookback_minutes"]))
    if not sig or sig.get("error"):
        log(f"SKIP: Signal error -> {sig}", force=True)
        return

    mom_pct = sig["momentum_pct"]
    if abs(mom_pct) < float(cfg["min_momentum_pct"]):
        log(f"SKIP: Weak momentum {mom_pct:.3f}%", force=True)
        return

    side = "yes" if sig["direction"] == "up" else "no"
    
    # 3b. Price Check
    buy_price = 0.5 
    try:
        prices = json.loads(best.get("outcome_prices", "[]"))
        if prices and len(prices) >= 2:
            buy_price = float(prices[0]) if side == "yes" else float(prices[1])
    except: pass

    amount = calc_position_size(api_key, float(cfg["max_position"]), float(cfg["smart_sizing_pct"]), smart_sizing)
    
    min_shares = 5.0
    estimated_shares = amount / buy_price if buy_price > 0 else 0
    if estimated_shares < min_shares:
        required_amount = (min_shares * buy_price) * 1.05
        amount = required_amount

    if amount < 1.0:
        log("SKIP: Amount too small.", force=True)
        return

    log(f"SIGNAL: {side.upper()} | Mom={mom_pct:.3f}% | Price={buy_price:.2f}", force=True)

    # 4. Execution
    market_id, err, imported = import_market(api_key, best["slug"])
    if not market_id:
        log(f"FAIL: Import {err}", force=True)
        return

    if not live:
        log(f"DRY RUN: Buy {side} ${amount}", force=True)
        return

    res = execute_trade(api_key, market_id, side, amount=amount)
    if res and res.get("success"):
        st["trades"] = int(st.get("trades", 0)) + 1
        st["last_trade_ts"] = now_utc().isoformat()
        save_state(st)
        log("TRADE: Success. Entering Monitor Mode...", force=True)
        
        # 5. SIMPLE MONITOR (No crazy wallet checks)
        monitor_and_close(api_key, market_id, best["end_time"], side)
        
        log("CYCLE COMPLETE: Trade closed.", force=True)
        
    else:
        log(f"TRADE: Failed -> {res}", force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Execute real trades (default dry-run)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output signal/errors")
    parser.add_argument("--smart-sizing", action="store_true", help="Use portfolio based sizing")
    args = parser.parse_args()

    cfg = load_config()
    run_once(cfg, live=args.live, quiet=args.quiet, smart_sizing=args.smart_sizing)
