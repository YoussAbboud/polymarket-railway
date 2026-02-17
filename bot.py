#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (v6.5 - Three-Strike Verification).

FIXES:
- ✅ THREE-STRIKE RULE: Requires 3 consecutive "0 share" checks to confirm close.
  -> Prevents the "API Lag" where shares momentarily disappear and reappear.
- ✅ STRICT MONITOR: Only tracks the side you bought.
- ✅ DEBUG LOGS: Prints detailed selling actions.
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

def append_journal(event: dict):
    event = dict(event)
    event["ts"] = now_utc().isoformat()
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def api_request(url, method="GET", data=None, headers=None, timeout=25):
    try:
        req_headers = headers or {}
        req_headers.setdefault("User-Agent", "railway-fastloop/6.5")
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
# Position Check Helper
# -----------------------
def get_positions(api_key: str):
    r = simmer_request("/api/sdk/positions", api_key=api_key, timeout=45)
    if isinstance(r, dict) and "positions" in r: return r["positions"]
    if isinstance(r, list): return r
    return []

def already_in_this_market(positions, slug: str, market_id: str = None):
    target_slug = (slug or "").strip().lower()

    for p in positions or []:
        # Check by ID
        if market_id:
            pid = str(p.get("market_id") or p.get("id") or "")
            if pid == str(market_id):
                if float(p.get("shares_yes", 0)) > 0 or float(p.get("shares_no", 0)) > 0:
                    return True
        
        # Check by Slug/URL
        p_slug = str(p.get("slug") or "").lower()
        p_url = str(p.get("polymarket_url") or "").lower()
        
        if target_slug and (target_slug in p_slug or target_slug in p_url):
            if float(p.get("shares_yes", 0)) > 0 or float(p.get("shares_no", 0)) > 0:
                return True
                
    return False

# -----------------------
# Config
# -----------------------
CONFIG_DEFAULTS = {
    "asset": "BTC",
    "window": "5m",
    "signal_source": "coingecko",
    "lookback_minutes": 12,
    "entry_threshold": 0.05,
    "min_momentum_pct": 0.08, 
    "min_time_remaining": 45,
    "min_volume_ratio": 0.15,
    "max_position": 5.0, 
    "smart_sizing_pct": 0.05,
    "max_open_fast_positions": 2,
    "cooldown_seconds": 80,
    "daily_trade_limit": 20,
    "daily_import_limit": 20,
    "fee_edge_buffer": 0.02,
}

ENV_MAP = {
    "asset": "SIMMER_SPRINT_ASSET",
    "window": "SIMMER_SPRINT_WINDOW",
    "signal_source": "SIMMER_SPRINT_SIGNAL",
    "lookback_minutes": "SIMMER_SPRINT_LOOKBACK",
    "entry_threshold": "SIMMER_SPRINT_ENTRY",
    "min_momentum_pct": "SIMMER_SPRINT_MOMENTUM",
    "min_time_remaining": "SIMMER_SPRINT_MIN_TIME",
    "max_position": "SIMMER_SPRINT_MAX_POSITION",
}

def load_config():
    cfg = dict(CONFIG_DEFAULTS)
    file_cfg = load_json("config.json", {})
    if isinstance(file_cfg, dict): cfg.update(file_cfg)

    for k, env in ENV_MAP.items():
        v = os.environ.get(env)
        if v is None: continue
        if isinstance(cfg.get(k), bool):
            cfg[k] = v.lower() in ("true", "1", "yes")
        elif isinstance(cfg.get(k), int):
            cfg[k] = int(v)
        elif isinstance(cfg.get(k), float):
            cfg[k] = float(v)
        else:
            cfg[k] = v

    cfg["asset"] = str(cfg["asset"]).upper()
    return cfg

# -----------------------
# Market Discovery
# -----------------------
def get_current_window_slug(asset: str):
    now = now_utc()
    minute = (now.minute // 5) * 5
    start_dt = now.replace(minute=minute, second=0, microsecond=0)
    ts = int(start_dt.timestamp())
    
    asset_prefix = asset.lower()
    if asset_prefix == "bitcoin": asset_prefix = "btc"
    
    slug = f"{asset_prefix}-updown-5m-{ts}"
    end_dt = start_dt + timedelta(minutes=5)
    
    return {
        "slug": slug,
        "start_time": start_dt,
        "end_time": end_dt,
        "question": f"{asset} Up or Down {start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}",
        "ts_id": ts
    }

def discover_fast_markets(asset: str, window: str):
    target = get_current_window_slug(asset)
    return [target]

def select_best_market(markets, min_seconds_left: int):
    now = now_utc()
    if not markets: return None
    m = markets[0]
    end = m.get("end_time")
    left = (end - now).total_seconds()
    if left < min_seconds_left:
        return None
    return m

def market_window_key(slug: str) -> str:
    return slug

# -----------------------
# Signal
# -----------------------
def get_coingecko_momentum(asset: str, lookback_minutes: int):
    coin_id = COINGECKO_IDS.get(asset, "bitcoin")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
    
    data = api_request(url, timeout=30)
    if isinstance(data, dict) and data.get("error"):
        return {"error": f"coingecko_error: {data.get('error')}"}
    if not isinstance(data, dict) or "prices" not in data:
        return {"error": "coingecko_invalid_format", "raw": str(data)[:100]}

    prices = data["prices"]
    if not prices or len(prices) < 2:
        return {"error": "coingecko_insufficient_data"}

    latest_ts, latest_price = prices[-1]
    target_ts = latest_ts - (lookback_minutes * 60 * 1000)
    
    past_price = None
    best_diff = float('inf')
    
    for t, p in reversed(prices):
        diff = abs(t - target_ts)
        if diff < best_diff:
            best_diff = diff
            past_price = p
        else:
            pass
            
    if past_price is None:
        return {"error": "coingecko_history_lookup_failed"}
        
    momentum_pct = ((latest_price - past_price) / past_price) * 100.0
    direction = "up" if momentum_pct > 0 else "down"

    return {
        "price_now": latest_price,
        "price_then": past_price,
        "momentum_pct": momentum_pct,
        "direction": direction,
        "volume_ratio": 1.0,
        "source": "coingecko"
    }

# -----------------------
# Simmer actions
# -----------------------
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
    
    # DEBUG PRINT: Verify exact intent
    if action == "buy":
        print(f"DEBUG: SENDING BUY ORDER -> Side: {side.upper()} | Amount: ${amount:.2f}")

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
# IRON-CLAD CLOSE (3-STRIKE VERIFICATION)
# -----------------------
def iron_clad_close(api_key: str, market_id: str):
    """
    Exits the trade and VERIFIES the wallet is 0. 
    Requires 3 CONSECUTIVE '0 share' confirmations to beat API lag.
    """
    print(f"IRON-CLAD CLOSE: Ensuring exit for market {market_id}...")
    consecutive_zeros = 0
    
    while True:
        r = simmer_request("/api/sdk/positions", api_key=api_key)
        positions = r.get("positions", []) if isinstance(r, dict) else []
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        
        s_yes = float(my_pos.get("shares_yes", 0)) if my_pos else 0
        s_no = float(my_pos.get("shares_no", 0)) if my_pos else 0

        # THREE-STRIKE RULE
        if s_yes <= 0.001 and s_no <= 0.001:
            consecutive_zeros += 1
            print(f"VERIFIED: Zero shares found. Confirmation {consecutive_zeros}/3...")
            if consecutive_zeros >= 3:
                print("VERIFIED: Position confirmed gone x3. Safe to exit.")
                return True
        else:
            # If we see shares, reset the counter to 0
            consecutive_zeros = 0 
            
            if s_yes > 0:
                print(f"ACTION: Selling {s_yes} YES shares...")
                execute_trade(api_key, market_id, "yes", shares=s_yes, action="sell")
            if s_no > 0:
                print(f"ACTION: Selling {s_no} NO shares...")
                execute_trade(api_key, market_id, "no", shares=s_no, action="sell")
        
        time.sleep(2.0) # Buffer between checks

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

def monitor_and_close(api_key: str, market_id: str, end_time: datetime, requested_side: str):
    """
    Blocks execution until buffer time.
    STRICT MODE: Only tracks the requested side.
    """
    target_close_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    side = requested_side # STRICT: We trust what we just bought.
    
    print("MONITOR: Fetching initial position details...")
    time.sleep(2) # Wait for trade to settle
    
    positions = get_positions(api_key)
    my_pos = next((p for p in positions if str(p.get("market_id") or p.get("id")) == str(market_id)), None)
    
    if not my_pos:
        print(f"MONITOR: Position not found yet. Waiting for {side.upper()} to appear...")
        entry_price = 0.5 
    else:
        # Check if we have the shares we expect
        s_expected = float(my_pos.get(f"shares_{side}", 0) or 0)
        if s_expected <= 0:
            print(f"MONITOR: WARNING! We bought {side.upper()} but wallet has 0. (Maybe lag or filled other side?)")
        
        entry_price = float(my_pos.get("avg_buy_price", 0) or my_pos.get("avg_price", 0) or 0)
        if entry_price <= 0:
            prices = get_market_price(market_id)
            if prices:
                idx = 0 if side == "yes" else 1
                entry_price = float(prices[idx])
            else:
                entry_price = 0.5

    print(f"MONITOR: Tracking {side.upper()}. Entry Est: {entry_price:.3f}. Holding until {target_close_time.strftime('%H:%M:%S')}")
    
    while True:
        now = now_utc()
        if now >= target_close_time:
            print(f"MONITOR: Time limit reached ({CLOSE_BUFFER_SECONDS}s buffer).")
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

    print("MONITOR: Closing position...")
    iron_clad_close(api_key, market_id)

def check_safety_close(api_key: str, positions):
    now = now_utc()
    for p in positions or []:
        slug = p.get("slug")
        if not slug: continue
        
        end_time = get_market_end_time_from_slug(slug)
        if not end_time: continue
        
        seconds_left = (end_time - now).total_seconds()
        
        if seconds_left < CLOSE_BUFFER_SECONDS:
            print(f"SAFETY: Found position {slug} near expiry ({seconds_left:.0f}s left). Closing now.")
            market_id = p.get("market_id") or p.get("id")
            iron_clad_close(api_key, market_id)

def run_once(cfg, live: bool, quiet: bool, smart_sizing: bool):
    def log(msg, force=False):
        if not quiet or force: print(msg)

    api_key = get_api_key()
    st = load_state()

    # 1. Safety Check
    positions = get_positions(api_key)
    check_safety_close(api_key, positions)

    # 2. Market Discovery
    markets = discover_fast_markets(cfg["asset"], cfg["window"])
    best = select_best_market(markets, int(cfg["min_time_remaining"]))
    
    if not best:
        log("SKIP: Time window too close to end.", force=True)
        return

    log(f"TARGET: {best['slug']} (End: {best['end_time'].strftime('%H:%M')})", force=True)

    window_key = market_window_key(best["slug"])
    if has_recent_lock(window_key):
        log("SKIP: Recent lock.", force=False)
        return
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
    
    # 3b. Price Check for Min Shares
    buy_price = 0.5 
    try:
        prices = json.loads(best.get("outcome_prices", "[]"))
        if prices and len(prices) >= 2:
            buy_price = float(prices[0]) if side == "yes" else float(prices[1])
    except: pass

    amount = calc_position_size(api_key, float(cfg["max_position"]), float(cfg["smart_sizing_pct"]), smart_sizing)
    
    # Enforce Min Shares (5.0)
    min_shares = 5.0
    estimated_shares = amount / buy_price if buy_price > 0 else 0
    
    if estimated_shares < min_shares:
        required_amount = (min_shares * buy_price) * 1.05
        if required_amount > float(cfg["max_position"]) * 3:
            log(f"SKIP: Required amount ${required_amount:.2f} too high.", force=True)
            return
        log(f"ADJUST: Bumping amount to ${required_amount:.2f} to meet 5 share min.")
        amount = required_amount

    if amount < 1.0:
        log("SKIP: Amount too small.", force=True)
        return

    log(f"SIGNAL: {side.upper()} | Mom={mom_pct:.3f}% | Price={buy_price:.2f}", force=True)
    write_lock(window_key, {"status": "pending"})

    # 4. Execution
    market_id, err, imported = import_market(api_key, best["slug"])
    if not market_id:
        log(f"FAIL: Import {err}", force=True)
        clear_lock(window_key)
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
        
        write_lock(window_key, {"status": "active", "slug": best["slug"]})
        
        # 5. BLOCKING MONITOR (with SL/TP)
        monitor_and_close(api_key, market_id, best["end_time"], side)
        
        log("CYCLE COMPLETE: Trade closed.", force=True)
        write_lock(window_key, {"status": "closed", "slug": best["slug"]})
        
    else:
        log(f"TRADE: Failed -> {res}", force=True)
        clear_lock(window_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Execute real trades (default dry-run)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output signal/errors")
    parser.add_argument("--smart-sizing", action="store_true", help="Use portfolio based sizing")
    args = parser.parse_args()

    cfg = load_config()
    run_once(cfg, live=args.live, quiet=args.quiet, smart_sizing=args.smart_sizing)
