#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (v5.5 - PARANOID CLOSER).

CHANGELOG:
- ✅ PARANOID CLOSE: Never assumes a trade is closed. Checks wallet repeatedly.
  -> If shares remain > 0, it retries the sell loop infinitely.
- ✅ HARD PROFIT: Instantly closes if PnL > 80% (Secure the bag).
- ✅ STRATEGY: CoinGecko Momentum (Simple Trend) + High Frequency Monitor.
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
TAKE_PROFIT_PCT = 0.15      # +15% Profit (Secure the bag early)

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
        req_headers.setdefault("User-Agent", "railway-fastloop/5.5")
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
# Signal: COINGECKO (Stable)
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
def get_positions(api_key: str):
    r = simmer_request("/api/sdk/positions", api_key=api_key, timeout=45)
    if isinstance(r, dict) and "positions" in r: return r["positions"]
    if isinstance(r, list): return r
    return []

def get_portfolio(api_key: str):
    return simmer_request("/api/sdk/portfolio", api_key=api_key, timeout=45)

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

# --- PARANOID CLOSER LOGIC ---
def paranoid_close_position(api_key: str, market_id: str):
    """
    Continually checks wallet. If shares > 0, SELLS.
    Does NOT stop until 0 shares remain.
    """
    print("PARANOID CLOSE: Engaging...")
    
    while True:
        # 1. Check current position size
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id") or p.get("id")) == str(market_id)), None)
        
        if not my_pos:
            print("PARANOID: Position disappeared. Confirmed Closed.")
            return True
            
        s_yes = float(my_pos.get("shares_yes", 0) or 0)
        s_no = float(my_pos.get("shares_no", 0) or 0)
        
        # 2. If empty, we are done
        if s_yes <= 0 and s_no <= 0:
            print("PARANOID: Zero shares found. Confirmed Closed.")
            return True
            
        # 3. If shares exist, SELL THEM
        if s_yes > 0:
            print(f"PARANOID: Found {s_yes} YES shares. Selling...")
            execute_trade(api_key, market_id, "yes", shares=s_yes, action="sell")
        if s_no > 0:
            print(f"PARANOID: Found {s_no} NO shares. Selling...")
            execute_trade(api_key, market_id, "no", shares=s_no, action="sell")
            
        # 4. Wait & Retry (Do not trust success message, trust the wallet check on next loop)
        time.sleep(1.5)

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
    """
    Blocks execution until 30 seconds before end_time, CHECKING PnL every 1s.
    """
    target_close_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    
    print("MONITOR: Fetching initial position details...")
    time.sleep(2) # Wait for trade to settle
    
    positions = get_positions(api_key)
    my_pos = next((p for p in positions if str(p.get("market_id") or p.get("id")) == str(market_id)), None)
    
    if not my_pos:
        print("MONITOR: Position not found (maybe delay?). Will try to track anyway.")
        entry_price = 0.5 # fallback
    else:
        # Try to find avg price
        entry_price = float(my_pos.get("avg_buy_price", 0) or my_pos.get("avg_price", 0) or 0)
        if entry_price <= 0:
            # Fallback to current market price
            prices = get_market_price(market_id)
            if prices:
                idx = 0 if side == "yes" else 1
                entry_price = float(prices[idx])
            else:
                entry_price = 0.5 # Absolute fallback

    print(f"MONITOR: Tracking {side.upper()}. Entry Est: {entry_price:.3f}. Holding until {target_close_time.strftime('%H:%M:%S')}")
    
    while True:
        now = now_utc()
        if now >= target_close_time:
            print("MONITOR: Time limit reached (30s buffer).")
            break
            
        # Check Price & PnL
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
        
        time.sleep(1.0) # High frequency check

    print("MONITOR: Time up or Level Hit. Engaging Paranoid Close...")
    paranoid_close_position(api_key, market_id)

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
            paranoid_close_position(api_key, market_id)

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

    log(f"TARGET: {best['slug']} (End: {best['end_
