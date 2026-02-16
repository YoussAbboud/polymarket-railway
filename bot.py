#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (v2.2 - Import Fix).

CHANGELOG:
- ✅ FIX: "Internal Server Error" on Import.
  -> Logic: If import fails, we now SEARCH Simmer for the market.
     Often the market exists but the import tool crashed.
- ✅ RETRY: Added 3x retries to the import_market function.
- ✅ ROBUST: Keeps all previous CoinGecko + Execution fixes.
"""

import os, sys, json, argparse, time
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
GAMMA_MARKETS_URL = (
    "https://gamma-api.polymarket.com/markets"
    "?limit=200&closed=false&tag=crypto&order=createdAt&ascending=false"
)

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "MATIC": "matic-network",
    "DOGE": "dogecoin"
}

ASSET_PATTERNS = {
    "BTC": ["bitcoin up or down"],
    "ETH": ["ethereum up or down"],
    "SOL": ["solana up or down"],
}

TRADE_SOURCE = "railway:fastloop"
LOCK_TTL_SECONDS = int(os.environ.get("LOCK_TTL_SECONDS", "300"))

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
        req_headers.setdefault("User-Agent", "railway-fastloop/2.2")
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
# Market discovery
# -----------------------
def parse_fast_market_window_utc(question: str):
    import re
    try: from zoneinfo import ZoneInfo
    except ImportError: pass

    q = (question or "").strip()
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{1,2}:\d{2}(?:AM|PM))-(\d{1,2}:\d{2}(?:AM|PM))\s*ET", q)
    if not m: return None

    month_str = m.group(1)[:3]
    day = int(m.group(2))
    t1 = m.group(3)
    t2 = m.group(4)

    try: ny = ZoneInfo("America/New_York")
    except Exception: ny = timezone(timedelta(hours=-5))

    year_now = now_utc().year
    candidates = [year_now, year_now - 1, year_now + 1]

    def try_parse(month_fmt: str, y: int):
        try:
            s_str = f"{month_str} {day} {y} {t1}"
            e_str = f"{month_str} {day} {y} {t2}"
            start_local = datetime.strptime(s_str, month_fmt).replace(tzinfo=ny)
            end_local   = datetime.strptime(e_str, month_fmt).replace(tzinfo=ny)
            return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)
        except Exception: return None

    best = None
    best_delta = float('inf')
    now = now_utc()
    for y in candidates:
        win = try_parse("%b %d %Y %I:%M%p", y)
        if not win: continue
        start_utc, _ = win
        delta = abs((start_utc - now).total_seconds())
        if delta < best_delta:
            best = win
            best_delta = delta

    return best

def parse_iso_dt(s):
    if not s: return None
    try:
        s = str(s).strip()
        if s.endswith("Z"): s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception: return None

def gamma_market_window(m: dict):
    start = parse_iso_dt(m.get("startDateIso")) or parse_iso_dt(m.get("start_date_iso"))
    end   = parse_iso_dt(m.get("endDateIso"))   or parse_iso_dt(m.get("end_date_iso"))
    if not start and m.get("startDate"):
        try:
            ts = float(m["startDate"])
            if ts > 1e12: ts /= 1000.0
            start = datetime.fromtimestamp(ts, tz=timezone.utc)
        except: pass
    if not end and m.get("endDate"):
        try:
            ts = float(m["endDate"])
            if ts > 1e12: ts /= 1000.0
            end = datetime.fromtimestamp(ts, tz=timezone.utc)
        except: pass
    return start, end

def is_live_window(start: datetime, end: datetime) -> bool:
    if not start or not end: return False
    now = now_utc()
    return start <= now <= end

def discover_fast_markets(asset: str, window: str):
    patterns = ASSET_PATTERNS.get(asset, ASSET_PATTERNS["BTC"])
    result = api_request(GAMMA_MARKETS_URL)
    if not isinstance(result, list): return []

    out = []
    for m in result:
        q_raw = (m.get("question") or "")
        q = q_raw.lower()
        slug = (m.get("slug") or "")

        # Safety: slug must exist
        if not slug: continue

        if not any(p in q for p in patterns): continue
        if m.get("closed"): continue

        start, end = gamma_market_window(m)
        if not is_live_window(start, end): continue

        out.append({
            "question": q_raw,
            "slug": slug,
            "start_time": start,
            "end_time": end,
            "outcome_prices": m.get("outcomePrices", "[]"),
            "fee_rate_bps": int(m.get("fee_rate_bps") or m.get("feeRateBps") or 0),
        })
    return out

def select_best_market(markets, min_seconds_left: int):
    now = now_utc()
    live = []
    for m in markets:
        end = m.get("end_time")
        if not end: continue
        left = (end - now).total_seconds()
        if left < min_seconds_left: continue
        live.append((left, m))

    if not live: return None
    live.sort(key=lambda x: x[0]) 
    return live[0][1]

def market_window_key(question: str) -> str:
    win = parse_fast_market_window_utc(question or "")
    if not win: return (question or "").strip()
    s, e = win
    return f"{(question or '').split('-')[0].strip()}|{s.isoformat()}|{e.isoformat()}"

# -----------------------
# Signal: CoinGecko
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

def find_simmer_market_by_slug(api_key: str, slug: str):
    """Fallback: search for market if import fails"""
    # Try searching by slug
    r = simmer_request(f"/api/sdk/markets?limit=100&search={slug}", api_key=api_key)
    if isinstance(r, dict) and "markets" in r:
        for m in r["markets"]:
            # Check if source url or slug matches
            if slug in str(m.get("polymarket_url", "")) or slug in str(m.get("slug", "")):
                return m.get("id")
    return None

def import_market(api_key: str, slug: str):
    url = f"https://polymarket.com/event/{slug}"
    
    # Retry loop for import
    for attempt in range(3):
        r = simmer_request(
            "/api/sdk/markets/import",
            method="POST",
            data={"polymarket_url": url, "shared": True},
            api_key=api_key,
            timeout=90
        )
        
        # Success case
        if isinstance(r, dict) and r.get("status") in ["imported", "already_exists"]:
            return r.get("market_id"), None, True
            
        # Error handling
        err = r.get("error") if isinstance(r, dict) else str(r)
        
        # If Internal Server Error, try Fallback Search immediately
        if "internal server error" in str(err).lower():
            # Fallback search
            fallback_id = find_simmer_market_by_slug(api_key, slug)
            if fallback_id:
                return fallback_id, None, True
            # If not found, sleep and retry import
            time.sleep(2)
            continue
            
        # Other errors (rate limit etc)
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

def calc_position_size(api_key: str, max_size: float, smart_pct: float, smart_sizing: bool):
    if not smart_sizing: return max_size
    pf = get_portfolio(api_key)
    if not isinstance(pf, dict) or pf.get("error"): return max_size
    bal = float(pf.get("balance_usdc", 0) or 0)
    if bal <= 0: return max_size
    return min(max_size, bal * smart_pct)

# -----------------------
# Main cycle
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

def classify_position_by_question(p: dict):
    q = (p.get("question") or "").strip()
    win = parse_fast_market_window_utc(q)
    if not win: return "active", None, None
    start_utc, end_utc = win
    now = now_utc()
    if end_utc < (now - timedelta(seconds=20)): return "expired", start_utc, end_utc
    if start_utc > (now + timedelta(hours=1)): return "future", start_utc, end_utc
    return "active", start_utc, end_utc

def count_open_fast_positions(positions):
    n = 0
    for p in positions or []:
        ql = (p.get("question") or "").lower()
        if "up or down" not in ql: continue
        if float(p.get("shares_yes", 0)) <= 0 and float(p.get("shares_no", 0)) <= 0: continue
        status, _, _ = classify_position_by_question(p)
        if status in ("active", "future"): n += 1
    return n

def already_in_this_market(positions, question: str, market_id: str = None):
    target_q = (question or "").strip()
    target_key = market_window_key(target_q)
    for p in positions or []:
        pq = (p.get("question") or "").strip()
        pid = str(p.get("market_id") or p.get("id") or "")
        if market_id and pid and str(market_id) == pid:
            if float(p.get("shares_yes", 0)) > 0 or float(p.get("shares_no", 0)) > 0: return True
        if pq == target_q or market_window_key(pq) == target_key:
            status, _, _ = classify_position_by_question(p)
            if status == "expired": continue
            if float(p.get("shares_yes", 0)) > 0 or float(p.get("shares_no", 0)) > 0: return True
    return False

def close_positions(api_key: str, positions, quiet: bool = False):
    closed_any = False
    for p in positions or []:
        q = (p.get("question") or "")
        if "up or down" not in q.lower(): continue
        status, _, end_utc = classify_position_by_question(p)
        if status != "expired": continue

        market_id = p.get("market_id") or p.get("id")
        shares_yes = float(p.get("shares_yes", 0) or 0)
        shares_no  = float(p.get("shares_no", 0) or 0)

        if not quiet: print(f"AUTO-CLOSE: EXPIRED | {q} | end={end_utc}")
        if shares_yes > 0:
            execute_trade(api_key, market_id, "yes", shares=shares_yes, action="sell")
            closed_any = True
        if shares_no > 0:
            execute_trade(api_key, market_id, "no", shares=shares_no, action="sell")
            closed_any = True
    return closed_any

def run_once(cfg, live: bool, quiet: bool, smart_sizing: bool):
    def log(msg, force=False):
        if not quiet or force: print(msg)

    api_key = get_api_key()
    st = load_state()

    positions = get_positions(api_key)
    closed_any = close_positions(api_key, positions, quiet=quiet)
    if closed_any: positions = get_positions(api_key) 

    open_fast = count_open_fast_positions(positions)
    if open_fast >= int(cfg["max_open_fast_positions"]):
        log(f"SKIP: max positions reached ({open_fast})", force=True)
        return

    if st.get("last_trade_ts"):
        since = (now_utc() - parse_iso_dt(st["last_trade_ts"])).total_seconds()
        if since < int(cfg["cooldown_seconds"]):
            log(f"SKIP: cooldown {since:.0f}s", force=False)
            return

    markets = discover_fast_markets(cfg["asset"], cfg["window"])
    if not markets:
        log("SKIP: no live markets strictly fitting schedule")
        return

    best = select_best_market(markets, int(cfg["min_time_remaining"]))
    if not best:
        log("SKIP: no strictly suitable market or not enough time left", force=True)
        return

    window_key = market_window_key(best["question"])
    if has_recent_lock(window_key):
        log("SKIP: recent lock", force=False)
        return
    if already_in_this_market(positions, best["question"]):
        log("SKIP: already in market", force=True)
        return

    sig = get_coingecko_momentum(cfg["asset"], int(cfg["lookback_minutes"]))
    if not sig or sig.get("error"):
        log(f"SKIP: signal error -> {sig}", force=True)
        return

    mom_pct = sig["momentum_pct"]
    if abs(mom_pct) < float(cfg["min_momentum_pct"]):
        log(f"SKIP: weak momentum {mom_pct:.3f}%", force=True)
        return

    yes_price = 0.5
    try:
        prices = json.loads(best.get("outcome_prices", "[]"))
        if prices: yes_price = float(prices[0])
    except: pass
    
    if sig["direction"] == "up":
        side = "yes"
        buy_price = yes_price
    else:
        side = "no"
        buy_price = 1.0 - yes_price

    amount = calc_position_size(api_key, float(cfg["max_position"]), float(cfg["smart_sizing_pct"]), smart_sizing)
    if amount < 1.0:
        log(f"SKIP: amount {amount} too small (<$1)", force=True)
        return

    log(f"SIGNAL: {side.upper()} | Mom={mom_pct:.3f}% | Price={buy_price:.2f}", force=True)

    write_lock(window_key, {"status": "pending"})

    market_id, err, imported = import_market(api_key, best["slug"])
    if not market_id:
        log(f"FAIL: import {err}", force=True)
        clear_lock(window_key)
        return

    if already_in_this_market(get_positions(api_key), best["question"], market_id):
        log("SKIP: position exists right before trade", force=True)
        return

    if not live:
        log(f"DRY RUN: Buy {side} ${amount}", force=True)
        return

    res = execute_trade(api_key, market_id, side, amount=amount)
    if res and res.get("success"):
        st["trades"] = int(st.get("trades", 0)) + 1
        st["last_trade_ts"] = now_utc().isoformat()
        save_state(st)
        log("TRADE: Success", force=True)
        append_journal({"type": "trade_success", "market_id": market_id, "question": best["question"], "window_key": window_key})
        write_lock(window_key, {"status": "done", "slug": best["slug"], "market_id": str(market_id)})
    else:
        err2 = res.get("error") if isinstance(res, dict) else "unknown"
        full_debug = res if isinstance(res, dict) else str(res)
        log(f"TRADE: failed -> {full_debug}", force=True)
        
        if "timed out" in str(err2).lower():
            time.sleep(2)
            positions2 = get_positions(api_key)
            if already_in_this_market(positions2, best["question"], market_id=market_id):
                log("TRADE: success (verified after timeout)", force=True)
                append_journal({"type": "trade_verified", "market_id": market_id, "question": best["question"], "window_key": window_key})
                write_lock(window_key, {"status": "done_verified", "slug": best["slug"], "market_id": str(market_id)})
                st["trades"] = int(st.get("trades", 0)) + 1
                st["last_trade_ts"] = now_utc().isoformat()
                save_state(st)
                return
        clear_lock(window_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Execute real trades (default dry-run)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output signal/errors")
    parser.add_argument("--smart-sizing", action="store_true", help="Use portfolio based sizing")
    args = parser.parse_args()

    cfg = load_config()
    run_once(cfg, live=args.live, quiet=args.quiet, smart_sizing=args.smart_sizing)