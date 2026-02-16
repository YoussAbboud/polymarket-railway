#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (one cycle then exit).

Fixes included (your 2 issues):
1) ✅ Crash on trade signal:
   - import_market() returns 3 values (market_id, err, imported_flag)
   - your run_once unpacked only 2 -> ValueError. Fixed.

2) ✅ “Same trade opens in one window then closes in another” / imprecise execution:
   - Hardened duplicate-prevention:
        * per-window lock is written BEFORE import/trade (prevents double-fire on crashes)
        * lock is cleared on hard failure (so you don’t miss the whole window)
        * also checks existing positions by market_id and by window signature
   - Hardened auto-close:
        * only closes EXPIRED positions by default
        * “future” positions are only closed if they are FAR in the future (likely a mistaken entry)
          so it won’t accidentally close a live position due to parsing edge cases.

Other hardening:
- Better parsing for month name (Jan/January) and robust year/day handling
- More defensive datetime parsing (state + simmer)
- Trade verification path if Simmer times out
"""

import os, sys, json, argparse, time
from datetime import datetime, timezone, timedelta, date
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

ASSET_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
ASSET_PATTERNS = {
    "BTC": ["bitcoin up or down"],
    "ETH": ["ethereum up or down"],
    "SOL": ["solana up or down"],
}

TRADE_SOURCE = "railway:fastloop"
MIN_SHARES_PER_ORDER = 5  # lower to 3 if you want smaller orders

# Lock TTL (seconds) to prevent duplicate orders on timeouts / crashes
LOCK_TTL_SECONDS = int(os.environ.get("LOCK_TTL_SECONDS", "900"))  # 15 minutes


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


def api_request(url, method="GET", data=None, headers=None, timeout=45):
    try:
        req_headers = headers or {}
        req_headers.setdefault("User-Agent", "railway-fastloop/1.1")
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
            return {"error": err.get("detail", str(e)), "status_code": e.code}
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
# Lock helpers (per window)
# -----------------------
def lock_path_for(key: str) -> str:
    safe = (key or "").replace("/", "_").replace(" ", "_")
    return os.path.join(DATA_DIR, f"lock_{safe}.json")


def read_lock(key: str):
    path = lock_path_for(key)
    data = load_json(path, None)
    if not isinstance(data, dict) or "ts" not in data:
        return None
    try:
        ts = datetime.fromisoformat(data["ts"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if (now_utc() - ts).total_seconds() < LOCK_TTL_SECONDS:
            return data
    except Exception:
        return None
    return None


def has_recent_lock(key: str) -> bool:
    return read_lock(key) is not None


def write_lock(key: str, extra: dict = None):
    path = lock_path_for(key)
    payload = {"ts": now_utc().isoformat(), "key": key}
    if isinstance(extra, dict):
        payload.update(extra)
    save_json(path, payload)


def clear_lock(key: str):
    path = lock_path_for(key)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# -----------------------
# Config (config.json > env > defaults)
# -----------------------
CONFIG_DEFAULTS = {
    "asset": "BTC",
    "window": "5m",
    "signal_source": "binance",
    "lookback_minutes": 5,
    "entry_threshold": 0.05,
    "min_momentum_pct": 0.30,
    "min_time_remaining": 45,
    "min_volume_ratio": 0.60,
    "max_position": 5.0,
    "smart_sizing_pct": 0.05,
    "max_open_fast_positions": 1,
    "cooldown_seconds": 120,
    "daily_trade_limit": 20,
    "daily_import_limit": 10,
    "fee_edge_buffer": 0.02,
    # only auto-close “future” positions if they are far out (likely a bug)
    "future_close_min_seconds": 8 * 60,  # 8 minutes
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
    if isinstance(file_cfg, dict):
        cfg.update(file_cfg)

    for k, env in ENV_MAP.items():
        v = os.environ.get(env)
        if v is None:
            continue
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
    """
    Parses:
      "Bitcoin Up or Down - February 16, 10:40PM-10:45PM ET"
    Returns (start_utc, end_utc) timezone-aware UTC datetimes.
    Uses America/New_York so DST is handled (EST/EDT).
    """
    import re
    from zoneinfo import ZoneInfo

    q = (question or "").strip()

    # allow "Feb" or "February"
    m = re.search(
        r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{1,2}:\d{2}(?:AM|PM))-(\d{1,2}:\d{2}(?:AM|PM))\s*ET",
        q
    )
    if not m:
        return None

    month_str = m.group(1)
    day = int(m.group(2))
    t1 = m.group(3)
    t2 = m.group(4)

    ny = ZoneInfo("America/New_York")

    # choose year that makes most sense around "now" (handles UTC date rollover)
    year_now = now_utc().year
    candidates = [year_now, year_now - 1, year_now + 1]

    # try parsing month as full then abbreviated
    def try_parse(month_fmt: str, y: int):
        try:
            start_local = datetime.strptime(f"{month_str} {day} {y} {t1}", month_fmt).replace(tzinfo=ny)
            end_local   = datetime.strptime(f"{month_str} {day} {y} {t2}", month_fmt).replace(tzinfo=ny)
            return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)
        except Exception:
            return None

    best = None
    best_delta = None
    for y in candidates:
        win = try_parse("%B %d %Y %I:%M%p", y) or try_parse("%b %d %Y %I:%M%p", y)
        if not win:
            continue
        start_utc, end_utc = win
        # pick the year with start closest to now (within a reasonable range)
        delta = abs((start_utc - now_utc()).total_seconds())
        if best is None or delta < best_delta:
            best, best_delta = win, delta

    return best


def parse_iso_dt(s):
    if not s:
        return None
    try:
        s = str(s).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def gamma_market_window(m: dict):
    start = parse_iso_dt(m.get("startDateIso")) or parse_iso_dt(m.get("start_date_iso"))
    end   = parse_iso_dt(m.get("endDateIso"))   or parse_iso_dt(m.get("end_date_iso"))

    if not start and m.get("startDate") is not None:
        try:
            ts = float(m["startDate"])
            if ts > 1e12:
                ts /= 1000.0
            start = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            start = None

    if not end and m.get("endDate") is not None:
        try:
            ts = float(m["endDate"])
            if ts > 1e12:
                ts /= 1000.0
            end = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            end = None

    return start, end


def is_live_window(start: datetime, end: datetime, *, grace_s=20) -> bool:
    if not start or not end:
        return False
    now = now_utc()
    return (start - timedelta(seconds=grace_s)) <= now <= (end + timedelta(seconds=grace_s))


def has_time_remaining(end: datetime, min_seconds_left: int) -> bool:
    if not end:
        return False
    return (end - now_utc()).total_seconds() >= min_seconds_left


def discover_fast_markets(asset: str, window: str):
    patterns = ASSET_PATTERNS.get(asset, ASSET_PATTERNS["BTC"])
    result = api_request(GAMMA_MARKETS_URL)
    if not isinstance(result, list):
        return []

    out = []
    for m in result:
        q_raw = (m.get("question") or "")
        q = q_raw.lower()
        slug = (m.get("slug") or "")

        if not any(p in q for p in patterns):
            continue
        if m.get("closed"):
            continue

        start, end = gamma_market_window(m)

        # only currently live windows
        if not is_live_window(start, end, grace_s=20):
            continue

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
        if not end:
            continue
        left = (end - now).total_seconds()
        if left < min_seconds_left:
            continue
        live.append((left, m))

    if not live:
        return None

    live.sort(key=lambda x: x[0])  # soonest ending = current window
    return live[0][1]


def market_window_key(question: str) -> str:
    """Stable key for the specific 5-min window (used for locking / dedupe)."""
    win = parse_fast_market_window_utc(question or "")
    if not win:
        return (question or "").strip()
    s, e = win
    return f"{(question or '').split('-')[0].strip()}|{s.isoformat()}|{e.isoformat()}"


# -----------------------
# Signal
# -----------------------
def get_binance_momentum(symbol: str, lookback_minutes: int):
    bases = [
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
        "https://data-api.binance.vision"
    ]

    last_err = None
    for base in bases:
        url = f"{base}/api/v3/klines?symbol={symbol}&interval=1m&limit={lookback_minutes}"
        candles = api_request(url, timeout=25)

        if isinstance(candles, dict) and candles.get("error"):
            last_err = {"base": base, **candles}
            continue

        if not isinstance(candles, list) or len(candles) < 2:
            last_err = {"base": base, "error": "bad_response", "sample": str(candles)[:200]}
            continue

        try:
            # open of first candle vs close of last candle
            price_then = float(candles[0][1])
            price_now = float(candles[-1][4])
            momentum_pct = ((price_now - price_then) / price_then) * 100.0
            direction = "up" if momentum_pct > 0 else "down"

            vols = [float(c[5]) for c in candles]
            avg_vol = sum(vols) / len(vols) if vols else 0.0
            latest_vol = vols[-1] if vols else 0.0
            vol_ratio = (latest_vol / avg_vol) if avg_vol > 0 else 1.0

            return {
                "price_now": price_now,
                "price_then": price_then,
                "momentum_pct": momentum_pct,
                "direction": direction,
                "volume_ratio": vol_ratio,
                "binance_base": base,
            }
        except Exception as e:
            last_err = {"base": base, "error": f"parse_error: {e}"}
            continue

    return {"error": "all_binance_endpoints_failed", "details": last_err}


# -----------------------
# Simmer actions
# -----------------------
def get_positions(api_key: str):
    r = simmer_request("/api/sdk/positions", api_key=api_key, timeout=45)
    if isinstance(r, dict) and "positions" in r:
        return r["positions"]
    if isinstance(r, list):
        return r
    return []


def get_portfolio(api_key: str):
    return simmer_request("/api/sdk/portfolio", api_key=api_key, timeout=45)


def import_market(api_key: str, slug: str):
    url = f"https://polymarket.com/event/{slug}"
    r = simmer_request(
        "/api/sdk/markets/import",
        method="POST",
        data={"polymarket_url": url, "shared": True},
        api_key=api_key,
        timeout=90
    )
    if not isinstance(r, dict) or r.get("error"):
        return None, (r.get("error") if isinstance(r, dict) else "import failed"), False

    status = r.get("status")
    if status == "imported":
        return r.get("market_id"), None, True
    if status == "already_exists":
        return r.get("market_id"), None, False

    return None, f"unexpected import status: {status}", False


def execute_trade(api_key: str, market_id: str, side: str, amount: float = None, shares: float = None, action: str = "buy"):
    payload = {
        "market_id": market_id,
        "side": side,               # "yes" or "no"
        "venue": "polymarket",
        "source": TRADE_SOURCE,
        "action": action,           # "buy" or "sell"
    }
    if action == "sell":
        payload["shares"] = float(shares or 0)
    else:
        payload["amount"] = float(amount or 0)

    return simmer_request(
        "/api/sdk/trade",
        method="POST",
        data=payload,
        api_key=api_key,
        timeout=90,
    )


def calc_position_size(api_key: str, max_size: float, smart_pct: float, smart_sizing: bool):
    if not smart_sizing:
        return max_size
    pf = get_portfolio(api_key)
    if not isinstance(pf, dict) or pf.get("error"):
        return max_size
    bal = float(pf.get("balance_usdc", 0) or 0)
    if bal <= 0:
        return max_size
    return min(max_size, bal * smart_pct)


# -----------------------
# Risk + state
# -----------------------
def load_state():
    st = load_json(STATE_PATH, {})
    if not isinstance(st, dict):
        st = {}
    today = date.today().isoformat()
    if st.get("day") != today:
        st = {"day": today, "trades": 0, "imports": 0, "last_trade_ts": None}
    st.setdefault("imports", 0)
    return st


def save_state(st: dict):
    save_json(STATE_PATH, st)


def parse_state_dt(dt_str: str):
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def classify_position_by_question(p: dict):
    """
    Returns: ("future" | "live" | "expired" | "unknown"), start_utc, end_utc
    """
    q = (p.get("question") or "").strip()
    win = parse_fast_market_window_utc(q)
    if not win:
        return "unknown", None, None

    start_utc, end_utc = win
    now = now_utc()

    if start_utc > now + timedelta(minutes=2):
        return "future", start_utc, end_utc

    if end_utc < now - timedelta(seconds=30):
        return "expired", start_utc, end_utc

    return "live", start_utc, end_utc


def is_market_expired(question: str, *, grace_after_s: int = 30) -> bool:
    win = parse_fast_market_window_utc(question or "")
    if not win:
        return False
    _, end_utc = win
    return end_utc < (now_utc() - timedelta(seconds=grace_after_s))


def is_position_live_now(p: dict, *, grace_before_s: int = 20, grace_after_s: int = 20) -> bool:
    q = (p.get("question") or "").strip()
    win = parse_fast_market_window_utc(q)
    if not win:
        return False
    start_utc, end_utc = win
    now = now_utc()
    return (start_utc - timedelta(seconds=grace_before_s)) <= now <= (end_utc + timedelta(seconds=grace_after_s))


def count_open_fast_positions(positions):
    n = 0
    for p in positions or []:
        ql = (p.get("question") or "").lower()
        if "up or down" not in ql:
            continue

        yes = float(p.get("shares_yes", 0) or 0)
        no  = float(p.get("shares_no", 0) or 0)
        if yes <= 0 and no <= 0:
            continue

        if not is_position_live_now(p):
            continue

        n += 1
    return n


def already_in_this_market(positions, question: str, market_id: str = None):
    """
    Strong dedupe:
    - if market_id is known, match by market_id first
    - else match by question exact OR by window key
    """
    target_q = (question or "").strip()
    target_key = market_window_key(target_q)

    for p in positions or []:
        pq = (p.get("question") or "").strip()

        # market_id match (best)
        pid = str(p.get("market_id") or p.get("id") or p.get("marketId") or "")
        if market_id and pid and str(market_id) == pid:
            yes = float(p.get("shares_yes", 0) or 0)
            no  = float(p.get("shares_no", 0) or 0)
            if yes > 0 or no > 0:
                return True

        # question match
        if pq == target_q or market_window_key(pq) == target_key:
            if is_market_expired(pq):
                continue
            yes = float(p.get("shares_yes", 0) or 0)
            no  = float(p.get("shares_no", 0) or 0)
            if yes > 0 or no > 0:
                return True

    return False


# -----------------------
# Main cycle
# -----------------------
def close_positions(api_key: str, positions, quiet: bool = False, future_close_min_seconds: int = 480):
    """
    Safer closing:
    - Always closes EXPIRED.
    - Only closes FUTURE if it's FAR in the future (likely mistaken entry or stale import).
    """
    def log(msg):
        if not quiet:
            print(msg)

    closed_any = False
    now = now_utc()

    for p in positions or []:
        q = (p.get("question") or "")
        if "up or down" not in q.lower():
            continue

        status, start_utc, end_utc = classify_position_by_question(p)
        if status == "live" or status == "unknown":
            continue

        if status == "future":
            if not start_utc:
                continue
            # only close if it's far out (avoid accidental close due to parsing edge)
            if (start_utc - now).total_seconds() < future_close_min_seconds:
                continue

        # status is expired OR far-future
        market_id = p.get("market_id") or p.get("id") or p.get("marketId")
        if not market_id:
            continue

        shares_yes = float(p.get("shares_yes", 0) or 0)
        shares_no  = float(p.get("shares_no", 0) or 0)

        log(f"AUTO-CLOSE: {status} | {q} | start={start_utc} end={end_utc}")

        if shares_yes > 0:
            res = execute_trade(api_key, market_id, "yes", shares=shares_yes, action="sell")
            append_journal({"type":"close_attempt","why":status,"side":"sell_yes","market_id":market_id,"shares":shares_yes,"result":res})
            closed_any = True

        if shares_no > 0:
            res = execute_trade(api_key, market_id, "no", shares=shares_no, action="sell")
            append_journal({"type":"close_attempt","why":status,"side":"sell_no","market_id":market_id,"shares":shares_no,"result":res})
            closed_any = True

    return closed_any


def run_once(cfg, live: bool, quiet: bool, smart_sizing: bool):
    def log(msg, force=False):
        if not quiet or force:
            print(msg)

    api_key = get_api_key()
    st = load_state()

    run_id = now_utc().isoformat()
    print(f"DEBUG RUN_ID={run_id}")

    if os.environ.get("TRADING_ENABLED", "true").lower() not in ("true", "1", "yes"):
        log("SKIP: TRADING_ENABLED is false", force=True)
        append_journal({"type": "skip", "reason": "trading_disabled"})
        return

    positions = get_positions(api_key)
    print(f"DEBUG: positions fetched = {len(positions)}")

    closed_any = close_positions(
        api_key,
        positions,
        quiet=quiet,
        future_close_min_seconds=int(cfg.get("future_close_min_seconds", 480))
    )
    if closed_any:
        positions = get_positions(api_key)
        print(f"DEBUG: positions after close attempt = {len(positions)}")

    open_fast = count_open_fast_positions(positions)
    if open_fast >= int(cfg["max_open_fast_positions"]):
        log(f"SKIP: open fast positions ({open_fast}) >= cap ({cfg['max_open_fast_positions']})", force=True)
        append_journal({"type": "skip", "reason": "max_open_positions", "open_fast": open_fast})
        return

    if st.get("last_trade_ts"):
        last = parse_state_dt(st["last_trade_ts"])
        if last:
            since = (now_utc() - last).total_seconds()
            if since < int(cfg["cooldown_seconds"]):
                log(f"SKIP: cooldown ({since:.0f}s < {cfg['cooldown_seconds']}s)", force=True)
                append_journal({"type": "skip", "reason": "cooldown", "since_s": since})
                return

    if int(st.get("trades", 0)) >= int(cfg["daily_trade_limit"]):
        log(f"SKIP: daily trade limit reached ({st.get('trades')} >= {cfg['daily_trade_limit']})", force=True)
        append_journal({"type": "skip", "reason": "daily_trade_limit", "trades": st.get("trades")})
        return

    markets = discover_fast_markets(cfg["asset"], cfg["window"])
    if not markets:
        log("SKIP: no active fast markets")
        append_journal({"type": "skip", "reason": "no_markets"})
        return

    best = select_best_market(markets, int(cfg["min_time_remaining"]))
    if not best:
        log("SKIP: no market with enough time remaining", force=True)
        append_journal({"type": "skip", "reason": "no_best_market"})
        return

    start = best.get("start_time")
    end   = best.get("end_time")

    if not is_live_window(start, end, grace_s=20):
        log("SKIP: selected market is not live (guard)", force=True)
        append_journal({"type": "skip", "reason": "best_not_live", "slug": best.get("slug")})
        return

    if not has_time_remaining(end, int(cfg["min_time_remaining"])):
        log("SKIP: too close to expiry", force=True)
        append_journal({"type": "skip", "reason": "too_close_to_expiry", "slug": best.get("slug")})
        return

    # Per-window key (stronger than slug)
    window_key = market_window_key(best["question"])
    slug = best.get("slug") or ""

    # ✅ lock check (prevents duplicate orders across cron runs / crashes)
    if has_recent_lock(window_key):
        lk = read_lock(window_key) or {}
        log(f"SKIP: recent lock exists for this window (status={lk.get('status')})", force=True)
        append_journal({"type": "skip", "reason": "recent_lock", "window_key": window_key, "slug": slug})
        return

    # Quick check for existing position in the same window
    if already_in_this_market(positions, best["question"]):
        log("SKIP: already have a position in this exact market window", force=True)
        append_journal({"type": "skip", "reason": "already_in_market", "question": best["question"]})
        return

    # Parse prices
    try:
        prices = json.loads(best.get("outcome_prices", "[]"))
        yes_price = float(prices[0]) if prices else 0.5
    except Exception:
        yes_price = 0.5

    fee_rate = (int(best.get("fee_rate_bps", 0)) / 10000.0) if best.get("fee_rate_bps") else 0.0

    # Signal
    symbol = ASSET_SYMBOLS.get(cfg["asset"], "BTCUSDT")
    sig = get_binance_momentum(symbol, int(cfg["lookback_minutes"]))
    if not sig or (isinstance(sig, dict) and sig.get("error")):
        log(f"SKIP: signal fetch failed | {sig}", force=True)
        append_journal({"type": "skip", "reason": "signal_fetch_failed", "detail": sig})
        return

    mom_abs = abs(sig["momentum_pct"])
    if mom_abs < float(cfg["min_momentum_pct"]):
        log(f"SKIP: momentum {mom_abs:.3f}% < {cfg['min_momentum_pct']}%", force=True)
        append_journal({"type": "skip", "reason": "weak_momentum", "momentum_pct": sig["momentum_pct"]})
        return

    if float(sig["volume_ratio"]) < float(cfg["min_volume_ratio"]):
        log(f"SKIP: volume ratio {sig['volume_ratio']:.2f} < {cfg['min_volume_ratio']}", force=True)
        append_journal({"type": "skip", "reason": "low_volume", "volume_ratio": sig["volume_ratio"]})
        return

    entry = float(cfg["entry_threshold"])
    if sig["direction"] == "up":
        side = "yes"
        divergence = (0.50 + entry) - yes_price
        buy_price = yes_price
    else:
        side = "no"
        divergence = yes_price - (0.50 - entry)
        buy_price = 1.0 - yes_price

    if divergence <= 0:
        log("SKIP: market already priced in", force=True)
        append_journal({"type": "skip", "reason": "priced_in", "yes_price": yes_price, "divergence": divergence})
        return

    if fee_rate > 0:
        win_profit = (1 - buy_price) * (1 - fee_rate)
        breakeven = buy_price / (win_profit + buy_price)
        fee_penalty = breakeven - 0.50
        min_div = fee_penalty + float(cfg["fee_edge_buffer"])
        if divergence < min_div:
            log("SKIP: fees eat edge", force=True)
            append_journal({"type": "skip", "reason": "fee_edge", "divergence": divergence, "min_div": min_div})
            return

    amount = calc_position_size(
        api_key=api_key,
        max_size=float(cfg["max_position"]),
        smart_pct=float(cfg["smart_sizing_pct"]),
        smart_sizing=smart_sizing,
    )

    if buy_price > 0:
        min_cost = MIN_SHARES_PER_ORDER * buy_price
        if min_cost > amount:
            log(f"SKIP: amount ${amount:.2f} too small for {MIN_SHARES_PER_ORDER} shares at ${buy_price:.2f}", force=True)
            append_journal({"type": "skip", "reason": "min_shares", "amount": amount, "buy_price": buy_price})
            return

    log(f"SIGNAL: {side.upper()} | YES=${yes_price:.3f} | mom={sig['momentum_pct']:+.3f}% | vol={sig['volume_ratio']:.2f}x", force=True)

    if int(st.get("imports", 0)) >= int(cfg.get("daily_import_limit", 10)):
        log(f"SKIP: daily import limit reached ({st.get('imports')} >= {cfg.get('daily_import_limit',10)})", force=True)
        append_journal({"type":"skip","reason":"daily_import_limit","imports":st.get("imports")})
        return

    # ✅ write a “pending” lock BEFORE import/trade so a crash can’t double-fire
    write_lock(window_key, {"status": "pending", "slug": slug, "question": best["question"]})

    # ✅ FIX 1: correct unpacking (3 values)
    market_id, err, imported_flag = import_market(api_key, slug)

    if not market_id:
        log(f"FAIL: import {err}", force=True)
        append_journal({"type": "error", "stage": "import", "error": err, "slug": slug, "window_key": window_key})
        clear_lock(window_key)  # allow retry this window if import truly failed
        return

    # update lock with market id
    write_lock(window_key, {"status": "imported", "slug": slug, "market_id": str(market_id), "question": best["question"]})

    # refresh positions and re-check: if we already hold this window, do NOT buy again
    positions_refresh = get_positions(api_key)
    if already_in_this_market(positions_refresh, best["question"], market_id=market_id):
        log("SKIP: position already exists after import (dedupe)", force=True)
        append_journal({"type":"skip","reason":"already_in_market_post_import","market_id":market_id,"window_key":window_key})
        # keep lock to prevent re-fires
        write_lock(window_key, {"status": "held", "slug": slug, "market_id": str(market_id), "question": best["question"]})
        return

    if not live:
        est_shares = (amount / buy_price) if buy_price > 0 else 0
        log(f"DRY RUN: would buy {side.upper()} ${amount:.2f} (~{est_shares:.1f} shares)", force=True)
        append_journal({
            "type": "dry_run",
            "question": best["question"],
            "slug": slug,
            "market_id": market_id,
            "side": side,
            "amount": amount,
            "yes_price": yes_price,
            "buy_price": buy_price,
            "momentum_pct": sig["momentum_pct"],
            "volume_ratio": sig["volume_ratio"],
            "divergence": divergence,
            "fee_rate": fee_rate,
            "imported": imported_flag,
        })
        # keep lock so it doesn't “dry-run spam”
        write_lock(window_key, {"status": "dry_run", "slug": slug, "market_id": str(market_id)})
        return

    # Execute trade
    write_lock(window_key, {"status": "trading", "slug": slug, "market_id": str(market_id)})

    result = execute_trade(api_key, market_id, side, amount=amount, action="buy")
    if isinstance(result, dict) and result.get("success"):
        st["trades"] = int(st.get("trades", 0)) + 1
        st["last_trade_ts"] = now_utc().isoformat()
        st["imports"] = int(st.get("imports", 0)) + (1 if imported_flag else 0)
        save_state(st)

        append_journal({
            "type": "trade",
            "question": best["question"],
            "slug": slug,
            "window_key": window_key,
            "market_id": market_id,
            "side": side,
            "amount": amount,
            "yes_price": yes_price,
            "buy_price": buy_price,
            "momentum_pct": sig["momentum_pct"],
            "volume_ratio": sig["volume_ratio"],
            "divergence": divergence,
            "fee_rate": fee_rate,
            "trade_id": result.get("trade_id"),
        })
        log("TRADE: success", force=True)
        write_lock(window_key, {"status": "done", "slug": slug, "market_id": str(market_id), "trade_id": result.get("trade_id")})
    else:
        err2 = (result.get("error") if isinstance(result, dict) else "no response")
        append_journal({"type": "error", "stage": "trade", "error": err2, "market_id": market_id, "window_key": window_key})
        log(f"TRADE: failed ({err2})", force=True)

        # If timeout, verify whether trade actually landed
        if "timed out" in str(err2).lower():
            time.sleep(2)
            positions2 = get_positions(api_key)
            if already_in_this_market(positions2, best["question"], market_id=market_id):
                log("TRADE: success (verified after timeout)", force=True)
                append_journal({"type": "trade_verified", "market_id": market_id, "question": best["question"], "window_key": window_key})
                write_lock(window_key, {"status": "done_verified", "slug": slug, "market_id": str(market_id)})
                # count it as a trade
                st["trades"] = int(st.get("trades", 0)) + 1
                st["last_trade_ts"] = now_utc().isoformat()
                save_state(st)
                return

        # Hard fail (not timeout): clear lock so you can retry this window if you want
        clear_lock(window_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Execute real trades (default dry-run)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output signal/errors")
    parser.add_argument("--smart-sizing", action="store_true", help="Use portfolio-based sizing")
    args = parser.parse_args()

    cfg = load_config()
    run_once(cfg, live=args.live, quiet=args.quiet, smart_sizing=args.smart_sizing)
