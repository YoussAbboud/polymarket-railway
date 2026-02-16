#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (one cycle then exit).

Core idea is the same as polymarket-clawd:
- Discover Polymarket fast markets via Gamma
- Pull Binance 1m klines to compute momentum
- If signal passes thresholds, import market via Simmer and trade via Simmer

Improvements:
- Prevent duplicate entries (skip if you already have a position for the same question)
- Risk caps: max open positions, cooldown, daily trade limit
- Persistent state + trade journal on /data (Railway Volume)

Additional hardening:
- Local lock w/ TTL in /data to prevent duplicate orders on API timeouts
- Longer Simmer timeouts for import/trade
- Verify position after timeout (trade may have succeeded server-side)
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
MIN_SHARES_PER_ORDER = 5  # consider 3 if you want smaller trades

# Lock TTL (seconds) to prevent duplicate orders on timeouts / lag
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
        req_headers.setdefault("User-Agent", "railway-fastloop/1.0")
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        req = Request(url, data=body, headers=req_headers, method=method)
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8"))
            return {"error": err.get("detail", str(e)), "status_code": e.code}
        except Exception:
            return {"error": str(e), "status_code": e.code}
    except URLError as e:
        return {"error": f"Connection error: {e.reason}"}
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
def lock_path_for(slug: str) -> str:
    safe = (slug or "").replace("/", "_")
    return os.path.join(DATA_DIR, f"lock_{safe}.json")


def has_recent_lock(slug: str) -> bool:
    path = lock_path_for(slug)
    data = load_json(path, None)
    if not isinstance(data, dict) or "ts" not in data:
        return False
    try:
        ts = datetime.fromisoformat(data["ts"])
        return (now_utc() - ts).total_seconds() < LOCK_TTL_SECONDS
    except Exception:
        return False


def write_lock(slug: str):
    path = lock_path_for(slug)
    save_json(path, {"ts": now_utc().isoformat(), "slug": slug})


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
      "Bitcoin Up or Down - February 16, 3:45PM-3:50PM ET"
    Returns (start_utc, end_utc) timezone-aware UTC datetimes.
    Uses America/New_York so DST is handled (EST/EDT).
    """
    import re
    from zoneinfo import ZoneInfo

    q = (question or "").strip()
    m = re.search(
        r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{1,2}:\d{2}(?:AM|PM))-(\d{1,2}:\d{2}(?:AM|PM))\s*ET",
        q
    )
    if not m:
        return None

    try:
        year = now_utc().year
        month = m.group(1)
        day = int(m.group(2))
        t1 = m.group(3)
        t2 = m.group(4)

        ny = ZoneInfo("America/New_York")

        start_local = datetime.strptime(f"{month} {day} {year} {t1}", "%B %d %Y %I:%M%p").replace(tzinfo=ny)
        end_local   = datetime.strptime(f"{month} {day} {year} {t2}", "%B %d %Y %I:%M%p").replace(tzinfo=ny)

        return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)
    except Exception:
        return None



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

        # ✅ ONLY CURRENTLY LIVE windows
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



def parse_ts_any(v):
    """
    Accepts numeric seconds, numeric ms, or numeric strings.
    Returns UTC datetime or None.
    """
    if v is None:
        return None
    try:
        # If it's already a datetime
        if isinstance(v, datetime):
            return v.astimezone(timezone.utc)

        # ISO string?
        iso = parse_iso_dt(v)
        if iso:
            return iso

        # numeric string/number
        ts = float(v)
        if ts > 1e12:  # ms
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None


def gamma_market_window(m: dict):
    # Prefer ISO fields if present
    start = parse_iso_dt(m.get("startDateIso")) or parse_iso_dt(m.get("start_date_iso"))
    end   = parse_iso_dt(m.get("endDateIso"))   or parse_iso_dt(m.get("end_date_iso"))

    # Fallback: numeric fields (sometimes ms) — HARDENED
    if not start and m.get("startDate") is not None:
        try:
            ts = float(m["startDate"])
            if ts > 1e12:  # ms
                ts /= 1000.0
            start = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            start = None

    if not end and m.get("endDate") is not None:
        try:
            ts = float(m["endDate"])
            if ts > 1e12:  # ms
                ts /= 1000.0
            end = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            end = None

    return start, end


def is_live_window(start: datetime, end: datetime, *, grace_s=20) -> bool:
    if not start or not end:
        return False
    now = now_utc()
    # allow a small grace before/after
    return (start - timedelta(seconds=grace_s)) <= now <= (end + timedelta(seconds=grace_s))

def has_time_remaining(end: datetime, min_seconds_left: int) -> bool:
    if not end:
        return False
    return (end - now_utc()).total_seconds() >= min_seconds_left

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
        timeout=60
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
        "side": side,
        "venue": "polymarket",
        "source": TRADE_SOURCE,
        "action": action,  # "buy" or "sell"
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
        timeout=60,
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


def _parse_iso_utc(dt_str: str):
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str)
        # Simmer gives naive ISO (no tz). Treat as UTC.
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

        # ✅ only count positions that are live right now
        if not is_position_live_now(p):
            continue

        n += 1
    return n






def already_in_this_market(positions, question: str):
    target = (question or "").strip()
    for p in positions or []:
        pq = (p.get("question") or "").strip()
        if pq != target:
            continue

        # ✅ expired market shouldn't block new trade
        if is_market_expired(pq):
            return False

        yes = float(p.get("shares_yes", 0) or 0)
        no  = float(p.get("shares_no", 0) or 0)
        if yes > 0 or no > 0:
            return True

    return False


def active_fast_market_slugs(asset: str, window: str):
    markets = discover_fast_markets(asset, window)
    return set(m["slug"] for m in markets if m.get("slug"))

# -----------------------
# Main cycle
# -----------------------
def close_future_positions(api_key: str, positions, quiet: bool = False):
    def log(msg):
        if not quiet:
            print(msg)

    closed_any = False

    for p in positions or []:
        q = (p.get("question") or "")
        if "up or down" not in q.lower():
            continue

        status, start_utc, end_utc = classify_position_by_question(p)
        if status not in ("future", "expired"):
            continue  # do NOT close live positions

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
    # DEBUG: print one position so we can see the exact Simmer fields
    if positions:
        print("DEBUG position keys:", list(positions[0].keys()))
        print("DEBUG first position sample:", json.dumps(positions[0], indent=2)[:4000])
    else:
        print("DEBUG: no positions returned")

    print(f"DEBUG: positions fetched = {len(positions)}")
    print("DEBUG: running close_future_positions() now...")
    closed_any = close_future_positions(api_key, positions, quiet=quiet)
    print(f"DEBUG: close_future_positions returned {closed_any}")

    print("DEBUG: after close_future_positions, continuing…")
    print(f"DEBUG: state trades={st.get('trades')} day={st.get('day')} last_trade_ts={st.get('last_trade_ts')}")
    print(f"DEBUG: cfg daily_trade_limit={cfg['daily_trade_limit']} cooldown={cfg['cooldown_seconds']} cap={cfg['max_open_fast_positions']}")

    if closed_any:
        print("INFO: attempted to close future positions")
        # refresh positions after closing
        positions = get_positions(api_key)
        print(f"DEBUG: positions after close attempt = {len(positions)}")

    open_fast = count_open_fast_positions(positions)
    if open_fast >= int(cfg["max_open_fast_positions"]):
        log(f"SKIP: open fast positions ({open_fast}) >= cap ({cfg['max_open_fast_positions']})", force=True)
        append_journal({"type": "skip", "reason": "max_open_positions", "open_fast": open_fast})
        return

    if st.get("last_trade_ts"):
        last = datetime.fromisoformat(st["last_trade_ts"])
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
    print("DEBUG: starting market discovery…")
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

    # Safety: should already be live because discover_fast_markets hard-filters it,
    # but keep a guard anyway.
    if not is_live_window(start, end, grace_s=20):
        log(f"SKIP: selected market is not live (gamma start={start}, end={end})", force=True)
        append_journal({"type": "skip", "reason": "best_not_live", "slug": best.get("slug"), "question": best.get("question")})
        return

    if not has_time_remaining(end, int(cfg["min_time_remaining"])):
        log("SKIP: no market with enough time remaining", force=True)
        append_journal({"type": "skip", "reason": "too_close_to_expiry", "slug": best.get("slug"), "question": best.get("question")})
        return


    # ✅ lock check (prevents duplicate orders across cron runs)
    if has_recent_lock(best["slug"]):
        log("SKIP: recent lock exists for this market (prevents duplicate orders)", force=True)
        append_journal({"type": "skip", "reason": "recent_lock", "slug": best["slug"]})
        return

    if already_in_this_market(positions, best["question"]):
        log("SKIP: already have a position in this exact market window", force=True)
        append_journal({"type": "skip", "reason": "already_in_market", "question": best["question"]})
        return

    try:
        prices = json.loads(best.get("outcome_prices", "[]"))
        yes_price = float(prices[0]) if prices else 0.5
    except Exception:
        yes_price = 0.5

    fee_rate = (int(best.get("fee_rate_bps", 0)) / 10000.0) if best.get("fee_rate_bps") else 0.0

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

    market_id, err = import_market(api_key, best["slug"])
    if not market_id:
        log(f"FAIL: import {err}", force=True)
        append_journal({"type": "error", "stage": "import", "error": err, "slug": best["slug"]})
        return

    if not live:
        est_shares = (amount / buy_price) if buy_price > 0 else 0
        log(f"DRY RUN: would buy {side.upper()} ${amount:.2f} (~{est_shares:.1f} shares)", force=True)
        append_journal({
            "type": "dry_run",
            "question": best["question"],
            "slug": best["slug"],
            "market_id": market_id,
            "side": side,
            "amount": amount,
            "yes_price": yes_price,
            "buy_price": buy_price,
            "momentum_pct": sig["momentum_pct"],
            "volume_ratio": sig["volume_ratio"],
            "divergence": divergence,
            "fee_rate": fee_rate,
        })
        return

    # ✅ write lock only when we're about to submit an order
    write_lock(best["slug"])

    result = execute_trade(api_key, market_id, side, amount)
    if isinstance(result, dict) and result.get("success"):
        st["trades"] = int(st.get("trades", 0)) + 1
        st["last_trade_ts"] = now_utc().isoformat()
        save_state(st)

        append_journal({
            "type": "trade",
            "question": best["question"],
            "slug": best["slug"],
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
    else:
        err = (result.get("error") if isinstance(result, dict) else "no response")
        append_journal({"type": "error", "stage": "trade", "error": err, "market_id": market_id})
        log(f"TRADE: failed ({err})", force=True)

        # ✅ If timeout, verify whether trade actually landed
        if "timed out" in str(err).lower():
            time.sleep(2)
            positions2 = get_positions(api_key)
            if already_in_this_market(positions2, best["question"]):
                log("TRADE: success (verified after timeout)", force=True)
                append_journal({"type": "trade_verified", "market_id": market_id, "question": best["question"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Execute real trades (default dry-run)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output signal/errors")
    parser.add_argument("--smart-sizing", action="store_true", help="Use portfolio-based sizing")
    args = parser.parse_args()

    cfg = load_config()
    run_once(cfg, live=args.live, quiet=args.quiet, smart_sizing=args.smart_sizing)
