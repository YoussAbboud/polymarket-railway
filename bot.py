#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (stable monitor + auto-close).

Key fixes:
- Auth happens FIRST and only once.
- Prevents multiple concurrent BTC 5m positions (uses Data API /positions).
- Fill + PnL tracking uses Data API (matches website), not flaky local balance/filled fields.
- Prints live PnL every 1 second.
- Auto-closes at TP/SL/buffer with a marketable SELL (0.01), and confirms closure via Data API.
"""

import os, sys, json, time, argparse, atexit
from datetime import datetime, timezone, timedelta
import requests

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

# ==============================================================================
# STRATEGY SETTINGS
# ==============================================================================
ASSET = "BTC"
LOOKBACK_MINS = 12
MIN_MOMENTUM_PCT = 0.12

# SAFETY SETTINGS
MAX_BET_SIZE = 5.0
STOP_LOSS_PCT = 0.15
TAKE_PROFIT_PCT = 0.15
CLOSE_BUFFER_SECONDS = 60
# ==============================================================================

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
DATA_API_POSITIONS = "https://data-api.polymarket.com/positions"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"

STATE_DIR = os.getenv("STATE_DIR", "/data")
STATE_PATH = os.path.join(STATE_DIR, "pm_state.json")
LOCK_PATH = os.path.join(STATE_DIR, "pm_lock")

SESSION = requests.Session()

# ------------------------ helpers ------------------------

def die(msg: str):
    print(msg, flush=True)
    sys.exit(1)

def get_env(key: str, default=None):
    val = os.environ.get(key, default)
    if val is None or val == "":
        die(f"❌ ERROR: Missing Env Variable: {key}")
    return val

def utc_now():
    return datetime.now(timezone.utc)

def round_to_5m(ts: datetime) -> datetime:
    return ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)

def slug_for_window(asset: str, window_start: datetime) -> str:
    return f"{asset.lower()}-updown-5m-{int(window_start.timestamp())}"

def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return None

def ensure_lock():
    os.makedirs(STATE_DIR, exist_ok=True)

    # If a stale lock exists, remove it (e.g., container crash)
    if os.path.exists(LOCK_PATH):
        try:
            age = time.time() - os.path.getmtime(LOCK_PATH)
            if age > 5 * 60:
                os.remove(LOCK_PATH)
        except Exception:
            pass

    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        die("⚠️ Another bot instance is running (lock exists). Exiting to avoid duplicate positions.")

    def _cleanup():
        try:
            if os.path.exists(LOCK_PATH):
                os.remove(LOCK_PATH)
        except Exception:
            pass

    atexit.register(_cleanup)

def load_state():
    if not os.path.exists(STATE_PATH):
        return None
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None

def save_state(state: dict):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)

def clear_state():
    try:
        if os.path.exists(STATE_PATH):
            os.remove(STATE_PATH)
    except Exception:
        pass

# ------------------------ polymarket / APIs ------------------------

def init_client():
    pk = get_env("PRIVATE_KEY")
    funder = get_env("POLYGON_ADDRESS")  # proxy wallet
    sig_type = int(os.getenv("SIGNATURE_TYPE", "1"))  # Magic/Email is usually 1

    print(f"📁 STATE_DIR = {STATE_DIR}", flush=True)
    print(f"🔐 Initializing Polymarket auth with signature_type={sig_type} ...", flush=True)

    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=sig_type, funder=funder)
    client.set_api_creds(client.create_or_derive_api_creds())

    print("✅ AUTH OK. Bot is Live.", flush=True)
    print("🧾 WALLET CHECK", flush=True)
    print("------------------------------------------------------", flush=True)
    print(f"🎯 Funder (proxy / funded wallet): {funder}", flush=True)
    print("------------------------------------------------------", flush=True)

    return client, funder, sig_type

def get_market_tokens(slug: str):
    # Gamma: /events?slug=...
    try:
        r = SESSION.get(f"{GAMMA_EVENTS_URL}?slug={slug}", timeout=8)
        data = safe_json(r)
        if not isinstance(data, list) or not data:
            return None

        markets = data[0].get("markets", [])
        if not markets:
            return None

        m = markets[0]
        clob_ids = json.loads(m.get("clobTokenIds", "[]"))
        if not isinstance(clob_ids, list) or len(clob_ids) < 2:
            return None

        # try to pull conditionId if present (useful for debugging; not required)
        condition_id = m.get("conditionId") or m.get("condition_id")

        return {
            "slug": slug,
            "market_id": m.get("id"),
            "condition_id": condition_id,
            "yes": str(clob_ids[0]),
            "no": str(clob_ids[1]),
        }
    except Exception:
        return None

def compute_momentum_pct(lookback_mins: int):
    # Uses CoinGecko 1-day chart; find price ~lookback_mins ago
    r = SESSION.get(COINGECKO_URL, timeout=8)
    j = safe_json(r) or {}
    prices = j.get("prices", [])
    if not prices or len(prices) < 5:
        return None

    # prices: [ [ms, price], ... ]
    now_ms, now_price = prices[-1]
    target_ms = now_ms - lookback_mins * 60_000

    # find closest price to target within ~6 minutes window
    best = None
    best_dist = 10**18
    for t, p in prices:
        d = abs(t - target_ms)
        if d < best_dist:
            best = p
            best_dist = d

    if best is None or best == 0:
        return None

    return ((now_price - best) / best) * 100.0

def data_api_positions(user: str, limit: int = 100):
    # Data API: /positions?user=...
    try:
        params = {
            "user": user,
            "sizeThreshold": 0,   # include small positions too
            "limit": limit,
            "offset": 0,
        }
        r = SESSION.get(DATA_API_POSITIONS, params=params, timeout=8)
        j = safe_json(r)
        if not isinstance(j, list):
            return []
        return j
    except Exception:
        return []

def find_position_for_asset(user: str, asset_token_id: str):
    positions = data_api_positions(user)
    for p in positions:
        if str(p.get("asset")) == str(asset_token_id) and float(p.get("size") or 0) > 0:
            return p
    return None

def find_any_open_btc5m_positions(user: str):
    prefix = f"{ASSET.lower()}-updown-5m-"
    positions = data_api_positions(user)
    out = []
    for p in positions:
        slug = (p.get("slug") or "")
        size = float(p.get("size") or 0)
        if size > 0 and slug.startswith(prefix):
            out.append(p)
    # newest first by slug timestamp if possible
    def slug_ts(p):
        s = p.get("slug") or ""
        try:
            return int(s.split("-")[-1])
        except Exception:
            return 0
    out.sort(key=slug_ts, reverse=True)
    return out

def best_bid_ask(client: ClobClient, token_id: str):
    try:
        ob = client.get_order_book(token_id)
        bid = float(ob.bids[0].price) if ob and ob.bids else None
        ask = float(ob.asks[0].price) if ob and ob.asks else None
        return bid, ask
    except Exception:
        return None, None

# ------------------------ trading ------------------------

def place_buy(client: ClobClient, token_id: str, dollars: float):
    bid, ask = best_bid_ask(client, token_id)
    if ask is None:
        raise RuntimeError("No asks in orderbook (cannot buy).")

    # conservative marketable buy: cross the best ask slightly
    tick = 0.01
    price = min(round(ask + tick, 2), 0.99)

    size = round(dollars / price, 4)  # keep precision; exchange will handle
    print(f"🚀 BUYING: {size:.4f} shares @ {price:.2f}...", flush=True)

    resp = client.create_and_post_order(
        OrderArgs(price=price, size=size, side=BUY, token_id=token_id)
    )
    oid = resp.get("orderID") if isinstance(resp, dict) else None
    if not oid:
        raise RuntimeError(f"Buy order failed: {resp}")
    print(f"✅ ORDER PLACED: {oid}", flush=True)
    return oid, price, size

def place_sell_marketable(client: ClobClient, token_id: str, size: float):
    # Marketable sell: very low limit price -> fills at best bid if bids exist
    price = 0.01
    size = round(float(size), 4)
    if size <= 0:
        return None

    print(f"🧾 SELLING (marketable): {size:.4f} @ {price:.2f}", flush=True)
    resp = client.create_and_post_order(
        OrderArgs(price=price, size=size, side=SELL, token_id=token_id)
    )
    oid = resp.get("orderID") if isinstance(resp, dict) else None
    if oid:
        print(f"✅ SELL ORDER PLACED: {oid}", flush=True)
    else:
        print(f"⚠️ Sell response: {resp}", flush=True)
    return oid

def monitor_and_autoclose(client: ClobClient, user: str, token_id: str, end_time: datetime,
                          tp_pct: float, sl_pct: float, buffer_s: int):
    """
    Prints PnL every 1s (matches website) and auto-closes at TP/SL/buffer.
    Uses Data API positions as the source of truth (fixes wrong PnL + fake 'not filled').
    """
    print("📊 MONITORING... (PnL 1s + AUTO-CLOSE)", flush=True)

    target_time = end_time - timedelta(seconds=buffer_s)
    tp = tp_pct * 100.0
    sl = -sl_pct * 100.0

    last_line_ts = 0.0

    while True:
        now = utc_now()
        secs_left = int((target_time - now).total_seconds())

        pos = find_position_for_asset(user, token_id)
        if not pos:
            # If position is gone, we're done (closed or resolved)
            print("✅ Position not found in Data API (closed/resolved).", flush=True)
            clear_state()
            return

        # Data API prices are in cents (matches the UI)
        avg_c = float(pos.get("avgPrice") or 0.0)
        cur_c = float(pos.get("curPrice") or 0.0)
        cash_pnl = float(pos.get("cashPnl") or 0.0)
        pct_pnl = float(pos.get("percentPnl") or 0.0)
        size = float(pos.get("size") or 0.0)

        # print every 1 second
        t = time.time()
        if t - last_line_ts >= 1.0:
            last_line_ts = t
            print(
                f"⏱️ Avg→Now: {avg_c:.0f}c→{cur_c:.0f}c | "
                f"Size: {size:.2f} | PnL: {pct_pnl:+.1f}% (${cash_pnl:+.2f}) | "
                f"closes in {max(secs_left,0)}s",
                flush=True
            )

        # triggers
        trigger = None
        if pct_pnl >= tp:
            trigger = "TAKE PROFIT"
        elif pct_pnl <= sl:
            trigger = "STOP LOSS"
        elif now >= target_time:
            trigger = "TIME BUFFER"

        if trigger:
            print(f"🧾 CLOSE TRIGGER: {trigger}", flush=True)

            # place sell and confirm position closes
            # retry a few times in case of transient allowance/balance sync delays
            for attempt in range(1, 11):
                try:
                    place_sell_marketable(client, token_id, size)
                except Exception as e:
                    print(f"⚠️ Sell attempt {attempt}/10 failed: {e}", flush=True)

                # wait briefly, then re-check position size
                time.sleep(1.5)
                pos2 = find_position_for_asset(user, token_id)
                if not pos2 or float(pos2.get("size") or 0.0) <= 0:
                    print("✅ CLOSED (position size is now 0).", flush=True)
                    clear_state()
                    return

                # update size in case it partially reduced
                size = float(pos2.get("size") or size)

            print("⚠️ Could not confirm close after retries. Position may still be open.", flush=True)
            # keep monitoring until expiry; DO NOT exit silently
        else:
            # keep monitoring until resolved/closed
            if secs_left < -120:
                # safety hard-stop well after target; market likely resolved
                print("⚠️ Monitor timeout past buffer. Exiting.", flush=True)
                return

        time.sleep(0.2)  # small sleep; printing still throttled to 1s

def run(live: bool):
    ensure_lock()
    client, user, sig_type = init_client()

    now = utc_now()
    ws = round_to_5m(now)
    slug = slug_for_window(ASSET, ws)
    end_time = ws + timedelta(minutes=5)

    print(f"🕒 Now UTC: {now.isoformat()} | live={live}", flush=True)

    # 1) If state exists, reconcile it with real positions
    state = load_state()
    if state and state.get("token_id"):
        tok = state["token_id"]
        pos = find_position_for_asset(user, tok)
        if pos and float(pos.get("size") or 0) > 0:
            print("🧠 Open trade found in state. Resuming monitoring (NO new orders).", flush=True)
            # prefer end_time from state if available
            try:
                et = datetime.fromtimestamp(int(state.get("end_ts")), tz=timezone.utc)
            except Exception:
                et = end_time
            monitor_and_autoclose(client, user, tok, et, TAKE_PROFIT_PCT, STOP_LOSS_PCT, CLOSE_BUFFER_SECONDS)
            return
        else:
            # state says open but it isn't -> clear it
            print("🧹 State was stale (no position found). Clearing state.", flush=True)
            clear_state()

    # 2) If ANY BTC 5m position is already open, monitor it (prevents 2 positions)
    open_positions = find_any_open_btc5m_positions(user)
    if open_positions:
        p = open_positions[0]
        tok = str(p.get("asset"))
        pslug = p.get("slug") or "(unknown)"
        print(f"🧠 Existing BTC 5m position found (slug={pslug}). Monitoring it (NO new orders).", flush=True)
        # try to parse endDate, else infer from slug
        et = None
        try:
            end_str = p.get("endDate")
            if end_str:
                # many endpoints return ISO strings; handle 'Z'
                end_str = end_str.replace("Z", "+00:00")
                et = datetime.fromisoformat(end_str).astimezone(timezone.utc)
        except Exception:
            et = None
        if et is None:
            try:
                ts = int((p.get("slug") or "").split("-")[-1])
                et = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(minutes=5)
            except Exception:
                et = end_time

        save_state({"token_id": tok, "end_ts": int(et.timestamp()), "resumed_from": "data_api"})
        monitor_and_autoclose(client, user, tok, et, TAKE_PROFIT_PCT, STOP_LOSS_PCT, CLOSE_BUFFER_SECONDS)
        return

    # 3) Find current market + compute signal
    tokens = get_market_tokens(slug)
    if not tokens:
        print(f"⚠️ No market found for slug={slug}. Exiting.", flush=True)
        return

    print(f"✅ Market found: {slug}", flush=True)
    print(f"✅ Tokens: YES={tokens['yes'][:10]}… | NO={tokens['no'][:10]}…", flush=True)

    mom = compute_momentum_pct(LOOKBACK_MINS)
    if mom is None:
        print("⚠️ Could not compute momentum. Exiting.", flush=True)
        return

    print(f"📈 Momentum ({LOOKBACK_MINS}m): {mom:+.3f}% | threshold={MIN_MOMENTUM_PCT:.3f}%", flush=True)
    if abs(mom) < MIN_MOMENTUM_PCT:
        print("😴 No signal. Exiting.", flush=True)
        return

    token_id = tokens["yes"] if mom > 0 else tokens["no"]
    side_name = "YES" if mom > 0 else "NO"
    print(f"📈 SIGNAL: {side_name}", flush=True)

    if not live:
        print("🧪 Dry mode (no trade). Exiting.", flush=True)
        return

    # 4) Place buy and immediately switch to monitoring using Data API truth
    try:
        oid, entry_price, size = place_buy(client, token_id, MAX_BET_SIZE)
        save_state({
            "order_id": oid,
            "token_id": token_id,
            "side": side_name,
            "end_ts": int(end_time.timestamp()),
            "opened_at": int(time.time()),
        })
        monitor_and_autoclose(client, user, token_id, end_time, TAKE_PROFIT_PCT, STOP_LOSS_PCT, CLOSE_BUFFER_SECONDS)
    except Exception as e:
        print(f"❌ Trade Failed: {e}", flush=True)
        # If order actually went through despite the exception, the next run will detect the open position via Data API.
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    args = parser.parse_args()
    run(args.live)
