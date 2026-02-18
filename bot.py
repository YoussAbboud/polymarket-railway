#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (stable v12.0)

Fixes:
- Single signature_type (default 1). No more "failed then still placed" confusion.
- Correct best bid/ask selection (max bid, min ask) + price normalization.
- Monitoring uses REAL filled size + avg fill price (no fake 0.50 / 0.01 panic).
- Auto-close on TP/SL/buffer with retry + reprice.
- Lock + persistent state: prevents opening multiple positions across restarts.

ENV:
REQUIRED:
- PRIVATE_KEY
- POLYGON_ADDRESS   (proxy/funder wallet from Polymarket settings)

OPTIONAL:
- SIGNATURE_TYPE    (1 or 2). Default: 1
- BOT_STATE_DIR     (where to store state/lock). Default: autodetect (/data, /app/data, .)
- COOLDOWN_SECONDS  (min seconds between NEW trades). Default: 120
"""

import os, sys, json, time, argparse, requests
from datetime import datetime, timezone, timedelta

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

# ==============================================================================
# STRATEGY SETTINGS
# ==============================================================================
ASSET = "BTC"
LOOKBACK_MINS = 12
MIN_MOMENTUM_PCT = 0.12

MAX_BET_SIZE = 5.0
STOP_LOSS_PCT = 0.15
TAKE_PROFIT_PCT = 0.20
CLOSE_BUFFER_SECONDS = 60

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA_URL = "https://gamma-api.polymarket.com/events"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"


# ==============================================================================
# Small utilities
# ==============================================================================
def log(msg: str):
    print(msg, flush=True)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def get_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        log(f"❌ ERROR: Missing Env Variable: {key}")
        sys.exit(1)
    return val.strip()

def checksum_lower(addr: str) -> str:
    a = addr.strip()
    if not a.startswith("0x"):
        a = "0x" + a
    return a.lower()

def floor_to_5m(ts: datetime) -> int:
    return int(ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0).timestamp())

def parse_slug_end_time(slug: str) -> datetime:
    ts = int(slug.split("-")[-1])
    return datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(minutes=5)

def normalize_price(p: float | None) -> float | None:
    """
    Some clients return 0-1, some can return cents (0-100).
    Normalize to 0-1 if needed.
    """
    if p is None:
        return None
    try:
        p = float(p)
    except Exception:
        return None
    if p > 1.0 and p <= 100.0:
        return p / 100.0
    return p

def pick_state_dir() -> str:
    candidates = []
    env_dir = os.getenv("BOT_STATE_DIR", "").strip()
    if env_dir:
        candidates.append(env_dir)
    candidates += ["/data", "/app/data", "."]
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            test_path = os.path.join(d, ".write_test")
            with open(test_path, "w") as f:
                f.write("ok")
            os.remove(test_path)
            return d
        except Exception:
            continue
    return "."

STATE_DIR = pick_state_dir()
STATE_PATH = os.path.join(STATE_DIR, "bot_state.json")
LOCK_PATH = os.path.join(STATE_DIR, "bot.lock")

COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "120"))


# ==============================================================================
# Lock + state (prevents double positions across restarts)
# ==============================================================================
def acquire_lock():
    import fcntl
    f = open(LOCK_PATH, "w")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f

def load_state() -> dict:
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(st: dict):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE_PATH)

def clear_open_trade(st: dict):
    st["open_trade"] = None
    save_state(st)


# ==============================================================================
# Client init
# ==============================================================================
def init_client() -> ClobClient:
    pk = get_env("PRIVATE_KEY")
    funder = checksum_lower(get_env("POLYGON_ADDRESS"))

    sig = int(os.getenv("SIGNATURE_TYPE", "1"))
    if sig not in (1, 2):
        sig = 1

    log(f"🔐 Initializing Polymarket auth with signature_type={sig} ...")
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=sig, funder=funder)
    client.set_api_creds(client.create_or_derive_api_creds())

    log("✅ AUTH OK. Bot is Live.")
    log("🧾 WALLET CHECK")
    log("------------------------------------------------------")
    try:
        log(f"🔑 Signer (from PRIVATE_KEY): {client.signer.address()}")
    except Exception:
        pass
    log(f"🎯 Funder (proxy / funded wallet): {funder}")
    log("------------------------------------------------------")
    return client


# ==============================================================================
# Market discovery
# ==============================================================================
def get_market_tokens(slug: str):
    try:
        r = requests.get(f"{GAMMA_URL}?slug={slug}", timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        market = data[0].get("markets", [])[0]
        raw = market.get("clobTokenIds")
        clob_ids = json.loads(raw) if isinstance(raw, str) else raw
        if not clob_ids or len(clob_ids) < 2:
            return None
        return {"yes": str(clob_ids[0]), "no": str(clob_ids[1])}
    except Exception:
        return None

def find_current_market(asset: str):
    base_ts = floor_to_5m(now_utc())
    candidates = [base_ts, base_ts - 300, base_ts + 300, base_ts - 600, base_ts + 600]
    for ts in candidates:
        slug = f"{asset.lower()}-updown-5m-{ts}"
        tokens = get_market_tokens(slug)
        if tokens:
            return slug, tokens
    return None, None


# ==============================================================================
# Pricing helpers (IMPORTANT: correct best bid/ask)
# ==============================================================================
def get_best_bid_ask(client: ClobClient, token_id: str):
    """
    Robust against any sorting:
    - best bid = max(bids)
    - best ask = min(asks)
    """
    try:
        ob = client.get_order_book(token_id)
        bids = getattr(ob, "bids", None) or []
        asks = getattr(ob, "asks", None) or []
        bid_prices = [normalize_price(b.price) for b in bids]
        ask_prices = [normalize_price(a.price) for a in asks]
        bid_prices = [p for p in bid_prices if p is not None]
        ask_prices = [p for p in ask_prices if p is not None]
        best_bid = max(bid_prices) if bid_prices else None
        best_ask = min(ask_prices) if ask_prices else None
        return best_bid, best_ask
    except Exception:
        return None, None

def get_mark_price(client: ClobClient, token_id: str):
    """
    Use best bid (realistic exit) if available, else midpoint from bid/ask.
    Never uses fake 0.50.
    """
    bid, ask = get_best_bid_ask(client, token_id)
    if bid is not None:
        return bid
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    # fallback to midpoint only if not exactly 0.5
    try:
        mid = client.get_midpoint(token_id)
        mid = normalize_price(mid)
        if mid is None:
            return None
        if abs(mid - 0.5) < 1e-9:
            return None
        return mid
    except Exception:
        return None


# ==============================================================================
# Signal
# ==============================================================================
def compute_momentum_pct(lookback_mins: int) -> float | None:
    try:
        r = requests.get(COINGECKO_URL, timeout=12).json()
        prices = r.get("prices", [])
        if not prices or len(prices) < 10:
            return None

        latest_ms, latest_price = prices[-1]
        target_ms = latest_ms - (lookback_mins * 60_000)

        best_p = None
        best_dt = 10**18
        for ms, p in reversed(prices):
            dt = abs(ms - target_ms)
            if dt < best_dt:
                best_dt, best_p = dt, p
            if ms < target_ms and best_p is not None:
                break

        if best_p is None or best_p <= 0:
            return None
        return float(((latest_price - best_p) / best_p) * 100.0)
    except Exception:
        return None


# ==============================================================================
# Order + monitoring (AUTO-CLOSE that actually works)
# ==============================================================================
def get_order_safe(client: ClobClient, order_id: str):
    fn = getattr(client, "get_order", None)
    if not fn:
        return None
    try:
        o = fn(order_id)
        return o if isinstance(o, dict) else None
    except Exception:
        return None

def parse_filled_avg(o: dict):
    status = (o.get("status") or o.get("state") or "").lower()
    filled = o.get("size_filled") or o.get("filledSize") or o.get("filled_size") or 0.0
    avg = o.get("avg_fill_price") or o.get("averageFillPrice") or o.get("avgFillPrice")
    try:
        filled = float(filled)
    except Exception:
        filled = 0.0
    try:
        avg = float(avg) if avg is not None else None
    except Exception:
        avg = None
    return status, filled, avg

def cancel_order_safe(client: ClobClient, order_id: str):
    for name in ("cancel_order", "cancelOrder", "cancel"):
        fn = getattr(client, name, None)
        if fn:
            try:
                fn(order_id)
                return True
            except Exception:
                pass
    return False

def monitor_and_autoclose(client: ClobClient, slug: str, token_id: str, buy_order_id: str, end_time: datetime, fallback_entry: float):
    """
    - Waits for BUY fill (so we know real size + avg price)
    - Prints once per second
    - Closes on TP/SL/buffer using real filled size
    - Retries close by repricing lower if not filled
    - NEVER raises (so it won't trigger "trade failed" and re-open)
    """
    try:
        target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
        log("📊 MONITORING... (PnL 1s + AUTO-CLOSE)")

        # 1) Wait for fill (or until buffer)
        filled_size = 0.0
        entry = float(fallback_entry)

        while now_utc() < target_time:
            o = get_order_safe(client, buy_order_id)
            if o:
                status, filled, avg = parse_filled_avg(o)
                if filled > 0:
                    filled_size = filled
                    if avg is not None:
                        entry = normalize_price(avg) or entry
                    log(f"✅ BUY FILLED: {filled_size:.4f} @ {entry:.3f} (status={status})")
                    break
            time.sleep(1)

        if filled_size <= 0:
            # not filled -> cancel buy and exit; no position should exist
            cancel_order_safe(client, buy_order_id)
            log("⚠️ Buy not filled before buffer. Cancelled buy order (best effort).")
            return False

        # 2) Live monitoring until TP/SL/buffer
        def should_close(pnl: float, now_dt: datetime):
            if pnl <= -STOP_LOSS_PCT:
                return "STOP LOSS"
            if pnl >= TAKE_PROFIT_PCT:
                return "TAKE PROFIT"
            if now_dt >= target_time:
                return "BUFFER"
            return None

        def place_close_sell(size_to_sell: float, reason: str) -> bool:
            # Try multiple reprices to force fill
            for attempt in range(1, 6):
                bid, ask = get_best_bid_ask(client, token_id)
                # Marketable sell: <= best bid
                if bid is None:
                    px = 0.01
                else:
                    px = bid - (0.01 * (attempt - 1))
                px = max(0.01, min(px, 0.99))
                px = round(px, 2)

                try:
                    log(f"🧾 CLOSING ({reason}) -> SELL {size_to_sell:.4f} @ {px:.2f} (attempt {attempt})")
                    resp = client.create_and_post_order(
                        OrderArgs(price=float(px), size=float(size_to_sell), side=SELL, token_id=token_id)
                    )
                    sell_id = resp.get("orderID") if isinstance(resp, dict) else None
                    log(f"✅ CLOSE ORDER SENT: {sell_id or resp}")

                    # Confirm fill (best effort)
                    if sell_id:
                        deadline = time.time() + 20
                        while time.time() < deadline:
                            o = get_order_safe(client, sell_id)
                            if o:
                                st, f, _ = parse_filled_avg(o)
                                if st == "filled" or f >= size_to_sell - 1e-6:
                                    log("✅ CLOSE FILLED.")
                                    return True
                            time.sleep(1)

                    # If no order id or not confirmed, try cancel and reprice
                    if sell_id:
                        cancel_order_safe(client, sell_id)
                except Exception as e:
                    msg = str(e).lower()
                    # if not enough balance, it usually means partial fill/race; wait and retry
                    if "not enough balance" in msg or "allowance" in msg:
                        time.sleep(1)
                        continue
                    log(f"⚠️ Close failed: {e}")
                    return False

            log("⚠️ Could not confirm auto-close fill after retries.")
            return False

        while True:
            now_dt = now_utc()

            mp = get_mark_price(client, token_id)
            if mp is None:
                secs_left = max(0, int((target_time - now_dt).total_seconds()))
                log(f"⏱️ Price: N/A | PnL: N/A | closes in {secs_left}s")
                if now_dt >= target_time:
                    # buffer -> close anyway
                    ok = place_close_sell(filled_size, "BUFFER")
                    return ok
                time.sleep(1)
                continue

            pnl = (mp - entry) / entry
            secs_left = max(0, int((target_time - now_dt).total_seconds()))
            log(f"⏱️ Price: {mp:.3f} | PnL: {pnl*100:+.2f}% | closes in {secs_left}s")

            reason = should_close(pnl, now_dt)
            if reason:
                ok = place_close_sell(filled_size, reason)
                return ok

            time.sleep(1)

    except Exception as e:
        log(f"⚠️ Monitor crashed but ignored: {e}")
        return False


# ==============================================================================
# Main
# ==============================================================================
def run(live: bool):
    with acquire_lock():
        st = load_state()
        now_ts = time.time()

        # If we have an open trade in state, resume/close it and DO NOT open another
        open_trade = st.get("open_trade")
        if open_trade:
            try:
                end_iso = open_trade.get("end_time")
                end_time = datetime.fromisoformat(end_iso) if end_iso else None
            except Exception:
                end_time = None

            if end_time and now_utc() <= end_time + timedelta(minutes=10):
                log("🧠 Found open trade in state. Resuming monitoring/auto-close (no new orders).")
                client = init_client()
                ok = monitor_and_autoclose(
                    client=client,
                    slug=open_trade["slug"],
                    token_id=open_trade["token_id"],
                    buy_order_id=open_trade["buy_order_id"],
                    end_time=end_time,
                    fallback_entry=float(open_trade.get("fallback_entry", 0.5)),
                )
                if ok:
                    clear_open_trade(st)
                return
            else:
                # stale state
                st["open_trade"] = None
                save_state(st)

        # cooldown to stop back-to-back opens on restarts
        last_new = st.get("last_new_trade_ts", 0)
        if now_ts - last_new < COOLDOWN_SECONDS:
            log(f"🛑 Cooldown active ({COOLDOWN_SECONDS}s). Skipping new trade.")
            return

        log(f"🕒 Now UTC: {now_utc().isoformat()} | live={live}")

        slug, tokens = find_current_market(ASSET)
        if not slug or not tokens:
            log("❌ No market found.")
            return

        log(f"✅ Market found: {slug}")
        log(f"✅ Tokens: YES={tokens['yes'][:10]}… | NO={tokens['no'][:10]}…")

        momentum = compute_momentum_pct(LOOKBACK_MINS)
        if momentum is None:
            log("❌ Could not compute momentum.")
            return

        log(f"📈 Momentum (12m): {momentum:+.3f}% | threshold={MIN_MOMENTUM_PCT:.3f}%")

        if abs(momentum) < MIN_MOMENTUM_PCT:
            log("🟡 No trade: momentum below threshold.")
            return

        token_to_buy = tokens["yes"] if momentum > 0 else tokens["no"]
        side_name = "YES" if momentum > 0 else "NO"
        log(f"📈 SIGNAL: {side_name}")

        if not live:
            log("🟡 Dry mode. Exiting.")
            return

        client = init_client()

        end_time = parse_slug_end_time(slug)

        # Determine a sane limit price from best ask (robust sorting + normalization)
        bid, ask = get_best_bid_ask(client, token_to_buy)
        if ask is None:
            log("❌ No ask liquidity. Skipping.")
            return

        # Slightly above best ask to get filled, but never above 0.99
        limit_price = min(ask + 0.01, 0.99)
        limit_price = max(0.01, round(limit_price, 2))

        shares = round(MAX_BET_SIZE / limit_price, 1)

        log(f"🚀 BUYING: {shares} shares @ {limit_price:.2f}...")

        resp = client.create_and_post_order(
            OrderArgs(price=float(limit_price), size=float(shares), side=BUY, token_id=token_to_buy)
        )
        order_id = resp.get("orderID") if isinstance(resp, dict) else None
        if not order_id:
            log(f"❌ Buy response missing orderID: {resp}")
            return

        log(f"✅ ORDER PLACED: {order_id}")

        # Save state so restarts won't open another position
        st["open_trade"] = {
            "slug": slug,
            "token_id": token_to_buy,
            "buy_order_id": order_id,
            "end_time": end_time.isoformat(),
            "fallback_entry": float(limit_price),
        }
        st["last_new_trade_ts"] = now_ts
        save_state(st)

        ok = monitor_and_autoclose(client, slug, token_to_buy, order_id, end_time, fallback_entry=limit_price)
        if ok:
            clear_open_trade(st)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    args = parser.parse_args()
    run(args.live)
