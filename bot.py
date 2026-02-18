#!/usr/bin/env python3
import os, sys, json, time, argparse, requests
from datetime import datetime, timezone, timedelta

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

# =========================
# STRATEGY SETTINGS
# =========================
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

# =========================
# PERSISTENCE (prevents double opens on restart)
# =========================
STATE_DIR = os.getenv("BOT_STATE_DIR", "/data").strip() or "/data"
os.makedirs(STATE_DIR, exist_ok=True)
STATE_PATH = os.path.join(STATE_DIR, "bot_state.json")
LOCK_PATH = os.path.join(STATE_DIR, "bot.lock")


def log(msg: str):
    print(msg, flush=True)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_env(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        log(f"❌ Missing env: {key}")
        sys.exit(1)
    return v.strip()


def checksum_lower(addr: str) -> str:
    a = addr.strip()
    if not a.startswith("0x"):
        a = "0x" + a
    return a.lower()


def acquire_lock():
    # Linux-only, perfect for Railway
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


# =========================
# AUTH FIRST (always)
# =========================
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


# =========================
# MARKET DISCOVERY
# =========================
def floor_to_5m(ts: datetime) -> int:
    return int(ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0).timestamp())


def parse_slug_end_time(slug: str) -> datetime:
    ts = int(slug.split("-")[-1])
    return datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(minutes=5)


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


# =========================
# PRICE HELPERS (fix wrong 0.01/0.50)
# =========================
def normalize_price(p):
    if p is None:
        return None
    try:
        p = float(p)
    except Exception:
        return None
    if 1.0 < p <= 100.0:
        return p / 100.0
    return p


def best_bid_ask(client: ClobClient, token_id: str):
    """
    Robust even if the lib returns bids unsorted:
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
        return (max(bid_prices) if bid_prices else None), (min(ask_prices) if ask_prices else None)
    except Exception:
        return None, None


def mark_price(client: ClobClient, token_id: str):
    """
    Use best bid as realistic exit.
    Never use fake 0.50; if midpoint is exactly 0.50, treat as unavailable.
    """
    bid, ask = best_bid_ask(client, token_id)
    if bid is not None:
        return bid
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    try:
        mid = normalize_price(client.get_midpoint(token_id))
        if mid is None or abs(mid - 0.5) < 1e-9:
            return None
        return mid
    except Exception:
        return None


# =========================
# SIGNAL
# =========================
def compute_momentum_pct(lookback_mins: int) -> float | None:
    try:
        r = requests.get(COINGECKO_URL, timeout=12).json()
        prices = r.get("prices", [])
        if not prices or len(prices) < 10:
            return None

        latest_ms, latest_price = prices[-1]
        target_ms = latest_ms - (lookback_mins * 60_000)

        best_p, best_dt = None, 10**18
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


# =========================
# ORDER STATUS
# =========================
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
    avg = normalize_price(avg)
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


# =========================
# MONITOR + AUTO-CLOSE (real filled size)
# =========================
def monitor_and_autoclose(client, slug, token_id, buy_order_id, end_time, fallback_entry: float):
    """
    Fixes:
    - Heartbeat prints while waiting for BUY fill (so it never looks frozen).
    - Detects fill via BOTH order status AND conditional token balance.
    - Uses CURRENT conditional balance for SELL size (prevents 'not enough balance/allowance').
    - Prints exactly 1 line/sec in monitor phase.
    - Never raises.
    """
    try:
        target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
        log("📊 MONITORING... (PnL 1s + AUTO-CLOSE)")

        # ---- helpers ----
        def _get_order():
            return get_order_safe(client, buy_order_id)

        def _parse_order(o):
            if not isinstance(o, dict):
                return ("unknown", 0.0, None)
            st, filled, avg = parse_filled_avg(o)
            return (st or "unknown", float(filled or 0.0), avg)

        def _get_conditional_balance() -> float:
            # IMPORTANT: your client wants a SINGLE dict arg, not kwargs
            params = {"asset_type": "CONDITIONAL", "token_id": str(token_id)}

            # refresh (best effort)
            for nm in ("update_balance_allowance", "updateBalanceAllowance"):
                fn = getattr(client, nm, None)
                if fn:
                    try:
                        fn(params)
                    except Exception:
                        pass

            for nm in ("get_balance_allowance", "getBalanceAllowance"):
                fn = getattr(client, nm, None)
                if fn:
                    try:
                        r = fn(params)
                        if isinstance(r, dict):
                            return float(r.get("balance") or 0.0)
                    except Exception:
                        pass
            return 0.0

        def _best_bid_ask():
            return best_bid_ask(client, token_id)

        def _mark_price():
            return mark_price(client, token_id)

        def _round_size(x: float) -> float:
            # keep it safe: 0.1 precision like your BUY sizing
            try:
                return max(0.0, round(float(x), 1))
            except Exception:
                return 0.0

        # ---- 1) wait for fill (with heartbeat) ----
        entry = float(fallback_entry)
        filled_size = 0.0

        # hard max wait for fill; after that we cancel the BUY (best effort) and exit
        FILL_MAX_WAIT_S = 25
        fill_deadline = time.time() + FILL_MAX_WAIT_S

        while now_utc() < target_time:
            o = _get_order()
            st, filled, avg = _parse_order(o)
            if avg is not None:
                entry = float(avg)

            bal = _get_conditional_balance()

            # If either reports a position, treat as filled.
            if filled > 0:
                filled_size = filled
                log(f"✅ BUY FILLED (order): {filled_size:.4f} @ {entry:.3f} (status={st})")
                break
            if bal > 0:
                filled_size = bal
                log(f"✅ BUY FILLED (balance): {filled_size:.4f} @ {entry:.3f} (status={st})")
                break

            bid, ask = _best_bid_ask()
            secs_left = max(0, int((target_time - now_utc()).total_seconds()))
            log(
                f"⏳ Waiting fill... status={st} filled={filled:.4f} bal={bal:.4f} "
                f"| bid={bid if bid is not None else 'N/A'} ask={ask if ask is not None else 'N/A'} "
                f"| closes in {secs_left}s",
            )

            if time.time() >= fill_deadline:
                cancel_order_safe(client, buy_order_id)
                log("⚠️ BUY did not fill fast enough. Canceled (best effort) and exiting to avoid ghost positions.")
                return False

            time.sleep(1)

        if filled_size <= 0:
            # buffer hit and still no balance
            cancel_order_safe(client, buy_order_id)
            log("⚠️ No filled position detected before buffer. Canceled buy (best effort).")
            return False

        # ---- 2) close helper (uses CURRENT balance, not stale filled_size) ----
        def place_close(reason: str) -> bool:
            for attempt in range(1, 12):
                bal = _get_conditional_balance()
                size = _round_size(bal)

                if size <= 0:
                    log("✅ No conditional balance -> position already closed.")
                    return True

                bid, _ = _best_bid_ask()
                # marketable sell: <= best bid; if bid missing, go low to get filled
                if bid is None:
                    px = 0.01
                else:
                    px = bid - 0.01 * (attempt - 1)

                px = max(0.01, min(px, 0.99))
                px = round(px, 2)

                try:
                    log(f"🧾 CLOSING ({reason}) -> SELL {size:.1f} @ {px:.2f} (attempt {attempt})")
                    resp = client.create_and_post_order(
                        OrderArgs(price=float(px), size=float(size), side=SELL, token_id=token_id)
                    )
                    sell_id = resp.get("orderID") if isinstance(resp, dict) else None
                    log(f"✅ CLOSE ORDER SENT: {sell_id or resp}")

                    # confirm fill OR balance drops to ~0
                    deadline = time.time() + 25
                    while time.time() < deadline:
                        bal2 = _get_conditional_balance()
                        if bal2 <= 1e-9:
                            log("✅ Balance is now ~0 (closed).")
                            return True
                        time.sleep(1)

                    # not confirmed -> cancel and retry more aggressive
                    if sell_id:
                        cancel_order_safe(client, sell_id)

                except Exception as e:
                    msg = str(e).lower()
                    if "not enough balance" in msg or "allowance" in msg:
                        time.sleep(1)
                        continue
                    log(f"⚠️ Close failed: {e}")
                    return False

            log("⚠️ Could not confirm close after retries.")
            return False

        # ---- 3) monitor loop (1 line/sec) ----
        while True:
            now_dt = now_utc()
            secs_left = max(0, int((target_time - now_dt).total_seconds()))

            mp = _mark_price()
            if mp is None:
                log(f"⏱️ Price: N/A | PnL: N/A | closes in {secs_left}s")
            else:
                pnl = (mp - entry) / entry
                log(f"⏱️ Price: {mp:.3f} | PnL: {pnl*100:+.2f}% | closes in {secs_left}s")

                if pnl <= -STOP_LOSS_PCT:
                    return place_close("STOP LOSS")
                if pnl >= TAKE_PROFIT_PCT:
                    return place_close("TAKE PROFIT")

            if now_dt >= target_time:
                return place_close("BUFFER")

            time.sleep(1)

    except Exception as e:
        log(f"⚠️ Monitor error ignored: {e}")
        return False


# =========================
# MAIN (resumes on restart)
# =========================
def run(live: bool):
    with acquire_lock():
        log(f"📁 STATE_DIR = {STATE_DIR}")
        st = load_state()

        # Auth FIRST
        client = init_client()

        # Resume if open trade exists
        open_trade = st.get("open_trade")
        if open_trade:
            try:
                end_time = datetime.fromisoformat(open_trade["end_time"])
            except Exception:
                end_time = None

            if end_time:
                log("🧠 Open trade found in state. Resuming monitoring (NO new orders).")
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
                clear_open_trade(st)

        log(f"🕒 Now UTC: {now_utc().isoformat()} | live={live}")

        slug, tokens = find_current_market(ASSET)
        if not slug:
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
            log("🟡 Dry mode, not placing order.")
            return

        end_time = parse_slug_end_time(slug)

        bid, ask = best_bid_ask(client, token_to_buy)
        if ask is None:
            log("❌ No ask liquidity. Skipping.")
            return

        limit_price = min(ask + 0.01, 0.99)
        limit_price = max(0.01, round(limit_price, 2))

        shares = round(MAX_BET_SIZE / limit_price, 1)
        log(f"🚀 BUYING: {shares} shares @ {limit_price:.2f}...")

        resp = client.create_and_post_order(OrderArgs(price=float(limit_price), size=float(shares), side=BUY, token_id=token_to_buy))
        order_id = resp.get("orderID") if isinstance(resp, dict) else None
        if not order_id:
            log(f"❌ Buy response missing orderID: {resp}")
            return

        log(f"✅ ORDER PLACED: {order_id}")

        # Save state immediately so restart won't open again
        st["open_trade"] = {
            "slug": slug,
            "token_id": token_to_buy,
            "buy_order_id": order_id,
            "end_time": end_time.isoformat(),
            "fallback_entry": float(limit_price),
        }
        save_state(st)

        ok = monitor_and_autoclose(client, slug, token_to_buy, order_id, end_time, fallback_entry=limit_price)
        if ok:
            clear_open_trade(st)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", "-l", action="store_true")
    args = ap.parse_args()
    run(args.live)
