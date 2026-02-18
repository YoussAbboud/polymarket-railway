#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot — signature fix for proxy/safe wallets.

What changed (vs your pristine file):
- ✅ Uses PartialCreateOrderOptions (not dict) to avoid: 'dict' object has no attribute 'tick_size'
- ✅ Workaround for POST /order invalid signature on proxy wallets:
     For signature_type 1/2 we send L2 header POLY_ADDRESS as the FUNDER (proxy wallet),
     not the signer address. (Matches known client-side issue pattern.) :contentReference[oaicite:2]{index=2}
- ✅ More explicit logging so it doesn't look like it "exited without doing anything"
"""

import os, sys, json, time, argparse, requests
from datetime import datetime, timezone, timedelta

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import BUY
from py_clob_client.constants import POLYGON

# --- Internal helpers from py_clob_client (for manual post with correct POLY_ADDRESS) ---
from py_clob_client.utilities import order_to_json
from py_clob_client.signing.hmac import build_hmac_signature
from py_clob_client.headers.headers import (
    POLY_ADDRESS, POLY_SIGNATURE, POLY_TIMESTAMP, POLY_API_KEY, POLY_PASSPHRASE
)

# ==============================================================================
# 🚀 STRATEGY SETTINGS
# ==============================================================================
ASSET = "BTC"
LOOKBACK_MINS = 12
MIN_MOMENTUM_PCT = 0.12

# --- SAFETY SETTINGS ---
MAX_BET_SIZE = 5.0
STOP_LOSS_PCT = 0.15
TAKE_PROFIT_PCT = 0.20
CLOSE_BUFFER_SECONDS = 60
STAGNATION_TIMEOUT = 20
# ==============================================================================

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA_URL = "https://gamma-api.polymarket.com/events"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"

def get_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        print(f"❌ ERROR: Missing Env Variable: {key}")
        sys.exit(1)
    return val.strip()

def checksum_lower(addr: str) -> str:
    # Polymarket APIs accept lowercase; EIP712 uses bytes anyway.
    # We just normalize the string.
    a = addr.strip()
    if not a.startswith("0x"):
        a = "0x" + a
    return a

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def floor_to_5m(ts: datetime) -> int:
    return int(ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0).timestamp())

def init_client(signature_type: int, pk: str, funder: str) -> ClobClient:
    signer_addr = None
    try:
        # Instantiate client
        client = ClobClient(
            HOST,
            key=pk,
            chain_id=CHAIN_ID,
            signature_type=signature_type,
            funder=funder,
        )
        signer_addr = client.signer.address()
        # Create/derive API creds
        client.set_api_creds(client.create_or_derive_api_creds())
        print(f"🔐 Initializing Polymarket auth with signature_type={signature_type} ...")
        print(f"✅ AUTH OK (signature_type={signature_type}). Bot is Live.")
    except Exception as e:
        print(f"❌ Initialization Failed (signature_type={signature_type}): {e}")
        raise

    # Wallet check print (matches your style)
    print("🧾 WALLET CHECK")
    print("------------------------------------------------------")
    if signer_addr:
        print(f"🔑 Signer (from PRIVATE_KEY): {signer_addr}")
    print(f"🎯 Funder (proxy / funded wallet): {funder}")
    print("------------------------------------------------------")
    return client

def get_market_tokens(slug: str):
    try:
        r = requests.get(f"{GAMMA_URL}?slug={slug}", timeout=8)
        data = r.json()
        if not data:
            return None
        market = data[0].get("markets", [])[0]
        clob_ids = json.loads(market.get("clobTokenIds"))
        return {"market_id": market["id"], "yes": clob_ids[0], "no": clob_ids[1]}
    except Exception:
        return None

def pick_active_slug(asset: str) -> tuple[str, datetime] | tuple[None, None]:
    """
    Try a few candidate slugs around current 5m boundary (prev/current/next),
    so you don't "exit without doing anything" due to tiny timing drift.
    """
    t = now_utc()
    base = floor_to_5m(t)
    candidates = [base, base - 300, base + 300]

    for ts in candidates:
        slug = f"{asset.lower()}-updown-5m-{ts}"
        tokens = get_market_tokens(slug)
        if tokens:
            end_time = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(minutes=5)
            return slug, end_time

    return None, None

def momentum_lookback_pct(lookback_mins: int) -> float | None:
    """
    Uses Coingecko 1-day chart, finds nearest price to now and now-lookback.
    """
    try:
        r = requests.get(COINGECKO_URL, timeout=8).json()
        prices = r.get("prices", [])
        if len(prices) < 10:
            return None

        # prices: [[ms, price], ...]
        latest_ms, latest_price = prices[-1]
        target_ms = latest_ms - (lookback_mins * 60_000)

        # Find closest point to target_ms
        best = None
        best_dt = 10**18
        for ms, p in reversed(prices):
            dt = abs(ms - target_ms)
            if dt < best_dt:
                best_dt = dt
                best = (ms, p)
            if ms < target_ms and best is not None:
                break

        if not best:
            return None

        _, past_price = best
        if past_price <= 0:
            return None

        return ((latest_price - past_price) / past_price) * 100.0
    except Exception:
        return None

def get_book_params(client: ClobClient, token_id: str) -> tuple[float, str, bool]:
    """
    Returns: best_ask_price, tick_size, neg_risk
    """
    book = client.get_order_book(token_id)
    # asks[0].price is a string
    best_ask = float(book.asks[0].price) if book.asks else 0.99
    tick_size = getattr(book, "tick_size", "0.01") or "0.01"
    neg_risk = bool(getattr(book, "neg_risk", False))
    return best_ask, tick_size, neg_risk

def post_order_fixed_poly_address(client: ClobClient, signed_order, order_type="GTC", post_only=False):
    """
    Manual POST /order using py_clob_client's HMAC util, but with POLY_ADDRESS set properly.

    Workaround: for signature_type 1/2 (proxy/safe), send POLY_ADDRESS = funder address. :contentReference[oaicite:3]{index=3}
    For signature_type 0 (EOA), POLY_ADDRESS = signer address.
    """
    creds = client.creds
    if creds is None:
        raise RuntimeError("Missing L2 creds; did you call set_api_creds()?")

    # Order payload
    body = order_to_json(signed_order, creds.api_key, order_type, post_only)
    serialized = json.dumps(body, separators=(",", ":"), ensure_ascii=False)

    # HMAC signature (L2)
    ts = int(datetime.now().timestamp())
    request_path = "/order"
    hmac_sig = build_hmac_signature(
        creds.api_secret,
        ts,
        "POST",
        request_path,
        serialized
    )

    # Decide POLY_ADDRESS header
    sig_type = getattr(client, "sig_type", None)
    funder = getattr(client, "funder", None)
    signer_addr = client.signer.address()

    poly_addr = signer_addr
    if sig_type in (1, 2) and funder:
        poly_addr = funder  # <- key workaround

    headers = {
        "Content-Type": "application/json",
        POLY_ADDRESS: poly_addr,
        POLY_SIGNATURE: hmac_sig,
        POLY_TIMESTAMP: str(ts),
        POLY_API_KEY: creds.api_key,
        POLY_PASSPHRASE: creds.api_passphrase,
    }

    resp = requests.post(f"{HOST}{request_path}", headers=headers, data=serialized, timeout=15)
    if resp.status_code >= 400:
        try:
            j = resp.json()
        except Exception:
            j = {"error": resp.text}
        raise Exception(f"POST /order failed [{resp.status_code}]: {j}")

    return resp.json()

def place_order_live(client: ClobClient, token_id: str, limit_price: float, shares: float, tick_size: str, neg_risk: bool):
    # Correct options type (fixes dict.tick_size crash)
    opts = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)

    order_args = OrderArgs(
        price=limit_price,
        size=shares,
        side=BUY,
        token_id=token_id
    )

    # Create signed order (EIP712 order signature)
    signed_order = client.create_order(order_args, opts)

    # Post using fixed POLY_ADDRESS workaround
    return post_order_fixed_poly_address(client, signed_order, order_type="GTC", post_only=False)

def monitor_trade(client, token_id, entry_price, end_time):
    """
    Clean monitor:
    - Prints ONE line per second (price + PnL + seconds left).
    - Never throws (so your outer loop won't treat it as "Trade Failed" and retry signatures).
    - Auto-closes with a SELL when TP/SL hit OR buffer time reached.
    - Avoids get_balance_allowance() entirely (your py_clob_client doesn't accept asset_type kwarg).
    """

    from py_clob_client.order_builder.constants import SELL

    # --- local knobs ---
    PRINT_EVERY_S = 1
    CLOSE_CONFIRM_TIMEOUT_S = 20  # how long to wait for SELL to fill
    BALANCE_WAIT_RETRIES = 20     # if SELL says "not enough balance", wait up to ~20s (1s each)
    EPS = 1e-6

    try:
        entry_price = float(entry_price)
    except Exception:
        entry_price = 0.99

    # close buffer time
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)

    # estimate shares same way you sized BUY
    try:
        shares = round(MAX_BET_SIZE / entry_price, 1)
    except Exception:
        shares = round(MAX_BET_SIZE / 0.99, 1)

    # tick size (best effort)
    try:
        tick_size = getattr(client, "get_tick_size", lambda _tid: "0.01")(token_id)
    except Exception:
        tick_size = "0.01"
    try:
        tick = float(tick_size or "0.01")
    except Exception:
        tick = 0.01

    def get_price_mid():
        """Return a usable price or None. Never returns fake 0.5."""
        # midpoint first
        try:
            mid = client.get_midpoint(token_id)
            if mid is not None:
                return float(mid)
        except Exception:
            pass

        # orderbook mid fallback
        try:
            book = client.get_order_book(token_id)
            bids = getattr(book, "bids", None) or []
            asks = getattr(book, "asks", None) or []
            if bids and asks:
                b = float(bids[0].price)
                a = float(asks[0].price)
                return (a + b) / 2.0
            if bids:
                return float(bids[0].price)
            if asks:
                return float(asks[0].price)
        except Exception:
            pass

        return None

    def best_marketable_sell_price():
        """Sell at best bid - 1 tick so it fills quickly."""
        try:
            book = client.get_order_book(token_id)
            bids = getattr(book, "bids", None) or []
            if bids:
                px = float(bids[0].price) - tick
            else:
                px = (get_price_mid() or entry_price) - tick
        except Exception:
            px = entry_price - tick

        px = max(0.01, min(px, 0.99))
        return round(px, 2)

    def try_cancel(order_id):
        """Best effort cancel (only if supported)."""
        for name in ("cancel_order", "cancelOrder", "cancel"):
            fn = getattr(client, name, None)
            if fn:
                try:
                    fn(order_id)
                    return True
                except Exception:
                    pass
        return False

    def get_order(order_id):
        """Best effort order fetch."""
        for name in ("get_order", "getOrder"):
            fn = getattr(client, name, None)
            if fn:
                try:
                    o = fn(order_id)
                    return o if isinstance(o, dict) else None
                except Exception:
                    return None
        return None

    def order_is_filled(o, target_size):
        """Parse different shapes safely."""
        if not isinstance(o, dict):
            return False
        status = (o.get("status") or o.get("state") or "").lower()
        filled = o.get("size_filled") or o.get("filledSize") or o.get("filled_size") or 0.0
        try:
            filled = float(filled)
        except Exception:
            filled = 0.0

        if status == "filled":
            return True
        return filled >= (float(target_size) - 1e-3)

    def close_position(reason):
        """
        Place SELL and confirm fill.
        Handles "not enough balance / allowance" by waiting for the BUY to actually settle.
        Never raises.
        """
        print(f"🧾 CLOSE TRIGGER: {reason}", flush=True)

        # If we reach here extremely fast, the position might not be settled yet.
        # We'll retry SELL if we get the 'not enough balance' error.
        last_err = None
        sell_order_id = None

        for attempt in range(1, 4):  # a few reprices max
            px = best_marketable_sell_price()
            # be more aggressive on later attempts
            if attempt > 1:
                px = max(0.01, round(px - (tick * (attempt - 1)), 2))

            # Wait-retry loop ONLY for "not enough balance"
            for w in range(BALANCE_WAIT_RETRIES):
                try:
                    resp = client.create_and_post_order(
                        OrderArgs(price=float(px), size=float(shares), side=SELL, token_id=token_id)
                    )
                    if isinstance(resp, dict):
                        sell_order_id = resp.get("orderID") or resp.get("orderId") or resp.get("id")
                    print(f"✅ CLOSE ORDER SENT: {sell_order_id or resp}", flush=True)
                    last_err = None
                    break
                except Exception as e:
                    msg = str(e).lower()
                    last_err = e
                    if "not enough balance" in msg or "allowance" in msg:
                        # BUY not settled yet; wait 1s and try again (no spam, just 1 line)
                        print("⏳ Waiting settlement (no balance yet)...", flush=True)
                        time.sleep(1)
                        continue
                    else:
                        # real error, break out
                        break

            if last_err is None:
                break

        if last_err is not None:
            print(f"⚠️ Could not close automatically: {last_err}", flush=True)
            return

        # Confirm fill if we have an order id and get_order is supported
        if not sell_order_id:
            return

        deadline = time.time() + CLOSE_CONFIRM_TIMEOUT_S
        while time.time() < deadline:
            o = get_order(sell_order_id)
            if o and order_is_filled(o, shares):
                print("✅ CLOSE FILLED.", flush=True)
                return
            time.sleep(1)

        # Not filled: try cancel; if canceled we can place a more aggressive close
        if try_cancel(sell_order_id):
            print("🧹 Close not filled in time — canceled, posting more aggressively...", flush=True)
            try:
                px = max(0.01, round(best_marketable_sell_price() - (2 * tick), 2))
                resp = client.create_and_post_order(
                    OrderArgs(price=float(px), size=float(shares), side=SELL, token_id=token_id)
                )
                oid2 = resp.get("orderID") if isinstance(resp, dict) else None
                print(f"✅ CLOSE ORDER SENT (retry): {oid2 or resp}", flush=True)
            except Exception as e:
                print(f"⚠️ Close retry failed: {e}", flush=True)
        else:
            print("⚠️ Close not filled and couldn't cancel. Check position on the website.", flush=True)

    # ---------------- Main monitor loop ----------------
    print("📊 MONITORING... (PnL 1s + AUTO-CLOSE)", flush=True)

    # If we are already in the buffer window, close immediately
    if datetime.now(timezone.utc) >= target_time:
        close_position("BUFFER (immediate)")
        return

    while True:
        try:
            now = datetime.now(timezone.utc)

            # buffer reached => close
            if now >= target_time:
                close_position("BUFFER")
                break

            price = get_price_mid()
            secs_left = int((target_time - now).total_seconds())

            if price is None:
                # print ONE line, do not compute fake pnl
                print(f"⏱️ Price: N/A | PnL: N/A | closes in {secs_left}s", flush=True)
                time.sleep(PRINT_EVERY_S)
                continue

            pnl = (price - entry_price) / entry_price

            # ONE clean line per second
            print(f"⏱️ Price: {price:.3f} | PnL: {pnl*100:+.2f}% | closes in {secs_left}s", flush=True)

            if pnl <= -STOP_LOSS_PCT:
                close_position("STOP LOSS")
                break
            if pnl >= TAKE_PROFIT_PCT:
                close_position("TAKE PROFIT")
                break

            time.sleep(PRINT_EVERY_S)

        except Exception as e:
            # CRITICAL: never throw from monitor (prevents signature retries & "trade failed" spam)
            print(f"⚠️ Monitor error (ignored): {e}", flush=True)
            time.sleep(1)

    print("✅ MONITOR COMPLETE.", flush=True)


def run_strategy(live: bool):
    pk = get_env("PRIVATE_KEY")
    funder = checksum_lower(get_env("POLYGON_ADDRESS"))

    # Let you override signature type explicitly (recommended)
    # 0=EOA, 1=POLY_PROXY (Magic), 2=GNOSIS_SAFE (browser wallet)
    # Docs mapping: :contentReference[oaicite:4]{index=4}
    sig_env = os.environ.get("SIGNATURE_TYPE", "").strip()
    sig_candidates = []
    if sig_env.isdigit():
        sig_candidates = [int(sig_env)]
    else:
        # Try safe first, then proxy, then eoa
        sig_candidates = [2, 1, 0]

    print(f"🕒 Now UTC: {now_utc().isoformat()} | live={live}")

    slug, end_time = pick_active_slug(ASSET)
    if not slug:
        print("❌ No active market found for current/adjacent 5m windows (Gamma returned nothing). Exiting.")
        return

    print(f"✅ Market found: {slug}")
    tokens = get_market_tokens(slug)
    if not tokens:
        print("❌ Market tokens not found. Exiting.")
        return

    print(f"✅ Tokens: YES={str(tokens['yes'])[:10]}… | NO={str(tokens['no'])[:10]}…")

    mom = momentum_lookback_pct(LOOKBACK_MINS)
    if mom is None:
        print("❌ Could not compute momentum (Coingecko). Exiting.")
        return

    print(f"📈 Momentum ({LOOKBACK_MINS}m): {mom:+.3f}% | threshold={MIN_MOMENTUM_PCT:.3f}%")
    if abs(mom) < MIN_MOMENTUM_PCT:
        print("🟡 No signal (below threshold). Exiting.")
        return

    token_to_buy = tokens["yes"] if mom > 0 else tokens["no"]
    side_name = "YES" if mom > 0 else "NO"
    print(f"📈 SIGNAL: {side_name}")

    if not live:
        print("🧪 Dry mode (not live). Exiting.")
        return

    last_err = None
    for sig_type in sig_candidates:
        try:
            client = init_client(sig_type, pk, funder)

            best_ask, tick_size, neg_risk = get_book_params(client, token_to_buy)
            print(f"🧩 Market params: tick_size={tick_size} | neg_risk={neg_risk}")

            # price = best ask + 1 tick
            tick = float(tick_size)
            limit_price = min(round(best_ask + tick, 2), 1.0 - tick)
            shares = round(MAX_BET_SIZE / limit_price, 1)

            print(f"🚀 BUYING: {shares} shares @ {limit_price:.2f}...")

            resp = place_order_live(
                client,
                token_to_buy,
                limit_price,
                shares,
                tick_size=tick_size,
                neg_risk=neg_risk
            )

            order_id = resp.get("orderID") or resp.get("orderId") or resp.get("id")
            if order_id:
                print(f"✅ ORDER PLACED: {order_id}")
                monitor_trade(client, token_to_buy, limit_price, end_time)
                return

            print(f"✅ Order response: {resp}")
            return

        except Exception as e:
            last_err = e
            msg = str(e)
            print(f"❌ Trade Failed (signature_type={sig_type}): {msg}")

            # If this is invalid signature, try next candidate (or your explicit SIGNATURE_TYPE)
            continue

    print("------------------------------------------------------")
    print("❌ All signature types failed to place order.")
    print(f"Last error: {last_err}")
    print("Quick sanity checks (from official docs):")
    print("- POLYGON_ADDRESS must be your Polymarket 'proxy wallet' (wallet address shown in settings).")
    print("- SIGNATURE_TYPE must match how you created/logged into your Polymarket account:")
    print("    1 = Magic/Email login, 2 = Browser wallet login, 0 = EOA wallet holding funds.")
    print("------------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    args = parser.parse_args()
    run_strategy(args.live)
