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
    Prints live PnL every 1s and AUTO-CLOSES when:
      - TP hit
      - SL hit
      - buffer time hit (end_time - CLOSE_BUFFER_SECONDS)

    Also CONFIRMS the SELL filled (via get_order). If not filled, cancels/reprices a few times.
    Critically: will NOT trigger TP/SL until it can read a real price AND it detects you actually hold the token.
    """

    from py_clob_client.order_builder.constants import SELL  # local import, no top-level edits

    # --- knobs (only inside this function) ---
    PRINT_EVERY_S = 1
    FILL_WAIT_S = 25                 # wait for position balance to appear
    CLOSE_CONFIRM_TIMEOUT_S = 20     # wait for close order to fill
    CLOSE_RETRIES = 3                # reprice attempts
    EPS = 1e-6

    entry_price = float(entry_price)
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)

    def _call(method_names, *args, **kwargs):
        for name in method_names:
            fn = getattr(client, name, None)
            if fn:
                return fn(*args, **kwargs)
        raise AttributeError(f"None of methods exist: {method_names}")

    def _get_conditional_balance():
        """
        Best-effort: update + fetch conditional token balance for token_id.
        Uses L2 methods: updateBalanceAllowance/getBalanceAllowance (python: snake_case usually).
        """
        params = {"asset_type": "CONDITIONAL", "token_id": token_id}

        # update cached balance (best effort)
        try:
            _call(("update_balance_allowance", "updateBalanceAllowance"), params)
        except Exception:
            try:
                _call(("update_balance_allowance", "updateBalanceAllowance"), **params)
            except Exception:
                pass

        # get balance/allowance
        try:
            resp = _call(("get_balance_allowance", "getBalanceAllowance"), params)
        except Exception:
            resp = _call(("get_balance_allowance", "getBalanceAllowance"), **params)

        if isinstance(resp, dict):
            try:
                return float(resp.get("balance") or 0.0)
            except Exception:
                return 0.0
        return 0.0

    def _get_price_mid():
        """
        Use midpoint; if missing, compute from orderbook bid/ask.
        Returns None if still unavailable.
        """
        # 1) midpoint
        try:
            mid = client.get_midpoint(token_id)
            if mid is not None:
                return float(mid)
        except Exception:
            pass

        # 2) order book mid
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

    def _best_bid():
        try:
            book = client.get_order_book(token_id)
            bids = getattr(book, "bids", None) or []
            if bids:
                return float(bids[0].price)
        except Exception:
            pass
        return None

    def _wait_for_position_balance(deadline_dt):
        """
        Wait until conditional token balance > 0 (meaning you actually hold shares),
        or until deadline/buffer time. Avoids 'not enough balance/allowance' when closing.
        """
        start = time.time()
        while True:
            now = datetime.now(timezone.utc)
            if now >= deadline_dt:
                return 0.0
            bal = _get_conditional_balance()
            if bal > EPS:
                return bal
            if time.time() - start > FILL_WAIT_S:
                return 0.0
            print(f"⏳ Waiting fill... conditional balance=0 | closes in {int((deadline_dt-now).total_seconds())}s", flush=True)
            time.sleep(1)

    def _get_order(order_id):
        try:
            o = _call(("get_order", "getOrder"), order_id)
            return o if isinstance(o, dict) else None
        except Exception:
            return None

    def _parse_filled_status(o: dict):
        status = (o.get("status") or o.get("state") or "").lower()
        filled = o.get("size_filled") or o.get("filledSize") or o.get("filled_size") or 0.0
        try:
            filled = float(filled)
        except Exception:
            filled = 0.0
        return filled, status

    def _try_cancel(order_id):
        for name in ("cancel_order", "cancelOrder", "cancel", "cancel_orders", "cancelOrders"):
            fn = getattr(client, name, None)
            if fn:
                try:
                    fn(order_id)
                    return True
                except Exception:
                    pass
        return False

    def _post_close(size_to_sell, reason, aggression_ticks):
        """
        Posts a marketable SELL (limit) using best bid.
        If no bid, uses current mid minus small amount.
        """
        # we keep it marketable: SELL price <= best bid
        bid = _best_bid()
        px = bid
        if px is None:
            mid = _get_price_mid()
            if mid is None:
                px = max(0.01, min(entry_price, 0.99))
            else:
                px = max(0.01, min(mid, 0.99))

        # nudge slightly lower each retry to get filled faster
        px = max(0.01, min(px - (0.01 * max(0, aggression_ticks - 1)), 0.99))
        px = round(px, 2)

        print(f"🧾 CLOSING ({reason}) -> SELL {size_to_sell:.4f} @ {px:.2f}", flush=True)

        resp = client.create_and_post_order(
            OrderArgs(price=float(px), size=float(size_to_sell), side=SELL, token_id=token_id)
        )

        oid = None
        if isinstance(resp, dict):
            oid = resp.get("orderID") or resp.get("orderId") or resp.get("id")

        if oid:
            print(f"✅ CLOSE ORDER SENT: {oid}", flush=True)
        else:
            print(f"✅ CLOSE ORDER SENT (resp): {resp}", flush=True)

        return oid

    def _close_and_confirm(reason):
        """
        Close whatever balance we currently have, confirm filled, retry if needed.
        Never raises.
        """
        for attempt in range(1, CLOSE_RETRIES + 1):
            bal = _get_conditional_balance()
            if bal <= EPS:
                print("✅ Position balance is 0 — already closed.", flush=True)
                return

            oid = None
            try:
                oid = _post_close(bal, reason if attempt == 1 else f"{reason} (retry {attempt})", aggression_ticks=attempt)
            except Exception as e:
                print(f"❌ Close post failed (attempt {attempt}): {e}", flush=True)
                time.sleep(1)
                continue

            # confirm fill by order status if we have id, else by balance dropping
            deadline = time.time() + CLOSE_CONFIRM_TIMEOUT_S
            last_bal = bal

            while time.time() < deadline:
                # If order id exists, prefer order status
                if oid:
                    o = _get_order(oid)
                    if o:
                        filled, status = _parse_filled_status(o)
                        if status == "filled" or filled >= bal - 1e-6:
                            print(f"✅ Close filled (status={status}, filled={filled:.4f}).", flush=True)
                            return

                # fallback: check balance drop
                new_bal = _get_conditional_balance()
                if new_bal <= EPS:
                    print("✅ Balance now 0 — closed.", flush=True)
                    return
                if new_bal < last_bal - EPS:
                    last_bal = new_bal

                time.sleep(1)

            # not filled in time: try cancel then retry
            if oid and _try_cancel(oid):
                print(f"🧹 Close not filled in time — canceled {oid}. Retrying...", flush=True)
            else:
                print("⚠️ Close not confirmed filled and couldn't cancel. Stopping to avoid duplicate sells.", flush=True)
                return

        print("⚠️ Close retries exhausted. Check position on Polymarket.", flush=True)

    # ---------------------------
    # Main monitoring loop
    # ---------------------------
    print("📊 MONITORING... (AUTO-CLOSE + CONFIRM-FILL enabled)", flush=True)

    # Wait until we actually have a filled position (conditional balance > 0),
    # otherwise we might trigger SL/TP on fake price and/or fail closing.
    pos_bal = _wait_for_position_balance(target_time)
    if pos_bal <= EPS:
        # Either not filled yet or couldn't fetch balance before buffer
        if datetime.now(timezone.utc) >= target_time:
            print("⏰ Buffer reached but no filled position detected. Check open orders on the site.", flush=True)
        else:
            print("⚠️ Could not detect filled position (balance still 0). Check the site for fills/open orders.", flush=True)
        return

    # live PnL prints every 1s
    while True:
        now = datetime.now(timezone.utc)
        if now >= target_time:
            _close_and_confirm("BUFFER")
            break

        price = _get_price_mid()
        bal = _get_conditional_balance()

        # If price unavailable, do NOT act on TP/SL
        if price is None:
            print(f"⏱️ Price: N/A | holding={bal:.4f} | closes in {int((target_time-now).total_seconds())}s", flush=True)
            time.sleep(1)
            continue

        pnl = (float(price) - entry_price) / entry_price
        secs_left = int((target_time - now).total_seconds())

        print(f"⏱️ Price: {price:.3f} | PnL: {pnl*100:+.2f}% | holding={bal:.4f} | closes in {secs_left}s", flush=True)

        # If balance dropped to ~0, we’re closed (manual or prior close filled)
        if bal <= EPS:
            print("✅ Position balance is 0 — closed.", flush=True)
            break

        if pnl <= -STOP_LOSS_PCT:
            _close_and_confirm("STOP LOSS")
            break

        if pnl >= TAKE_PROFIT_PCT:
            _close_and_confirm("TAKE PROFIT")
            break

        time.sleep(PRINT_EVERY_S)

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
