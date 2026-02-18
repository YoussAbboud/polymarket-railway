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
    print("📊 MONITORING... (Manual Sell available on Polymarket website)")
    last_price, stagnation_start = None, time.time()
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)

    while datetime.now(timezone.utc) < target_time:
        try:
            mid = client.get_midpoint(token_id)
            curr = float(mid) if mid else 0.5

            if curr == last_price:
                if time.time() - stagnation_start >= STAGNATION_TIMEOUT:
                    print("\n❄️ STAGNATION. Exit.")
                    break
            else:
                last_price, stagnation_start = curr, time.time()

            pnl = (curr - entry_price) / entry_price
            print(f"⏱️ Price: {curr:.3f} | PnL: {pnl*100:+.1f}%")
            if pnl <= -STOP_LOSS_PCT or pnl >= TAKE_PROFIT_PCT:
                break
        except Exception:
            pass
        time.sleep(2)

    print("⏰ MONITOR COMPLETE. Please check your position on the website.")

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
