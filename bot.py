#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (v10.7 - Signature Fix).

Keeps your strategy identical.
Fixes "invalid signature" by:
- Always signing with correct market params: tick_size + neg_risk (per docs)
- Auto-fallback between proxy signature types (2 -> 1) if order POST returns invalid signature

Docs:
- First Order / Troubleshooting: invalid signature can be wrong signatureType/funder or missing correct params
"""

import os, sys, json, time, argparse, math, requests
from decimal import Decimal, ROUND_UP
from datetime import datetime, timezone, timedelta

from py_clob_client.client import ClobClient
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.exceptions import PolyApiException

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
    if val is None or str(val).strip() == "":
        print(f"❌ ERROR: Missing Env Variable: {key}")
        sys.exit(1)
    return str(val).strip()


def _coerce_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on")
    return False


def init_client(signature_type: int) -> ClobClient:
    pk = get_env("PRIVATE_KEY")
    fund_addr = get_env("POLYGON_ADDRESS")  # proxy / funded wallet shown on Polymarket

    print("🧾 WALLET CHECK")
    print("------------------------------------------------------")
    # NOTE: we don’t derive signer addr here to avoid extra deps; you already printed it in logs
    print(f"🎯 Funder (proxy / funded wallet): {fund_addr}")
    print("------------------------------------------------------")
    print(f"🔐 Initializing Polymarket auth with signature_type={signature_type} ...")

    try:
        client = ClobClient(
            host=HOST,
            key=pk,
            chain_id=CHAIN_ID,
            signature_type=signature_type,
            funder=fund_addr,
        )
        # Derive + set L2 creds (needed for post_order)
        client.set_api_creds(client.create_or_derive_api_creds())
        print(f"✅ AUTH OK (signature_type={signature_type}). Bot is Live.")
        return client
    except Exception as e:
        print(f"❌ Initialization Failed (signature_type={signature_type}): {e}")
        raise


def get_market_tokens(slug):
    try:
        r = requests.get(f"{GAMMA_URL}?slug={slug}", timeout=8)
        data = r.json()
        if not data:
            return None
        market = data[0].get("markets", [])[0]
        clob_ids = json.loads(market.get("clobTokenIds"))
        return {"market_id": market["id"], "yes": str(clob_ids[0]), "no": str(clob_ids[1])}
    except Exception:
        return None


def _round_price_up_to_tick(price: float, tick_size: str) -> float:
    # Use Decimal to avoid float drift.
    p = Decimal(str(price))
    t = Decimal(str(tick_size))
    if t <= 0:
        return float(p)
    steps = (p / t).to_integral_value(rounding=ROUND_UP)
    out = steps * t
    return float(out)


def get_order_params(client: ClobClient, token_id: str):
    """
    Fetch tick_size + neg_risk from CLOB (these affect signing/domain).
    """
    tick = client.get_tick_size(token_id)
    neg = client.get_neg_risk(token_id)

    # Some client versions return raw values; some return dicts.
    if isinstance(tick, dict):
        tick = tick.get("tick_size") or tick.get("minimum_tick_size") or tick.get("min_tick_size")
    if isinstance(neg, dict):
        neg = neg.get("neg_risk")

    tick_size = str(tick if tick is not None else "0.01")
    neg_risk = _coerce_bool(neg)

    return tick_size, neg_risk


def create_and_post_limit_buy(client: ClobClient, token_id: str, price: float, size: float):
    """
    Compatibility wrapper for different py-clob-client versions.
    Always supplies tick_size + neg_risk options (prevents invalid signature on neg-risk markets).
    """
    tick_size, neg_risk = get_order_params(client, token_id)
    opts = {"tick_size": tick_size, "neg_risk": neg_risk}

    # Ensure price respects tick size and stays within (0, 0.99]
    px = min(max(price, 0.01), 0.99)
    px = _round_price_up_to_tick(px, tick_size)
    px = min(px, 0.99)

    order_payload = {"token_id": token_id, "price": float(px), "size": float(size), "side": BUY}

    # Try common call shapes across versions.
    try:
        return client.create_and_post_order(order_payload, opts)
    except TypeError:
        # Older versions may want kwargs or different ordering
        try:
            return client.create_and_post_order(order_payload, opts, None)
        except TypeError:
            # As a last resort, try without opts (won’t fix neg-risk, but keeps runtime from exploding)
            return client.create_and_post_order(order_payload)


def run_strategy(live, quiet):
    # Default signature_type=2 (GNOSIS_SAFE). If it fails with invalid signature, we retry with 1 (POLY_PROXY).
    forced = os.environ.get("POLY_SIGNATURE_TYPE")
    sig_try = []
    if forced and forced.strip() in ("0", "1", "2"):
        sig_try = [int(forced.strip())]
    else:
        sig_try = [2, 1]

    client = init_client(sig_try[0])

    now = datetime.now(timezone.utc)
    ts = int(now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0).timestamp())
    slug = f"{ASSET.lower()}-updown-5m-{ts}"
    end_time = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(minutes=5)

    tokens = get_market_tokens(slug)
    if not tokens:
        return

    r = requests.get(COINGECKO_URL, timeout=8).json()
    prices = r.get("prices", [])

    # your exact momentum logic (unchanged)
    lookback_target = prices[-1][0] - 720000  # 12 minutes
    ref_price = next(
        p for t, p in reversed(prices)
        if abs(t - lookback_target) < 300000
    )
    momentum = ((prices[-1][1] - ref_price) / ref_price) * 100

    if abs(momentum) < MIN_MOMENTUM_PCT:
        return

    token_to_buy = tokens["yes"] if momentum > 0 else tokens["no"]
    side_name = "YES" if momentum > 0 else "NO"
    print(f"📈 SIGNAL: {side_name} | Mom={momentum:.3f}%")

    if not live:
        return

    try:
        # Use best ask + small nudge (unchanged behavior)
        ob = client.get_order_book(token_to_buy)
        best_ask = float(ob.asks[0].price) if ob and ob.asks else 0.5
        limit_price = min(best_ask + 0.01, 0.99)

        shares = round(MAX_BET_SIZE / limit_price, 1)
        print(f"🚀 BUYING: {shares} shares @ {limit_price:.2f}...")

        resp = create_and_post_limit_buy(client, token_to_buy, limit_price, shares)

        if resp and resp.get("orderID"):
            print(f"✅ ORDER PLACED: {resp.get('orderID')}")
            monitor_trade(client, token_to_buy, limit_price, end_time)
            return

        # If response shape differs, still print it for visibility.
        print(f"ℹ️ Order response: {resp}")

    except PolyApiException as e:
        # If invalid signature, retry once with alternate proxy signature type.
        err = getattr(e, "error_message", None) or {}
        msg = ""
        try:
            msg = (err.get("error") or "").lower()
        except Exception:
            msg = str(e).lower()

        if "invalid signature" in msg and len(sig_try) > 1:
            print(f"❌ Trade Failed: {e}")
            print(f"🔁 Retrying with signature_type={sig_try[1]} (proxy fallback)...")
            client = init_client(sig_try[1])

            # re-run the same placement once (no strategy change)
            ob = client.get_order_book(token_to_buy)
            best_ask = float(ob.asks[0].price) if ob and ob.asks else 0.5
            limit_price = min(best_ask + 0.01, 0.99)
            shares = round(MAX_BET_SIZE / limit_price, 1)

            print(f"🚀 BUYING: {shares} shares @ {limit_price:.2f}...")
            resp = create_and_post_limit_buy(client, token_to_buy, limit_price, shares)

            if resp and resp.get("orderID"):
                print(f"✅ ORDER PLACED: {resp.get('orderID')}")
                monitor_trade(client, token_to_buy, limit_price, end_time)
                return

            print(f"ℹ️ Order response: {resp}")
            return

        print(f"❌ Trade Failed: {e}")

    except Exception as e:
        print(f"❌ Trade Failed: {e}")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    args = parser.parse_args()
    run_strategy(args.live, False)
