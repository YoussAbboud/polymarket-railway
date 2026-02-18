#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (v10.9 - FIX options object)

Fixes:
- ✅ Uses PartialCreateOrderOptions instead of dict (fixes: 'dict' object has no attribute 'tick_size')
- ✅ Keeps your debug logs + reliable market pick + tick_size/neg_risk signing params
"""

import os, sys, json, time, argparse, requests
from datetime import datetime, timezone, timedelta

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY
from py_clob_client.exceptions import PolyApiException

# ✅ IMPORTANT: options MUST be an object, not dict
try:
    from py_clob_client.clob_types import PartialCreateOrderOptions
except Exception:
    PartialCreateOrderOptions = None


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


def log(msg: str):
    print(msg, flush=True)


def get_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        log(f"❌ ERROR: Missing Env Variable: {key}")
        sys.exit(1)
    return val.strip()


def init_client(signature_type: int) -> ClobClient:
    pk = get_env("PRIVATE_KEY")
    fund_addr = get_env("POLYGON_ADDRESS")

    log("🧾 WALLET CHECK")
    log("------------------------------------------------------")
    log(f"🎯 Funder (proxy / funded wallet): {fund_addr}")
    log("------------------------------------------------------")
    log(f"🔐 Initializing Polymarket auth with signature_type={signature_type} ...")

    client = ClobClient(
        HOST,
        key=pk,
        chain_id=CHAIN_ID,
        signature_type=signature_type,
        funder=fund_addr,
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    log(f"✅ AUTH OK (signature_type={signature_type}). Bot is Live.")
    return client


def get_market_tokens(slug: str):
    try:
        r = requests.get(f"{GAMMA_URL}?slug={slug}", timeout=8)
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

        return {"market_id": market.get("id"), "yes": str(clob_ids[0]), "no": str(clob_ids[1])}
    except Exception:
        return None


def find_current_market(asset: str):
    now = datetime.now(timezone.utc)
    base_ts = int(now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0).timestamp())

    candidates = [base_ts, base_ts - 300, base_ts + 300, base_ts - 600, base_ts + 600]
    tried = []

    for ts in candidates:
        slug = f"{asset.lower()}-updown-5m-{ts}"
        tried.append(slug)
        tokens = get_market_tokens(slug)
        if tokens:
            end_time = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(minutes=5)
            return slug, tokens, end_time

    return None, tried, None


def compute_momentum():
    r = requests.get(COINGECKO_URL, timeout=10).json()
    prices = r.get("prices", [])
    if not prices or len(prices) < 5:
        return None

    anchor_t = prices[-1][0] - 720000  # 12 minutes
    anchor_p = next(p for t, p in reversed(prices) if abs(t - anchor_t) < 300000)
    momentum = ((prices[-1][1] - anchor_p) / anchor_p) * 100
    return float(momentum)


def place_order_with_options(client: ClobClient, token_id: str, price: float, shares: float):
    tick_size = client.get_tick_size(token_id)
    neg_risk = client.get_neg_risk(token_id)

    log(f"🧩 Market params: tick_size={tick_size} | neg_risk={bool(neg_risk)}")

    order_args = OrderArgs(price=price, size=shares, side=BUY, token_id=token_id)

    # ✅ MUST pass an object (PartialCreateOrderOptions), not dict
    if PartialCreateOrderOptions is not None:
        opts = PartialCreateOrderOptions(tick_size=str(tick_size), neg_risk=bool(neg_risk))
        try:
            return client.create_and_post_order(order_args, opts)
        except TypeError:
            return client.create_and_post_order(order_args, options=opts)

    # Fallback (shouldn’t happen on modern versions)
    return client.create_and_post_order(order_args)


def run_strategy(live: bool):
    # Default try: 2 (proxy safe), fallback: 1 (magic/email proxy)
    sigs = [int(os.getenv("POLY_SIGNATURE_TYPE", "2"))]
    if "POLY_SIGNATURE_TYPE" not in os.environ:
        sigs = [2, 1]

    client = init_client(sigs[0])

    log(f"🕒 Now UTC: {datetime.now(timezone.utc).isoformat()} | live={live}")

    slug, tokens_or_tried, end_time = find_current_market(ASSET)
    if not slug:
        log("❌ No market found on Gamma for current window.")
        log("Tried slugs:")
        for s in tokens_or_tried:
            log(f"  - {s}")
        return

    tokens = tokens_or_tried
    log(f"✅ Market found: {slug}")
    log(f"✅ Tokens: YES={tokens['yes'][:10]}… | NO={tokens['no'][:10]}…")

    momentum = compute_momentum()
    if momentum is None:
        log("❌ Could not compute momentum (CoinGecko missing/empty). Exiting.")
        return

    log(f"📈 Momentum (12m): {momentum:+.3f}% | threshold={MIN_MOMENTUM_PCT:.3f}%")

    if abs(momentum) < MIN_MOMENTUM_PCT:
        log("🟡 No trade: momentum below threshold. Exiting.")
        return

    token_to_buy = tokens["yes"] if momentum > 0 else tokens["no"]
    side_name = "YES" if momentum > 0 else "NO"
    log(f"📈 SIGNAL: {side_name}")

    if not live:
        log("🟡 Not live mode (--live not set). Exiting without placing order.")
        return

    try:
        ob = client.get_order_book(token_to_buy)
        best_ask = float(ob.asks[0].price) if ob and ob.asks else 0.5
        limit_price = min(best_ask + 0.01, 0.99)
        shares = round(MAX_BET_SIZE / limit_price, 1)

        log(f"🚀 BUYING: {shares} shares @ {limit_price:.2f}...")

        resp = place_order_with_options(client, token_to_buy, limit_price, shares)
        if resp and resp.get("orderID"):
            log(f"✅ ORDER PLACED: {resp.get('orderID')}")
            monitor_trade(client, token_to_buy, limit_price, end_time)
            return

        log(f"ℹ️ Order response: {resp}")

    except PolyApiException as e:
        msg = ""
        try:
            msg = (e.error_message.get("error") or "").lower()
        except Exception:
            msg = str(e).lower()

        log(f"❌ Trade Failed: {e}")

        if "invalid signature" in msg and len(sigs) > 1:
            log(f"🔁 Retrying with signature_type={sigs[1]} ...")
            client = init_client(sigs[1])

            ob = client.get_order_book(token_to_buy)
            best_ask = float(ob.asks[0].price) if ob and ob.asks else 0.5
            limit_price = min(best_ask + 0.01, 0.99)
            shares = round(MAX_BET_SIZE / limit_price, 1)

            log(f"🚀 BUYING: {shares} shares @ {limit_price:.2f}...")
            resp = place_order_with_options(client, token_to_buy, limit_price, shares)
            if resp and resp.get("orderID"):
                log(f"✅ ORDER PLACED: {resp.get('orderID')}")
                monitor_trade(client, token_to_buy, limit_price, end_time)
                return

            log(f"ℹ️ Order response: {resp}")

    except Exception as e:
        log(f"❌ Trade Failed: {e}")


def monitor_trade(client, token_id, entry_price, end_time):
    log("📊 MONITORING... (Manual Sell available on Polymarket website)")
    last_price, stagnation_start = None, time.time()
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)

    while datetime.now(timezone.utc) < target_time:
        try:
            mid = client.get_midpoint(token_id)
            curr = float(mid) if mid else 0.5

            if curr == last_price:
                if time.time() - stagnation_start >= STAGNATION_TIMEOUT:
                    log("❄️ STAGNATION. Exit.")
                    break
            else:
                last_price, stagnation_start = curr, time.time()

            pnl = (curr - entry_price) / entry_price
            log(f"⏱️ Price: {curr:.3f} | PnL: {pnl*100:+.1f}%")

            if pnl <= -STOP_LOSS_PCT or pnl >= TAKE_PROFIT_PCT:
                break
        except Exception:
            pass
        time.sleep(2)

    log("⏰ MONITOR COMPLETE. Please check your position on the website.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    args = parser.parse_args()
    run_strategy(args.live)
