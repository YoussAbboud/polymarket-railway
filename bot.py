#!/usr/bin/env python3
"""
Railway-ready Polymarket BTC 5m Bot (v11.3)

Fixes:
- ✅ Avoids minified JSON (some CLOB endpoints reject {"a":"b"} but accept {"a": "b"}) :contentReference[oaicite:3]{index=3}
- ✅ Adds small timestamp drift buffer (-5s) to reduce intermittent 401s :contentReference[oaicite:4]{index=4}
- ✅ owner + POLY_ADDRESS always match API key owner (proxy wallet) :contentReference[oaicite:5]{index=5}
- ✅ Monitoring auto-closes TP/SL or close buffer

REQUIRED ENV:
- PRIVATE_KEY
- POLYGON_ADDRESS   (FUNDER / proxy wallet shown on Polymarket)

OPTIONAL ENV:
- SIGNATURE_TYPE    (force 2 or 1). If not set: tries [2, 1]
"""

import os, sys, json, time, argparse, requests
from datetime import datetime, timezone, timedelta
from typing import Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import BUY, SELL

try:
    from py_clob_client.signing.hmac import build_hmac_signature
except Exception:
    build_hmac_signature = None

# ==============================================================================
# 🚀 STRATEGY SETTINGS
# ==============================================================================
ASSET = "BTC"
LOOKBACK_MINS = 12
MIN_MOMENTUM_PCT = 0.08

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

# Polymarket L2 headers (string keys — stable)
H_POLY_ADDRESS = "POLY_ADDRESS"
H_POLY_SIGNATURE = "POLY_SIGNATURE"
H_POLY_TIMESTAMP = "POLY_TIMESTAMP"
H_POLY_API_KEY = "POLY_API_KEY"
H_POLY_PASSPHRASE = "POLY_PASSPHRASE"


# ==============================================================================
# Helpers
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

def parse_slug_ts(slug: str) -> int:
    return int(slug.split("-")[-1])

def parse_slug_end_time(slug: str) -> datetime:
    ts = parse_slug_ts(slug)
    return datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(minutes=5)

def to_primitive(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple, set)):
        return [to_primitive(v) for v in x]
    if isinstance(x, dict):
        return {str(k): to_primitive(v) for k, v in x.items()}

    if hasattr(x, "model_dump") and callable(getattr(x, "model_dump")):  # pydantic v2
        try:
            return to_primitive(x.model_dump())
        except Exception:
            pass

    if hasattr(x, "dict") and callable(getattr(x, "dict")):  # pydantic v1
        try:
            return to_primitive(x.dict())
        except Exception:
            pass

    if hasattr(x, "__dict__"):
        try:
            return to_primitive(vars(x))
        except Exception:
            pass

    return str(x)


# ==============================================================================
# Client init
# ==============================================================================
def init_client(signature_type: int, pk: str, funder: str) -> ClobClient:
    log(f"🔐 Initializing Polymarket auth with signature_type={signature_type} ...")
    client = ClobClient(
        host=HOST,
        key=pk,
        chain_id=CHAIN_ID,
        signature_type=signature_type,
        funder=funder,
    )
    client.set_api_creds(client.create_or_derive_api_creds())

    client._sig_type = signature_type
    client._funder = funder

    log(f"✅ AUTH OK (signature_type={signature_type}). Bot is Live.")
    log("🧾 WALLET CHECK")
    log("------------------------------------------------------")
    try:
        signer = client.signer.address()
        log(f"🔑 Signer (from PRIVATE_KEY): {signer}")
    except Exception:
        pass
    log(f"🎯 Funder (proxy / funded wallet): {funder}")
    log("------------------------------------------------------")
    return client


# ==============================================================================
# Manual POST /order (proxy-safe)
# ==============================================================================
def post_order_fixed_poly_address(client: ClobClient, signed_order, order_type="GTC", post_only=False):
    if build_hmac_signature is None:
        raise RuntimeError(
            "build_hmac_signature not available from py_clob_client. "
            "Pin/upgrade py-clob-client to a version that includes py_clob_client.signing.hmac"
        )

    creds = getattr(client, "creds", None)
    if creds is None:
        raise RuntimeError("Missing L2 creds; did you call set_api_creds()?")

    sig_type = getattr(client, "_sig_type", 2)
    funder = getattr(client, "_funder", None)
    if not funder:
        raise RuntimeError("Missing funder on client.")

    # Proxy/Safe: owner must be funder; (we only use 1/2 here)
    owner_addr = funder if sig_type in (1, 2) else client.signer.address()

    # Build body and enforce owner
    if hasattr(client, "build_post_order_body"):
        body = client.build_post_order_body(signed_order, order_type=order_type, post_only=post_only)
        body = to_primitive(body)
        if isinstance(body, dict):
            body["owner"] = owner_addr
    else:
        body = {
            "order": signed_order,
            "owner": owner_addr,
            "orderType": order_type,
            "apiKey": creds.api_key,
            "postOnly": bool(post_only),
        }
        body = to_primitive(body)

    # ✅ IMPORTANT FIX: DO NOT minify JSON; keep default spacing
    serialized = json.dumps(body, ensure_ascii=False)

    # ✅ IMPORTANT FIX: small drift buffer
    ts = int(time.time()) - 5
    request_path = "/order"
    hmac_sig = build_hmac_signature(
        creds.api_secret,
        ts,
        "POST",
        request_path,
        serialized
    )

    headers = {
        "Content-Type": "application/json",
        H_POLY_ADDRESS: owner_addr,
        H_POLY_SIGNATURE: hmac_sig,
        H_POLY_TIMESTAMP: str(ts),
        H_POLY_API_KEY: creds.api_key,
        H_POLY_PASSPHRASE: creds.api_passphrase,
    }

    resp = requests.post(f"{HOST}{request_path}", headers=headers, data=serialized, timeout=20)
    if resp.status_code >= 400:
        try:
            j = resp.json()
        except Exception:
            j = {"error": resp.text}
        raise Exception(f"POST /order failed [{resp.status_code}]: {j}")

    return resp.json()


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

        return {"market_id": market.get("id"), "yes": str(clob_ids[0]), "no": str(clob_ids[1])}
    except Exception:
        return None

def find_current_market(asset: str):
    base_ts = floor_to_5m(now_utc())
    candidates = [base_ts, base_ts - 300, base_ts + 300, base_ts - 600, base_ts + 600]
    tried = []
    for ts in candidates:
        slug = f"{asset.lower()}-updown-5m-{ts}"
        tried.append(slug)
        tokens = get_market_tokens(slug)
        if tokens:
            return slug, tokens, tried
    return None, None, tried


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
# Order placement
# ==============================================================================
def best_ask(client: ClobClient, token_id: str) -> float:
    ob = client.get_order_book(token_id)
    if ob and ob.asks:
        return float(ob.asks[0].price)
    return 0.5

def best_bid(client: ClobClient, token_id: str) -> float:
    ob = client.get_order_book(token_id)
    if ob and ob.bids:
        return float(ob.bids[0].price)
    return 0.5

def get_market_params(client: ClobClient, token_id: str) -> tuple[str, bool]:
    tick_size = client.get_tick_size(token_id)
    neg_risk = client.get_neg_risk(token_id)
    return str(tick_size if tick_size is not None else "0.01"), bool(neg_risk)

def place_buy_live(client: ClobClient, token_id: str, limit_price: float, shares: float, tick_size: str, neg_risk: bool):
    opts = PartialCreateOrderOptions(tick_size=str(tick_size), neg_risk=bool(neg_risk))
    args = OrderArgs(price=float(limit_price), size=float(shares), side=BUY, token_id=str(token_id))
    signed = client.create_order(args, opts)
    return post_order_fixed_poly_address(client, signed, order_type="GTC", post_only=False)

def place_sell_live(client: ClobClient, token_id: str, limit_price: float, shares: float, tick_size: str, neg_risk: bool):
    opts = PartialCreateOrderOptions(tick_size=str(tick_size), neg_risk=bool(neg_risk))
    args = OrderArgs(price=float(limit_price), size=float(shares), side=SELL, token_id=str(token_id))
    signed = client.create_order(args, opts)
    return post_order_fixed_poly_address(client, signed, order_type="GTC", post_only=False)


# ==============================================================================
# Monitoring + AUTO-CLOSE
# ==============================================================================
def safe_midpoint(client, token_id: str) -> float | None:
    try:
        mid = client.get_midpoint(token_id)
        return float(mid) if mid else None
    except Exception:
        return None

def wait_for_fill(client, order_id: str, timeout_s: int = 35):
    deadline = time.time() + timeout_s
    last = (0.0, None, "unknown")
    while time.time() < deadline:
        try:
            if hasattr(client, "get_order"):
                o = client.get_order(order_id)
            else:
                o = None

            if isinstance(o, dict):
                status = (o.get("status") or o.get("state") or "").lower()
                filled = float(o.get("size_filled") or o.get("filledSize") or o.get("filled_size") or 0.0)
                avg = o.get("avg_fill_price") or o.get("averageFillPrice") or o.get("avgFillPrice")
                avg = float(avg) if avg is not None else None
                last = (filled, avg, status)
                if filled > 0:
                    return filled, avg, status
        except Exception:
            pass
        time.sleep(1.5)
    return last

def best_marketable_sell_price(client, token_id: str, tick_size: str) -> float:
    try:
        tick = float(tick_size or "0.01")
    except Exception:
        tick = 0.01
    b = best_bid(client, token_id)
    px = max(0.01, min(b - tick, 0.99))
    return round(px, 2)

def monitor_trade_and_close(
    client: ClobClient,
    slug: str,
    token_id: str,
    entry_price: float,
    tick_size: str,
    neg_risk: bool,
    buy_order_id: str,
):
    log("📊 MONITORING... (AUTO-CLOSE enabled)")

    end_time = parse_slug_end_time(slug)
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)

    filled_size, avg_fill, status = wait_for_fill(client, buy_order_id, timeout_s=35)
    effective_entry = float(avg_fill) if avg_fill is not None else float(entry_price)

    if filled_size > 0:
        log(f"✅ Filled: {filled_size} @ avg {effective_entry:.3f} (status={status})")
    else:
        log(f"⚠️ Buy not filled yet (status={status}). Will still attempt timed close later.")

    if now_utc() >= target_time:
        log("⏳ Close buffer reached immediately — attempting close now.")
        if filled_size > 0:
            sell_px = best_marketable_sell_price(client, token_id, tick_size)
            log(f"🧾 Closing: SELL {filled_size} @ {sell_px:.2f}...")
            resp = place_sell_live(client, token_id, sell_px, filled_size, tick_size, neg_risk)
            log(f"✅ CLOSE ORDER SENT: {resp.get('orderID') or resp}")
        else:
            log("⚠️ No filled size detected to close yet.")
        return

    last_print = 0.0
    stagnation_start = time.time()
    last_mid = None

    while now_utc() < target_time:
        mid = safe_midpoint(client, token_id)
        if mid is None:
            time.sleep(2)
            continue

        if mid == last_mid:
            if time.time() - stagnation_start >= STAGNATION_TIMEOUT:
                log("❄️ STAGNATION. Will close at buffer.")
                break
        else:
            last_mid = mid
            stagnation_start = time.time()

        pnl = (mid - effective_entry) / effective_entry

        if time.time() - last_print >= 2:
            log(f"⏱️ Price: {mid:.3f} | PnL: {pnl*100:+.1f}% | closes at {target_time.time()} UTC")
            last_print = time.time()

        if pnl <= -STOP_LOSS_PCT:
            log("🛑 STOP LOSS hit — closing now.")
            break
        if pnl >= TAKE_PROFIT_PCT:
            log("🎯 TAKE PROFIT hit — closing now.")
            break

        time.sleep(2)

    filled2, avg2, _ = wait_for_fill(client, buy_order_id, timeout_s=6)
    if filled2 > 0:
        filled_size = filled2
        if avg2 is not None:
            effective_entry = float(avg2)

    if filled_size <= 0:
        log("⚠️ No filled position detected to close (buy may still be pending).")
        log("⏰ MONITOR COMPLETE. Please check your position on the website.")
        return

    sell_px = best_marketable_sell_price(client, token_id, tick_size)
    log(f"🧾 Closing: SELL {filled_size} @ {sell_px:.2f}...")
    resp = place_sell_live(client, token_id, sell_px, filled_size, tick_size, neg_risk)
    log(f"✅ CLOSE ORDER SENT: {resp.get('orderID') or resp}")
    log("✅ DONE.")


# ==============================================================================
# Main
# ==============================================================================
def run_strategy(live: bool):
    pk = get_env("PRIVATE_KEY")
    funder = checksum_lower(get_env("POLYGON_ADDRESS"))

    log(f"🕒 Now UTC: {now_utc().isoformat()} | live={live}")

    slug, tokens, tried = find_current_market(ASSET)
    if not slug or not tokens:
        log("❌ No market found on Gamma for current window.")
        for s in tried:
            log(f"  - {s}")
        return

    log(f"✅ Market found: {slug}")
    log(f"✅ Tokens: YES={tokens['yes'][:10]}… | NO={tokens['no'][:10]}…")

    momentum = compute_momentum_pct(LOOKBACK_MINS)
    if momentum is None:
        log("❌ Could not compute momentum (CoinGecko empty). Exiting.")
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

    # Proxy-only signature types (avoid 0 in proxy setups)
    sig_env = os.getenv("SIGNATURE_TYPE", "").strip()
    if sig_env.isdigit():
        forced = int(sig_env)
        sig_candidates = [forced] if forced in (1, 2) else [2, 1]
    else:
        sig_candidates = [2, 1]

    last_err = None
    for sig_type in sig_candidates:
        try:
            client = init_client(sig_type, pk, funder)

            tick_size, neg_risk = get_market_params(client, token_to_buy)
            log(f"🧩 Market params: tick_size={tick_size} | neg_risk={neg_risk}")

            a = best_ask(client, token_to_buy)
            limit_price = min(a + float(tick_size or "0.01"), 0.99)
            limit_price = round(limit_price, 2)

            shares = round(MAX_BET_SIZE / limit_price, 1)
            log(f"🚀 BUYING: {shares} shares @ {limit_price:.2f}...")

            resp = place_buy_live(client, token_to_buy, limit_price, shares, tick_size, neg_risk)
            order_id = resp.get("orderID") or resp.get("orderId") or resp.get("id")
            if not order_id:
                log(f"ℹ️ Order response (no orderID?): {resp}")
                return

            log(f"✅ ORDER PLACED: {order_id}")

            monitor_trade_and_close(
                client=client,
                slug=slug,
                token_id=token_to_buy,
                entry_price=limit_price,
                tick_size=tick_size,
                neg_risk=neg_risk,
                buy_order_id=order_id,
            )
            return

        except Exception as e:
            last_err = e
            log(f"❌ Trade Failed (signature_type={sig_type}): {e}")

    log("------------------------------------------------------")
    log("❌ All signature types failed to place order.")
    log(f"Last error: {last_err}")
    log("------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    args = parser.parse_args()
    run_strategy(args.live)
