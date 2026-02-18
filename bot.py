#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (v10.6 - Authenticated Pilot).

FIXES (ONLY AUTH/SIGNATURE LAYER):
- ✅ Proxy wallet mismatch is expected (Signer != Funder)
- ✅ Auto-selects working signature_type (2 then 1) unless forced via env
- ✅ Supports using pre-saved API creds (API_KEY/SECRET/PASSPHRASE) to avoid re-deriving
"""

import os, sys, json, time, argparse, requests
from datetime import datetime, timezone, timedelta

from eth_account import Account
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, ApiCreds, OpenOrderParams
from py_clob_client.order_builder.constants import BUY, SELL

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

def normalize_pk(pk: str) -> str:
    pk = pk.strip()
    if not pk.startswith("0x"):
        pk = "0x" + pk
    if len(pk) != 66:
        raise ValueError("PRIVATE_KEY must be 32 bytes hex (64 chars) with optional 0x prefix.")
    return pk

def build_client(signature_type: int, pk: str, funder: str, creds: ApiCreds | None) -> ClobClient:
    # For proxy wallets you MUST pass funder + signature_type (1 or 2)
    client = ClobClient(
        host=HOST,
        chain_id=CHAIN_ID,
        key=pk,
        creds=creds,
        signature_type=signature_type,
        funder=funder,
    )
    if creds is None:
        client.set_api_creds(client.create_or_derive_api_creds())
    return client

def init_client() -> ClobClient:
    pk = normalize_pk(get_env("PRIVATE_KEY"))

    # IMPORTANT:
    # This must be the Polymarket wallet shown in your settings / under your nickname (proxy wallet / funded wallet).
    funder = get_env("POLYGON_ADDRESS")  # keep your env name, but this is the FUNDER (proxy) address

    signer = Account.from_key(pk).address
    print("\n🧾 WALLET CHECK")
    print("------------------------------------------------------")
    print(f"🔑 Signer (from PRIVATE_KEY): {signer}")
    print(f"🎯 Funder (proxy / funded wallet): {funder}")
    print("------------------------------------------------------\n")

    # Optional: use saved L2 creds (recommended so you don't constantly derive keys)
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("SECRET")
    api_passphrase = os.getenv("PASSPHRASE")
    creds = None
    if api_key and api_secret and api_passphrase:
        creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase)

    # If you want to force one:
    #   POLY_SIGNATURE_TYPE=2  (or 1)
    forced = os.getenv("POLY_SIGNATURE_TYPE")
    if forced:
        candidates = [int(forced)]
    else:
        # Default: try the most common proxy first (2), then Magic proxy (1)
        candidates = [2, 1]

    last_err = None
    for sig_type in candidates:
        try:
            print(f"🔐 Initializing Polymarket auth with signature_type={sig_type} ...")
            client = build_client(sig_type, pk, funder, creds)

            # Sanity L2 call: proves headers/auth are working
            _ = client.get_orders(OpenOrderParams())

            print(f"✅ AUTH OK (signature_type={sig_type}). Bot is Live.\n")
            client._sig_type = sig_type  # debug tag
            return client
        except Exception as e:
            last_err = e
            print(f"⚠️ AUTH FAILED (signature_type={sig_type}): {e}\n")

    print(f"❌ Initialization Failed: {last_err}")
    sys.exit(1)

def get_market_tokens(slug):
    try:
        r = requests.get(f"{GAMMA_URL}?slug={slug}", timeout=5)
        data = r.json()
        if not data:
            return None
        market = data[0].get("markets", [])[0]
        clob_ids = json.loads(market.get("clobTokenIds"))
        return {"market_id": market["id"], "yes": clob_ids[0], "no": clob_ids[1]}
    except:
        return None

def run_strategy(live, quiet):
    client = init_client()

    now = datetime.now(timezone.utc)
    ts = int(now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0).timestamp())
    slug = f"{ASSET.lower()}-updown-5m-{ts}"
    end_time = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(minutes=5)

    tokens = get_market_tokens(slug)
    if not tokens:
        return

    r = requests.get(COINGECKO_URL, timeout=5).json()
    prices = r.get("prices", [])
    if len(prices) < 5:
        return

    # your original momentum logic (unchanged)
    anchor_t = prices[-1][0] - 720000
    anchor_p = next(p for t, p in reversed(prices) if abs(t - anchor_t) < 300000)
    momentum = ((prices[-1][1] - anchor_p) / anchor_p) * 100

    if abs(momentum) < MIN_MOMENTUM_PCT:
        return

    token_to_buy = tokens["yes"] if momentum > 0 else tokens["no"]
    side_name = "YES" if momentum > 0 else "NO"
    print(f"📈 SIGNAL: {side_name} | Mom={momentum:.3f}%")

    if not live:
        return

    try:
        limit_price = float(client.get_order_book(token_to_buy).asks[0].price) + 0.01
        limit_price = min(limit_price, 0.99)
        shares = round(MAX_BET_SIZE / limit_price, 1)

        print(f"🚀 BUYING: {shares} shares @ {limit_price:.2f}...")
        resp = client.create_and_post_order(
            OrderArgs(price=limit_price, size=shares, side=BUY, token_id=token_to_buy)
        )

        if resp and resp.get("orderID"):
            print(f"✅ ORDER PLACED: {resp.get('orderID')}")
            monitor_trade(client, token_to_buy, limit_price, end_time)

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
        except:
            pass
        time.sleep(2)

    print("⏰ MONITOR COMPLETE. Please check your position on the website.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    args = parser.parse_args()
    run_strategy(args.live, False)
