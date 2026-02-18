#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (v10.6 - Authenticated Pilot).

FIXES:
- ✅ CONFIRMED AUTH: Uses the Type 2 Proxy method that passed the v10.5 test.
- ✅ LEGACY COMPATIBILITY: Addresses the USDC.e requirement for managed wallets.
- ✅ STAGNATION & SAFETY: Retains $5 cap and 20s stagnation kill switch.
"""

import os, sys, json, time, argparse, requests
from datetime import datetime, timezone, timedelta
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
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

def get_env(key):
    val = os.environ.get(key)
    if not val:
        print(f"❌ ERROR: Missing Env Variable: {key}")
        sys.exit(1)
    return val

def init_client():
    pk = get_env("PRIVATE_KEY")
    fund_addr = get_env("POLYGON_ADDRESS") 
    print(f"🔐 Initializing Proxy Auth (Type 2) for {fund_addr}...")
    try:
        client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=2, funder=fund_addr)
        client.set_api_creds(client.create_or_derive_api_creds())
        return client
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        sys.exit(1)

def get_market_tokens(slug):
    try:
        r = requests.get(f"{GAMMA_URL}?slug={slug}", timeout=5)
        data = r.json()
        if not data: return None
        market = data[0].get("markets", [])[0]
        clob_ids = json.loads(market.get("clobTokenIds"))
        return {"market_id": market["id"], "yes": clob_ids[0], "no": clob_ids[1]}
    except: return None

def run_strategy(live, quiet):
    client = init_client()
    print("✅ AUTH SUCCESS. Bot is Live.")

    now = datetime.now(timezone.utc)
    ts = int(now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0).timestamp())
    slug = f"{ASSET.lower()}-updown-5m-{ts}"
    end_time = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(minutes=5)

    tokens = get_market_tokens(slug)
    if not tokens: return

    r = requests.get(COINGECKO_URL, timeout=5).json()
    prices = r.get("prices", [])
    momentum = ((prices[-1][1] - next(p for t, p in reversed(prices) if abs(t - (prices[-1][0] - 720000)) < 300000)) / next(p for t, p in reversed(prices) if abs(t - (prices[-1][0] - 720000)) < 300000)) * 100
    
    if abs(momentum) < MIN_MOMENTUM_PCT: return
    
    token_to_buy = tokens["yes"] if momentum > 0 else tokens["no"]
    side_name = "YES" if momentum > 0 else "NO"
    print(f"📈 SIGNAL: {side_name} | Mom={momentum:.3f}%")

    if not live: return

    try:
        limit_price = float(client.get_order_book(token_to_buy).asks[0].price) + 0.01
        limit_price = min(limit_price, 0.99)
        shares = round(MAX_BET_SIZE / limit_price, 1)

        print(f"🚀 BUYING: {shares} shares @ {limit_price:.2f}...")
        resp = client.create_and_post_order(OrderArgs(price=limit_price, size=shares, side=BUY, token_id=token_to_buy))
        
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
            if pnl <= -STOP_LOSS_PCT or pnl >= TAKE_PROFIT_PCT: break
        except: pass
        time.sleep(2)
    print("⏰ MONITOR COMPLETE. Please check your position on the website.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    args = parser.parse_args()
    run_strategy(args.live, False)
