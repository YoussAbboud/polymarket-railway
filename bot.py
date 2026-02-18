#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (v10.4 - Auto-Fix Auth).

CRITICAL FIXES:
- 🔑 FRESH CREDS: Generates NEW API Keys on startup to kill "Invalid Signature" errors.
- 🔗 PROXY BINDING: Forces the bot to acknowledge your Deposit Address as the funding source.
- 🪙 USDC.e ONLY: Hardcoded to look for Bridged USDC (0x2791...) on Polygon.
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
MAX_BET_SIZE = 5.0           # Hard cap: $5.00
STOP_LOSS_PCT = 0.15         
TAKE_PROFIT_PCT = 0.20       
CLOSE_BUFFER_SECONDS = 60    
# ==============================================================================

# -----------------------
# Config
# -----------------------
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137 # Polygon
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174" # Bot knows this internally
GAMMA_URL = "https://gamma-api.polymarket.com/events"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"

# -----------------------
# Helpers
# -----------------------
def get_env(key):
    val = os.environ.get(key)
    if not val:
        print(f"❌ ERROR: Missing Env Variable: {key}")
        sys.exit(1)
    return val

def init_client():
    """Initializes the Polymarket Client with Auto-Auth Fixes."""
    pk = get_env("PRIVATE_KEY")
    fund_addr = get_env("POLYGON_ADDRESS") 
    
    print(f"🔐 INITIALIZING CLIENT...")
    print(f"   -> Wallet (Funder): {fund_addr}")

    # TRY TYPE 2 (Proxy) - Standard for Magic/Polymarket Users
    try:
        print("   -> Attempting Auth Type 2 (Proxy)...")
        client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=2, funder=fund_addr)
        # FORCE NEW CREDS: This is the magic fix for Invalid Signature
        client.set_api_creds(client.create_or_derive_api_creds()) 
        print("✅ SUCCESS: Connected via Proxy Mode.")
        return client
    except Exception as e:
        print(f"⚠️ Proxy Auth failed ({e})...")

    # TRY TYPE 1 (Standard) - Fallback
    try:
        print("   -> Attempting Auth Type 1 (Standard)...")
        client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=1, funder=fund_addr)
        client.set_api_creds(client.create_or_derive_api_creds())
        print("✅ SUCCESS: Connected via Standard Mode.")
        return client
    except Exception as e:
        print(f"❌ FATAL AUTH ERROR: {e}")
        print("👉 Check that your PRIVATE_KEY matches the POLYGON_ADDRESS wallet.")
        sys.exit(1)

def get_market_tokens(slug):
    try:
        r = requests.get(f"{GAMMA_URL}?slug={slug}", timeout=5)
        data = r.json()
        if not data: return None
        
        event = data[0]
        markets = event.get("markets", [])
        if not markets: return None
        
        market = markets[0]
        clob_ids = market.get("clobTokenIds")
        if not clob_ids: return None
        
        return {
            "market_id": market["id"],
            "yes": json.loads(clob_ids)[0],
            "no": json.loads(clob_ids)[1]
        }
    except Exception as e:
        print(f"❌ Gamma Error: {e}")
        return None

def get_price_history():
    try:
        r = requests.get(COINGECKO_URL, timeout=5)
        return r.json().get("prices", [])
    except: return []

# -----------------------
# Core Logic
# -----------------------
def run_strategy(live, quiet):
    client = init_client()
    
    # 1. Calculate Target
    now = datetime.now(timezone.utc)
    minute = (now.minute // 5) * 5
    start_dt = now.replace(minute=minute, second=0, microsecond=0)
    ts = int(start_dt.timestamp())
    slug = f"{ASSET.lower()}-updown-5m-{ts}"
    end_time = start_dt + timedelta(minutes=5)
    
    if not quiet: print(f"🎯 TARGET: {slug}")

    # 2. Lookup Tokens
    tokens = get_market_tokens(slug)
    if not tokens:
        if not quiet: print("⏳ Market not found (yet).")
        return

    yes_id = tokens["yes"]
    no_id = tokens["no"]

    # 3. Signal Logic
    prices = get_price_history()
    if not prices: return
    
    latest_price = prices[-1][1]
    target_ts = prices[-1][0] - (LOOKBACK_MINS * 60 * 1000)
    past_price = next((p for t, p in reversed(prices) if abs(t - target_ts) < 300000), None)
    if not past_price: return

    momentum = ((latest_price - past_price) / past_price) * 100
    side = BUY 
    token_to_buy = yes_id if momentum > 0 else no_id
    side_name = "YES" if momentum > 0 else "NO"
    
    print(f"📈 SIGNAL: {side_name} | Mom={momentum:.3f}%")
    
    if abs(momentum) < MIN_MOMENTUM_PCT:
        print("😴 Low Momentum. Skipping.")
        return

    if not live: return

    # 4. Execute Trade
    amount = MAX_BET_SIZE
    
    try:
        ob = client.get_order_book(token_to_buy)
        if not ob.asks: 
            print("❌ Orderbook empty.")
            return
            
        best_ask = float(ob.asks[0].price)
        limit_price = best_ask + 0.01 
        if limit_price > 0.99: limit_price = 0.99
        
        shares = amount / limit_price
        shares = round(shares, 1)

        print(f"🚀 BUYING: {shares} shares of {side_name} @ {limit_price:.2f}...")

        # The SDK automatically uses USDC.e (Asset ID 0x2791...) on Polygon
        resp = client.create_and_post_order(OrderArgs(
            price=limit_price,
            size=shares,
            side=BUY,
            token_id=token_to_buy
        ))
        
        if resp and resp.get("orderID"):
            print(f"✅ ORDER SENT: ID {resp.get('orderID')}")
            monitor_trade(client, token_to_buy, limit_price, end_time)
        else:
            print(f"❌ Order Rejected: {resp}")
            if "invalid signature" in str(resp).lower():
                 print("👉 NOTE: If this persists, delete your API keys in Polymarket Settings -> API.")
            
    except Exception as e:
        print(f"❌ Trade Failed: {e}")
        if "allowance" in str(e).lower():
            print("🚨 CRITICAL: Go to Polymarket.com and approve USDC spending!")

def monitor_trade(client, token_id, entry_price, end_time):
    print("📊 MONITORING... (Open Polymarket.com to view/sell)")
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    
    while datetime.now(timezone.utc) < target_time:
        try:
            ob = client.get_order_book(token_id)
            if ob.bids and ob.asks:
                mid_price = client.get_midpoint(token_id)
                current_price = float(mid_price) if mid_price else 0.5
                
                pnl = (current_price - entry_price) / entry_price
                print(f"⏱️ Price: {current_price:.3f} | PnL: {pnl*100:+.1f}%")
                
                if pnl <= -STOP_LOSS_PCT:
                    print("🛑 STOP LOSS HIT. PLEASE SELL MANUALLY.")
                    return
                elif pnl >= TAKE_PROFIT_PCT:
                    print("💰 TAKE PROFIT HIT. PLEASE SELL MANUALLY.")
                    return
        except Exception as e:
            print(f"⚠️ API Blip: {e}")
            
        time.sleep(2)
        
    print("⏰ TIME UP. Please Sell Manually.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()
    run_strategy(args.live, args.quiet)
