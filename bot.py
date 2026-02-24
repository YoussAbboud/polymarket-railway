#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (v17 - Sniper Edition).
- EXIT FIX: Fetches the exact blockchain balance right before selling.
- ENTRY FIX: Requires high momentum (0.15%) AND dominant aggressive volume (>55%).
- RISK MANAGEMENT: Max 1 trade per 15m window to prevent revenge trading.
- DISCORD ALERTS: Active.
"""

import os, sys, json, time, argparse, atexit
from datetime import datetime, timezone, timedelta
import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, BalanceAllowanceParams, AssetType
from py_clob_client.order_builder.constants import BUY, SELL

# ==============================================================================
# 🎯 STRATEGY SETTINGS (v17 - SNIPER MODE)
# ==============================================================================
ASSET = "BTC"
BASE_THRESHOLD = 0.15      # INCREASED: Needs a real breakout, not just noise.
MAX_SPREAD = 0.10          
MAX_BET_SIZE = 10.0        # SCALED UP: $10 sniper shots.
TAKE_PROFIT_PCT = 0.15     
STOP_LOSS_PCT = 0.33       
CLOSE_BUFFER_SECONDS = 120 
MAX_TRADES_PER_WINDOW = 1  # DECREASED: 1 shot per window. No revenge trading.
# ==============================================================================

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
BINANCE_URL = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=5"

STATE_DIR = os.getenv("STATE_DIR", "/data")
STATE_PATH = os.path.join(STATE_DIR, "window_lock.json")
LOCK_PATH = os.path.join(STATE_DIR, "pm_lock")
SESSION = requests.Session()

def die(msg: str): 
    print(msg, flush=True)
    sys.exit(1)

def get_env(key: str): 
    val = os.environ.get(key)
    if not val: die(f"❌ ERROR: Missing Env Variable: {key}")
    return val

def utc_now(): 
    return datetime.now(timezone.utc)

def round_to_15m(ts: datetime): 
    return ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)

def safe_json(resp: requests.Response): 
    try: return resp.json()
    except: return None

# --- DISCORD INTEGRATION ---
def send_discord_alert(msg: str):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url: 
        return
    try:
        requests.post(webhook_url, json={"content": msg}, timeout=5)
    except Exception as e:
        print(f"⚠️ Discord Alert Failed: {e}", flush=True)

def get_trade_count(ws_ts: datetime):
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, 'r') as f:
                data = json.load(f)
                if data.get("window") == int(ws_ts.timestamp()):
                    return data.get("count", 0)
    except: pass
    return 0

def increment_trade_count(ws_ts: datetime):
    try:
        count = get_trade_count(ws_ts) + 1
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(STATE_PATH, 'w') as f:
            json.dump({"window": int(ws_ts.timestamp()), "count": count}, f)
    except Exception as e:
        print(f"⚠️ Failed to save window state: {e}", flush=True)

def ensure_process_lock():
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(LOCK_PATH):
        try:
            if time.time() - os.path.getmtime(LOCK_PATH) > 5 * 60: os.remove(LOCK_PATH)
        except: pass
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError: 
        die("⚠️ Duplicate instance detected. Exiting.")
        
    def _cleanup():
        try:
            if os.path.exists(LOCK_PATH): os.remove(LOCK_PATH)
        except: pass
    atexit.register(_cleanup)

def init_client():
    pk = get_env("PRIVATE_KEY")
    funder = get_env("POLYGON_ADDRESS")
    sig_type = int(os.getenv("SIGNATURE_TYPE", "1"))
    
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=sig_type, funder=funder)
    client.set_api_creds(client.create_or_derive_api_creds())
    return client

def compute_binance_trend():
    try:
        r = SESSION.get(BINANCE_URL, timeout=5)
        data = safe_json(r)
        if not data or len(data) < 5: return None
        
        # Extract closing prices
        old_close = float(data[0][4])
        current_close = float(data[-1][4])
        
        # Calculate Price Momentum
        price_momentum = ((current_close - old_close) / old_close) * 100.0
        
        # Analyze Volume for the last 5 minutes 

[Image of Volume Price Trend Indicator]

        total_volume = 0
        taker_buy_volume = 0
        
        for candle in data:
            total_volume += float(candle[5])        # Total Base Volume
            taker_buy_volume += float(candle[9])    # Taker Buy Base Asset Volume (Aggressive Buys)
            
        if total_volume == 0: return None
        
        buy_pressure_pct = (taker_buy_volume / total_volume) * 100.0
        
        print(f"🔍 Trend Check: Momentum {price_momentum:+.3f}% | Buy Pressure: {buy_pressure_pct:.1f}%", flush=True)

        # THE SNIPER FILTER: 
        if price_momentum > 0 and buy_pressure_pct < 55.0:
            print("⚠️ Price moved UP, but volume is weak. Ignoring fake-out.", flush=True)
            return 0.0 
            
        if price_momentum < 0 and buy_pressure_pct > 45.0:
            print("⚠️ Price moved DOWN, but sell volume is weak. Ignoring fake-out.", flush=True)
            return 0.0

        return price_momentum
        
    except Exception as e: 
        print(f"⚠️ Binance Fetch Error: {e}")
        return None

def get_market_tokens(base_ts: int):
    slugs = [f"btc-updown-15m-{base_ts}", f"btc-up-or-down-15m-{base_ts}"]
    for slug in slugs:
        try:
            r = SESSION.get(f"{GAMMA_EVENTS_URL}?slug={slug}", timeout=8)
            data = safe_json(r)
            if data and isinstance(data, list) and len(data) > 0:
                m = data[0].get("markets", [])[0]
                clob_ids = json.loads(m.get("clobTokenIds", "[]"))
                return {"yes": str(clob_ids[0]), "no": str(clob_ids[1]), "slug": slug}
        except: pass
    return None

def place_buy(client: ClobClient, token_id: str, dollars: float, momentum: float):
    ask_resp = client.get_price(token_id, side=BUY)
    bid_resp = client.get_price(token_id, side=SELL)
    
    if not isinstance(ask_resp, dict) or not isinstance(bid_resp, dict):
        raise RuntimeError("Invalid response from Polymarket Live Price API.")
        
    ask_str = ask_resp.get("price")
    bid_str = bid_resp.get("price")
    
    if ask_str is None or bid_str is None:
        raise RuntimeError("No live pricing available right now.")
        
    ask = float(ask_str)
    bid = float(bid_str)
    spread = ask - bid
    
    print(f"📊 LIVE Market Stats: Ask={ask:.2f} | Bid={bid:.2f} | Spread={spread:.2f}", flush=True)
    
    if spread > MAX_SPREAD:
        raise RuntimeError(f"Spread {spread:.2f} too extreme. Capital protection active.")
    
    if abs(momentum) < BASE_THRESHOLD:
        raise RuntimeError(f"Momentum {abs(momentum):.3f}% too weak to justify entry.")

    if ask > 0.75: 
        raise RuntimeError("Price too high (Bad EV). Will not chase.")
        
    if ask < 0.30:
        raise RuntimeError("Price too low (Dead Token). Refusing to buy trash.")

    price = min(ask, 0.75)
    size = round(dollars / price, 4) 
    
    print(f"🚀 SNIPER ENTRY: Buying {size} shares @ {price}...", flush=True)
    resp = client.create_and_post_order(OrderArgs(price=price, size=size, side=BUY, token_id=token_id))
    
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(f"Polymarket API rejected the BUY: {resp.get('error')}")
        
    print(f"✅ Order sent. Trusting the fill.", flush=True)
    send_discord_alert(f"🟢 **POSITION OPENED**\nAction: Bought **{size}** shares @ **${price:.2f}**\nBinance Momentum: `{momentum:+.3f}%`")
    
    return price, size

def monitor_and_autoclose(client: ClobClient, token_id: str, end_time: datetime, entry_price: float, size: float):
    print("📊 MONITORING 15m POSITION (Live CLOB PnL Only)...", flush=True)
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    
    while True:
        if utc_now() >= end_time:
            msg = "⏳ **MARKET EXPIRED**\nPosition locked by Polymarket. Awaiting automatic resolution."
            print(msg, flush=True)
            send_discord_alert(msg)
            return

        try:
            bid_resp = client.get_price(token_id, side=SELL)
            bid_str = bid_resp.get("price")
            
            if bid_str:
                current_bid = float(bid_str)
                pct_pnl = ((current_bid - entry_price) / entry_price) * 100.0
                print(f"⏱️ PnL: {pct_pnl:+.1f}% (Current Bid: {current_bid:.2f})", flush=True)

                trigger = None
                if pct_pnl >= (TAKE_PROFIT_PCT * 100): trigger = "TAKE PROFIT"
                elif pct_pnl <= -(STOP_LOSS_PCT * 100): trigger = "STOP LOSS"
                elif utc_now() >= target_time: trigger = "TIME LIMIT"

                if trigger:
                    print(f"🧾 EXITING POSITION: {trigger} | Fetching exact blockchain balance...", flush=True)
                    try:
                        bal_resp = client.get_balance_allowance(
                            BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
                        )
                        raw_balance = float(bal_resp.get("balance", 0)) / 1_000_000.0
                        
                        safe_sell_size = int(raw_balance * 10000) / 10000.0
                        
                        if safe_sell_size < 0.01:
                            print("❌ FATAL: 0 shares detected. Phantom trade or manual exit. Shutting down.", flush=True)
                            return
                            
                        client.create_and_post_order(OrderArgs(price=0.01, size=safe_sell_size, side=SELL, token_id=token_id))
                        print(f"✅ Sell order ({safe_sell_size} shares) executed successfully for {trigger}.", flush=True)
                        send_discord_alert(f"🔴 **POSITION CLOSED** [{trigger}]\nAction: Sold **{safe_sell_size}** shares\nEntry: **${entry_price:.2f}** | Exit PnL: **{pct_pnl:+.1f}%**")
                        return
                    except Exception as sell_err:
                        if "not enough balance" in str(sell_err).lower() or "allowance" in str(sell_err).lower():
                            print("❌ FATAL: 0 shares detected. Phantom trade or manual exit. Shutting down.", flush=True)
                            return
                        print(f"⚠️ Sell Failed ({trigger}): {sell_err}. Retrying in 2s...", flush=True)
        except Exception as e: 
            pass 
        time.sleep(2)

def run(live: bool):
    ensure_process_lock()
    client = init_client()
    now = utc_now()
    ws = round_to_15m(now)
    
    trade_count = get_trade_count(ws)
    if trade_count >= MAX_TRADES_PER_WINDOW:
        print(f"✅ Already traded {MAX_TRADES_PER_WINDOW} times in this 15m window. Sleeping.", flush=True)
        return
    
    mom = compute_binance_trend()
    if mom is None: 
        print("⚠️ Could not fetch Binance data.", flush=True)
        return
        
    print(f"🕒 Window: {ws.strftime('%H:%M')} UTC | 📈 Binance Trend (5m -> 15m): {mom:+.3f}%", flush=True)

    tokens = get_market_tokens(int(ws.timestamp()))
    if not tokens: 
        print("⚠️ 15m Market not found. Waiting for next window...", flush=True)
        return
        
    print(f"✅ Market Found: {tokens['slug']}", flush=True)
    
    # NOTE: The new sniper logic returns 0.0 if volume doesn't match momentum.
    # The BASE_THRESHOLD check in place_buy will catch it and prevent the trade.
    token_id = tokens["yes"] if mom > 0 else tokens["no"]
    
    if live:
        try:
            entry_price, size = place_buy(client, token_id, MAX_BET_SIZE, mom)
            increment_trade_count(ws) 
            monitor_and_autoclose(client, token_id, ws + timedelta(minutes=15), entry_price, size)
        except Exception as e: 
            print(f"❌ {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    run(parser.parse_args().live)
