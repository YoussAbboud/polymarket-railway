#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (v12.4 - The Middle Ground).
- FIX 1: Max Spread tightened to 0.08c to prevent heavy instant losses.
- FIX 2: Removed the +0.01 'Fill Penalty' to get better entry EV.
"""

import os, sys, json, time, argparse, atexit
from datetime import datetime, timezone, timedelta
import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

# ==============================================================================
# 🚀 STRATEGY SETTINGS (v12.4)
# ==============================================================================
ASSET = "BTC"
BASE_THRESHOLD = 0.05      
MAX_SPREAD = 0.08          # Tightened from 0.20 to prevent -15% open
MAX_BET_SIZE = 5.0
TAKE_PROFIT_PCT = 0.25     
STOP_LOSS_PCT = 0.25       
CLOSE_BUFFER_SECONDS = 90  
# ==============================================================================

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
DATA_API_POSITIONS = "https://data-api.polymarket.com/positions"
BINANCE_URL = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=15"

STATE_DIR = os.getenv("STATE_DIR", "/data")
STATE_PATH = os.path.join(STATE_DIR, "pm_state.json")
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

def ensure_lock():
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

def clear_state():
    try:
        if os.path.exists(STATE_PATH): os.remove(STATE_PATH)
    except: pass

def init_client():
    pk = get_env("PRIVATE_KEY")
    funder = get_env("POLYGON_ADDRESS")
    sig_type = int(os.getenv("SIGNATURE_TYPE", "1"))
    
    print(f"🔐 Auth Init (Type {sig_type}) ...", flush=True)
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=sig_type, funder=funder)
    client.set_api_creds(client.create_or_derive_api_creds())
    return client, funder

def compute_binance_trend():
    try:
        r = SESSION.get(BINANCE_URL, timeout=5)
        data = safe_json(r)
        if not data or len(data) < 15: return None
        
        old_close = float(data[0][4])
        current_close = float(data[-1][4])
        return ((current_close - old_close) / old_close) * 100.0
    except: return None

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
    
    required_mom = BASE_THRESHOLD if spread < 0.05 else BASE_THRESHOLD * 2.0
    if abs(momentum) < required_mom:
        raise RuntimeError(f"Momentum {abs(momentum):.3f}% too weak to justify spread {spread:.2f}.")

    if ask > 0.80: 
        raise RuntimeError("Price too high (Bad EV).")

    # Removed the +0.01 penalty to get better entries
    price = min(ask, 0.80)
    size = round(dollars / price, 4) 
    
    print(f"🚀 TAKING CALCULATED RISK: Buying {size} shares @ {price}...", flush=True)
    resp = client.create_and_post_order(OrderArgs(price=price, size=size, side=BUY, token_id=token_id))
    return resp.get("orderID")

def monitor_and_autoclose(client: ClobClient, user: str, token_id: str, end_time: datetime):
    print("📊 MONITORING 15m POSITION...", flush=True)
    target_time = end_time - timedelta(seconds=CLOSE_BUFFER_SECONDS)
    not_found_count = 0 
    
    while True:
        try:
            r = SESSION.get(DATA_API_POSITIONS, params={"user": user}, timeout=8)
            pos_list = safe_json(r) or []
            pos = next((p for p in pos_list if str(p.get("asset")) == str(token_id) and float(p.get("size") or 0) > 0), None)
            
            if not pos:
                not_found_count += 1
                if not_found_count <= 10:  
                    time.sleep(2)
                    continue
                print("✅ Position closed or resolved.", flush=True)
                clear_state()
                return
            else:
                not_found_count = 0  
            
            pct_pnl = float(pos.get("percentPnl") or 0.0)
            print(f"⏱️ PnL: {pct_pnl:+.1f}%", flush=True)

            trigger = None
            if pct_pnl >= (TAKE_PROFIT_PCT * 100): trigger = "TAKE PROFIT"
            elif pct_pnl <= -(STOP_LOSS_PCT * 100): trigger = "STOP LOSS"
            elif utc_now() >= target_time: trigger = "TIME LIMIT"

            if trigger:
                print(f"🧾 EXITING POSITION: {trigger}", flush=True)
                client.create_and_post_order(OrderArgs(price=0.01, size=float(pos['size']), side=SELL, token_id=token_id))
                clear_state()
                return
        except Exception as e: 
            pass
        time.sleep(2)

def run(live: bool):
    ensure_lock()
    client, user = init_client()
    now = utc_now()
    ws = round_to_15m(now)
    
    mom = compute_binance_trend()
    if mom is None: 
        print("⚠️ Could not fetch Binance data.", flush=True)
        return
        
    print(f"🕒 Window: {ws.strftime('%H:%M')} UTC | 📈 Binance Trend (15m): {mom:+.3f}%", flush=True)

    tokens = get_market_tokens(int(ws.timestamp()))
    if not tokens: 
        print("⚠️ 15m Market not found. Waiting for next window...", flush=True)
        return
        
    print(f"✅ Market Found: {tokens['slug']}", flush=True)
    token_id = tokens["yes"] if mom > 0 else tokens["no"]
    
    if live:
        try:
            place_buy(client, token_id, MAX_BET_SIZE, mom)
            monitor_and_autoclose(client, user, token_id, ws + timedelta(minutes=15))
        except Exception as e: 
            print(f"❌ {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    run(parser.parse_args().live)
