#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (v11.2 - Calculated Risk).
- DYNAMIC SPREAD: Allows up to 0.35c spread if the trend is strong.
- MOMENTUM TIERS: Higher spread requires a stronger Binance signal.
- NO PANIC: Removed the tight stop-loss to let the spread settle.
"""

import os, sys, json, time, argparse, atexit
from datetime import datetime, timezone, timedelta
import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

# ==============================================================================
# 🚀 STRATEGY SETTINGS (v11.2 - THE RISK-TAKER)
# ==============================================================================
ASSET = "BTC"
BASE_THRESHOLD = 0.033      # Start looking at trades here
MAX_SPREAD = 0.33          # Hard cap on spread (allows the ~$1 loss entry)
MAX_BET_SIZE = 5.0
TAKE_PROFIT_PCT = 0.22     
# ==============================================================================

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
DATA_API_POSITIONS = "https://data-api.polymarket.com/positions"
BINANCE_URL = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=5"

STATE_DIR = os.getenv("STATE_DIR", "/data")
STATE_PATH = os.path.join(STATE_DIR, "pm_state.json")
LOCK_PATH = os.path.join(STATE_DIR, "pm_lock")
SESSION = requests.Session()

def die(msg: str): print(msg, flush=True); sys.exit(1)
def get_env(key: str): 
    val = os.environ.get(key)
    if not val: die(f"❌ ERROR: Missing Env Variable: {key}")
    return val

def utc_now(): return datetime.now(timezone.utc)
def round_to_5m(ts: datetime): return ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
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
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
    except FileExistsError: die("⚠️ Duplicate instance. Exiting.")
    def _cleanup():
        try:
            if os.path.exists(LOCK_PATH): os.remove(LOCK_PATH)
        except: pass
    atexit.register(_cleanup)

def init_client():
    pk = get_env("PRIVATE_KEY"); funder = get_env("POLYGON_ADDRESS")
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=1, funder=funder)
    client.set_api_creds(client.create_or_derive_api_creds())
    return client, funder

def compute_binance_trend():
    try:
        r = SESSION.get(BINANCE_URL, timeout=5)
        data = safe_json(r)
        old_close, current_close = float(data[0][4]), float(data[-1][4])
        return ((current_close - old_close) / old_close) * 100.0
    except: return None

def place_buy(client: ClobClient, token_id: str, dollars: float, momentum: float):
    ob = client.get_order_book(token_id)
    if not ob.asks or not ob.bids: raise RuntimeError("Empty book.")
    
    bid, ask = float(ob.bids[0].price), float(ob.asks[0].price)
    spread = ask - bid
    
    # --- DYNAMIC RISK LOGIC ---
    print(f"📊 Market Stats: Ask={ask:.2f} | Bid={bid:.2f} | Spread={spread:.2f}", flush=True)
    
    if spread > MAX_SPREAD:
        raise RuntimeError(f"Spread {spread:.2f} too extreme. Capital protection active.")
    
    # If spread is wide, we need more momentum to justify the entry
    required_mom = BASE_THRESHOLD if spread < 0.15 else BASE_THRESHOLD * 2.5
    
    if abs(momentum) < required_mom:
        raise RuntimeError(f"Momentum {abs(momentum):.3f}% too weak for spread {spread:.2f}.")

    if ask > 0.80: raise RuntimeError("Price too high (Bad EV).")

    price = min(round(ask + 0.01, 2), 0.80)
    size = round(dollars / price, 4) 
    print(f"🚀 TAKING RISK: Buying {size} @ {price}...", flush=True)
    resp = client.create_and_post_order(OrderArgs(price=price, size=size, side=BUY, token_id=token_id))
    return resp.get("orderID")

def monitor_and_autoclose(client: ClobClient, user: str, token_id: str, end_time: datetime):
    print("📊 MONITORING...", flush=True)
    target_time = end_time - timedelta(seconds=60)
    
    while True:
        try:
            r = SESSION.get(DATA_API_POSITIONS, params={"user": user}, timeout=8)
            pos_list = safe_json(r) or []
            pos = next((p for p in pos_list if str(p.get("asset")) == str(token_id) and float(p.get("size") or 0) > 0), None)
            
            if not pos:
                print("✅ Position closed.", flush=True); return
            
            pct_pnl = float(pos.get("percentPnl") or 0.0)
            print(f"⏱️ PnL: {pct_pnl:+.1f}%", flush=True)

            if pct_pnl >= (TAKE_PROFIT_PCT * 100) or utc_now() >= target_time:
                print("🧾 EXITING...", flush=True)
                client.create_and_post_order(OrderArgs(price=0.01, size=float(pos['size']), side=SELL, token_id=token_id))
                return
        except: pass
        time.sleep(2)

def run(live: bool):
    ensure_lock(); client, user = init_client()
    now = utc_now(); ws = round_to_5m(now); slug = f"btc-updown-5m-{int(ws.timestamp())}"
    
    mom = compute_binance_trend()
    print(f"📈 Binance Trend: {mom:+.3f}%", flush=True)
    if mom is None: return

    try:
        r = SESSION.get(f"{GAMMA_EVENTS_URL}?slug={slug}", timeout=8).json()
        clob_ids = json.loads(r[0]['markets'][0]['clobTokenIds'])
        token_id = clob_ids[0] if mom > 0 else clob_ids[1]
        
        if live:
            place_buy(client, token_id, MAX_BET_SIZE, mom)
            monitor_and_autoclose(client, user, token_id, ws + timedelta(minutes=5))
    except Exception as e: print(f"❌ {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--live", "-l", action="store_true")
    run(parser.parse_args().live)
