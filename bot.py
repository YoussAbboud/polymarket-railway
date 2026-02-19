#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (v11.0 - The Sniper).

Key fixes:
- BINANCE DATA: Replaces laggy CoinGecko with Binance 1m candles.
- SPREAD GUARD: Refuses to trade if spread is > 15c.
- EV CAP: Refuses to buy shares priced > 65c.
- DIAMOND HANDS: Removes the Stop Loss to prevent spread-whipsaws.
"""

import os, sys, json, time, argparse, atexit
from datetime import datetime, timezone, timedelta
import requests

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

# ==============================================================================
# STRATEGY SETTINGS (v11.0)
# ==============================================================================
ASSET = "BTC"
MIN_MOMENTUM_PCT = 0.05    # Lowered threshold for tighter 5m live data

# SAFETY SETTINGS
MAX_BET_SIZE = 5.0
TAKE_PROFIT_PCT = 0.25     # 25% TP to cover taker fees
CLOSE_BUFFER_SECONDS = 60  
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

# ------------------------ helpers ------------------------

def die(msg: str):
    print(msg, flush=True)
    sys.exit(1)

def get_env(key: str, default=None):
    val = os.environ.get(key, default)
    if val is None or val == "":
        die(f"❌ ERROR: Missing Env Variable: {key}")
    return val

def utc_now():
    return datetime.now(timezone.utc)

def round_to_5m(ts: datetime) -> datetime:
    return ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)

def slug_for_window(asset: str, window_start: datetime) -> str:
    return f"{asset.lower()}-updown-5m-{int(window_start.timestamp())}"

def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return None

def ensure_lock():
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(LOCK_PATH):
        try:
            if time.time() - os.path.getmtime(LOCK_PATH) > 5 * 60:
                os.remove(LOCK_PATH)
        except Exception:
            pass
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        die("⚠️ Another bot instance is running. Exiting.")

    def _cleanup():
        try:
            if os.path.exists(LOCK_PATH): os.remove(LOCK_PATH)
        except Exception: pass
    atexit.register(_cleanup)

def load_state():
    if not os.path.exists(STATE_PATH): return None
    try:
        with open(STATE_PATH, "r") as f: return json.load(f)
    except Exception: return None

def save_state(state: dict):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f: json.dump(state, f)
    os.replace(tmp, STATE_PATH)

def clear_state():
    try:
        if os.path.exists(STATE_PATH): os.remove(STATE_PATH)
    except Exception: pass

# ------------------------ polymarket / APIs ------------------------

def init_client():
    pk = get_env("PRIVATE_KEY")
    funder = get_env("POLYGON_ADDRESS")  
    sig_type = int(os.getenv("SIGNATURE_TYPE", "1"))  

    print(f"🔐 Auth Init (Type {sig_type}) ...", flush=True)
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=sig_type, funder=funder)
    client.set_api_creds(client.create_or_derive_api_creds())
    return client, funder, sig_type

def get_market_tokens(slug: str):
    try:
        r = SESSION.get(f"{GAMMA_EVENTS_URL}?slug={slug}", timeout=8)
        data = safe_json(r)
        if not data: return None
        m = data[0].get("markets", [])[0]
        clob_ids = json.loads(m.get("clobTokenIds", "[]"))
        return {"yes": str(clob_ids[0]), "no": str(clob_ids[1])}
    except Exception:
        return None

def compute_binance_trend():
    """Fetches real-time 1m candles from Binance to determine exact momentum."""
    try:
        r = SESSION.get(BINANCE_URL, timeout=5)
        data = safe_json(r)
        if not data or len(data) < 5: return None
        
        old_close = float(data[0][4])
        current_close = float(data[-1][4])
        return ((current_close - old_close) / old_close) * 100.0
    except Exception:
        return None

def data_api_positions(user: str, limit: int = 100):
    try:
        params = {"user": user, "sizeThreshold": 0, "limit": limit, "offset": 0}
        r = SESSION.get(DATA_API_POSITIONS, params=params, timeout=8)
        return safe_json(r) or []
    except Exception:
        return []

def find_position_for_asset(user: str, asset_token_id: str):
    for p in data_api_positions(user):
        if str(p.get("asset")) == str(asset_token_id) and float(p.get("size") or 0) > 0:
            return p
    return None

def find_any_open_btc5m_positions(user: str):
    prefix = f"{ASSET.lower()}-updown-5m-"
    out = []
    now_ts = int(time.time())
    for p in data_api_positions(user):
        slug = (p.get("slug") or "")
        if float(p.get("size") or 0) > 0 and slug.startswith(prefix):
            try:
                if now_ts - int(slug.split("-")[-1]) > 900: continue
            except Exception: pass
            out.append(p)
    return out

def best_bid_ask(client: ClobClient, token_id: str):
    try:
        ob = client.get_order_book(token_id)
        bid = float(ob.bids[0].price) if ob and ob.bids else None
        ask = float(ob.asks[0].price) if ob and ob.asks else None
        return bid, ask
    except Exception:
        return None, None

# ------------------------ trading ------------------------

def place_buy(client: ClobClient, token_id: str, dollars: float):
    bid, ask = best_bid_ask(client, token_id)
    if ask is None or bid is None:
        raise RuntimeError("Orderbook missing liquidity.")

    # SPREAD GUARD: Don't get chopped up
    spread = ask - bid
    if spread > 0.15:
        raise RuntimeError(f"Spread too wide ({spread:.2f}c). Skipping to protect capital.")

    # EV GUARD: Don't buy the top
    if ask > 0.65:
        raise RuntimeError(f"Ask price too high ({ask:.2f}c). Terrible EV. Skipping.")

    price = min(round(ask + 0.01, 2), 0.65)
    size = round(dollars / price, 4) 
    
    print(f"🚀 BUYING: {size:.4f} shares @ {price:.2f}...", flush=True)
    resp = client.create_and_post_order(OrderArgs(price=price, size=size, side=BUY, token_id=token_id))
    oid = resp.get("orderID") if isinstance(resp, dict) else None
    if not oid: raise RuntimeError(f"Buy failed: {resp}")
    
    print(f"✅ ORDER PLACED: {oid}", flush=True)
    return oid, price, size

def place_sell_marketable(client: ClobClient, token_id: str, size: float):
    print(f"🧾 SELLING (marketable): {size:.4f}", flush=True)
    resp = client.create_and_post_order(OrderArgs(price=0.01, size=size, side=SELL, token_id=token_id))
    if isinstance(resp, dict) and resp.get("orderID"):
        print(f"✅ SELL ORDER PLACED: {resp.get('orderID')}", flush=True)
    return resp

def monitor_and_autoclose(client: ClobClient, user: str, token_id: str, end_time: datetime, tp_pct: float, buffer_s: int):
    print("📊 MONITORING... (PnL 1s + AUTO-CLOSE)", flush=True)
    target_time = end_time - timedelta(seconds=buffer_s)
    tp = tp_pct * 100.0
    not_found_count = 0  

    while True:
        now = utc_now()
        pos = find_position_for_asset(user, token_id)
        
        if not pos:
            not_found_count += 1
            if not_found_count <= 20:
                time.sleep(1)
                continue
            print("✅ Position closed/resolved.", flush=True)
            clear_state()
            return
        else:
            not_found_count = 0  

        pct_pnl = float(pos.get("percentPnl") or 0.0)
        cash_pnl = float(pos.get("cashPnl") or 0.0)
        
        print(f"⏱️ Size: {float(pos.get('size') or 0):.2f} | PnL: {pct_pnl:+.1f}% (${cash_pnl:+.2f})", flush=True)

        trigger = None
        if pct_pnl >= tp: trigger = "TAKE PROFIT"
        elif now >= target_time: trigger = "TIME BUFFER"

        if trigger:
            print(f"🧾 CLOSE TRIGGER: {trigger}", flush=True)
            for _ in range(10):
                try:
                    place_sell_marketable(client, token_id, float(pos.get("size") or 0))
                    break 
                except Exception as e:
                    if "does not exist" in str(e).lower():
                        clear_state()
                        return
            time.sleep(1.5)
            if not find_position_for_asset(user, token_id):
                clear_state()
                return
        time.sleep(1) 

def run(live: bool):
    ensure_lock()
    client, user, sig_type = init_client()

    now = utc_now()
    ws = round_to_5m(now)
    slug = slug_for_window(ASSET, ws)
    end_time = ws + timedelta(minutes=5)

    open_positions = find_any_open_btc5m_positions(user)
    if open_positions:
        tok = str(open_positions[0].get("asset"))
        print(f"🧠 Existing position found. Monitoring...", flush=True)
        monitor_and_autoclose(client, user, tok, end_time, TAKE_PROFIT_PCT, CLOSE_BUFFER_SECONDS)
        return

    tokens = get_market_tokens(slug)
    if not tokens: return

    mom = compute_binance_trend()
    if mom is None: return

    print(f"📈 Binance Trend (5m): {mom:+.3f}% | threshold={MIN_MOMENTUM_PCT:.3f}%", flush=True)
    if abs(mom) < MIN_MOMENTUM_PCT: return

    token_id = tokens["yes"] if mom > 0 else tokens["no"]
    
    if not live: return

    try:
        place_buy(client, token_id, MAX_BET_SIZE)
        save_state({"token_id": token_id})
        monitor_and_autoclose(client, user, token_id, end_time, TAKE_PROFIT_PCT, CLOSE_BUFFER_SECONDS)
    except Exception as e:
        print(f"❌ Trade Failed: {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    run(parser.parse_args().live)
