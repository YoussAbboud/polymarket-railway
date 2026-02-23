#!/usr/bin/env python3
"""
Railway-ready Direct Polymarket Bot (v16.1 - The Discord Dispatcher).
- DISCORD ALERTS: Pushes live notifications to a Discord Webhook when positions open and close.
- CLEAN ENGINE: Retains the stable v16.0 sell logic, exact size matching, and hard time kill.
"""

import os, sys, json, time, argparse, atexit
from datetime import datetime, timezone, timedelta
import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, BalanceAllowanceParams, AssetType
from py_clob_client.order_builder.constants import BUY, SELL

# ==============================================================================
# 🎯 STRATEGY SETTINGS (v16.1)
# ==============================================================================
ASSET = "BTC"
BASE_THRESHOLD = 0.04      
MAX_SPREAD = 0.10          
MAX_BET_SIZE = 5.0
TAKE_PROFIT_PCT = 0.18     
STOP_LOSS_PCT = 0.40       
CLOSE_BUFFER_SECONDS = 120 
MAX_TRADES_PER_WINDOW = 3  
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
        requests.post(webhook_url, json
