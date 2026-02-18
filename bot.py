#!/usr/bin/env python3
"""
Railway-ready Wallet Detective (v10.5).

PURPOSE:
- 🕵️ REVEAL THE TRUTH: Tells you exactly which wallet address your Private Key unlocks.
- 🛑 STOP TRADING: No trades will be attempted. This is purely for debugging the "Invalid Signature".
"""

import os, sys, time
from eth_account import Account
from py_clob_client.client import ClobClient

# ==============================================================================
# ⚙️ CONFIG
# ==============================================================================
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

def get_env(key):
    val = os.environ.get(key)
    if not val:
        print(f"❌ ERROR: Missing Env Variable: {key}")
        sys.exit(1)
    return val

def run_detective():
    print("🕵️ WALLET DETECTIVE STARTED...")
    print("------------------------------------------------------")
    
    # 1. Get the Key
    pk = get_env("PRIVATE_KEY")
    target_proxy = get_env("POLYGON_ADDRESS")
    
    # 2. Derive the Address from the Key
    try:
        account = Account.from_key(pk)
        signer_address = account.address
        print(f"🔑 YOUR PRIVATE KEY UNLOCKS ADDRESS:  {signer_address}")
    except Exception as e:
        print(f"❌ INVALID PRIVATE KEY FORMAT: {e}")
        return

    # 3. Compare with the Target
    print(f"🎯 YOU ARE TRYING TO SPEND FROM:      {target_proxy}")
    print("------------------------------------------------------")
    
    # 4. Analysis
    if signer_address.lower() == target_proxy.lower():
        print("✅ MATCH! This key directly controls this wallet (EOA).")
        print("   -> You should use Signature Type 1.")
    else:
        print("⚠️ MISMATCH (Normal for Magic Link/Proxy Wallets).")
        print("   -> This means 'Address A' (Signer) is trying to control 'Address B' (Proxy).")
        print("   -> For this to work, Polymarket must 'know' that A is authorized for B.")
        
    # 5. Test Auth
    print("\n🔐 TESTING AUTHENTICATION WITH POLYMARKET...")
    
    # Attempt Type 1
    try:
        print("   [1] Testing Type 1 (Standard)... ", end="")
        c1 = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=1)
        c1.set_api_creds(c1.create_or_derive_api_creds())
        print("OK ✅")
    except Exception as e:
        print(f"FAIL ❌ ({e})")

    # Attempt Type 2
    try:
        print("   [2] Testing Type 2 (Proxy)...    ", end="")
        c2 = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=2, funder=target_proxy)
        c2.set_api_creds(c2.create_or_derive_api_creds())
        print("OK ✅")
    except Exception as e:
        print(f"FAIL ❌ ({e})")

    print("\n------------------------------------------------------")
    print("💡 DIAGNOSIS:")
    print("If both AUTH tests failed, your Private Key is wrong for this account.")
    print("If one worked, we will hardcode that method in the next bot.")
    print("------------------------------------------------------")

    # Keep alive so logs can be read
    while True:
        time.sleep(10)

if __name__ == "__main__":
    run_detective()
