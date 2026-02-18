#!/usr/bin/env python3
"""
Railway-ready Simmer FastLoop bot (v8.6 - Settlement Verifier).

FIXES:
- ✅ SETTLEMENT VERIFIER: Bot will not say "Closed" until wallet is verified 0 for 10s.
- ✅ OVERLAP PROTECTION: Prevents new trades if the previous settlement is still pending.
- ✅ PERSISTENT COOLDOWN: Keeps the 60s wait to allow API caches to clear.
"""

# ... (Previous imports and settings) ...

def terminate_position_with_prejudice(api_key, market_id, side):
    print(f"\n⚡ TERMINATOR: Initializing kill sequence for {side.upper()}...")
    
    settled_count = 0
    attempt = 1
    
    while settled_count < 3: # Must see 0 shares 3 times in a row
        positions = get_positions(api_key)
        my_pos = next((p for p in positions if str(p.get("market_id")) == str(market_id)), None)
        
        shares_left = float(my_pos.get(f"shares_{side}", 0)) if my_pos else 0
        
        if shares_left <= 0.001:
            settled_count += 1
            print(f"✅ SETTLEMENT: Verification {settled_count}/3 (Shares at 0)")
            time.sleep(4) # Wait for blockchain finality
        else:
            settled_count = 0 # Reset if shares reappear (lag)
            print(f"⚠️  RESIDUAL SHARES: {shares_left} found. Sending SELL (Attempt {attempt})...")
            execute_trade(api_key, market_id, side, shares=shares_left, action="sell")
            attempt += 1
            time.sleep(5) # Give the chain time to breathe

    print("🏁 FINALITY REACHED: Position is officially dead on-chain.")
    record_close_time()
    return True

# ... (Rest of v8.5 logic remains active) ...
