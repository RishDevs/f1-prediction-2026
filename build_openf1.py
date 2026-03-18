import urllib.request
import json
import ssl
import pandas as pd
from datetime import datetime

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch_json(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx) as response:
        return json.loads(response.read().decode('utf-8'))

print("Fetching sessions...")
try:
    sessions = fetch_json("https://api.openf1.org/v1/sessions?session_name=Practice%201")
    now_iso = datetime.utcnow().isoformat()
    past_sessions = [s for s in sessions if s['date_start'] < now_iso]
    latest_session = sorted(past_sessions, key=lambda x: x['date_start'], reverse=True)[0]
    session_key = latest_session['session_key']
    
    print(f"Chosen session: {latest_session['circuit_short_name']} {latest_session['year']} (Key: {session_key})")
    
    laps = fetch_json(f"https://api.openf1.org/v1/laps?session_key={session_key}")
    laps_df = pd.DataFrame(laps)
    
    if len(laps_df) == 0:
        print("No laps found!")
    else:
        valid_laps = laps_df.dropna(subset=['lap_duration'])
        best_laps = valid_laps.groupby('driver_number')['lap_duration'].min().reset_index()
        
        drivers = fetch_json(f"https://api.openf1.org/v1/drivers?session_key={session_key}")
        drivers_df = pd.DataFrame(drivers)
        
        merged = pd.merge(best_laps, drivers_df[['driver_number', 'full_name']], on='driver_number')
        
        fp1_results = {}
        for _, row in merged.iterrows():
            fp1_results[row['full_name']] = row['lap_duration']
            
        print("Scraped FP1 Results:")
        for k, v in fp1_results.items():
            print(f"  {k}: {v}")
except Exception as e:
    import traceback
    traceback.print_exc()
