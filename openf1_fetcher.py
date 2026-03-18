import urllib.request
import json
import ssl
import pandas as pd

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch_json(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx) as response:
        return json.loads(response.read().decode())

print("Fetching latest Practice 1 session...")
sessions = fetch_json("https://api.openf1.org/v1/sessions?session_name=Practice%201")
if not sessions:
    print("No practice sessions found.")
    exit(1)

latest_session = sorted(sessions, key=lambda x: x['date_start'], reverse=True)[0]
session_key = latest_session['session_key']
print("Latest FP1 Session:", latest_session['circuit_short_name'], latest_session['year'], "- Key:", session_key)

print("Fetching lap times...")
laps = fetch_json(f"https://api.openf1.org/v1/laps?session_key={session_key}")
laps_df = pd.DataFrame(laps)

if len(laps_df) == 0:
    print("No laps found for session.")
else:
    # Filter valid laps 
    valid_laps = laps_df.dropna(subset=['lap_duration'])
    best_laps = valid_laps.groupby('driver_number')['lap_duration'].min().reset_index()

    print("Fetching driver details...")
    drivers = fetch_json(f"https://api.openf1.org/v1/drivers?session_key={session_key}")
    drivers_df = pd.DataFrame(drivers)

    merged = pd.merge(best_laps, drivers_df[['driver_number', 'full_name']], on='driver_number')
    print("\nBest Laps in FP1:")
    for _, row in merged.sort_values('lap_duration').iterrows():
        print(f"{row['full_name']}: {row['lap_duration']}")
