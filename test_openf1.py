import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch_json(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx) as response:
        return json.loads(response.read().decode())

print("Fetching recent sessions...")
sessions = fetch_json("https://api.openf1.org/v1/sessions?year=2026")
if not sessions:
    print("No 2026 sessions found. Fetching 2025...")
    sessions = fetch_json("https://api.openf1.org/v1/sessions?year=2025")

if sessions:
    print(f"Found {len(sessions)} sessions.")
    latest_session = sessions[-1]
    print("Latest session:", latest_session['session_name'], "at", latest_session['circuit_short_name'], "ID:", latest_session['session_key'])
else:
    print("No sessions found.")
