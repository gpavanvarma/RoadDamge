import requests
import json

BASE_URL = "http://127.0.0.1:5000"
SESSION = requests.Session()

def register_and_login():
    # 1. Sign Up
    print("1. Signing up...")
    r = SESSION.post(f"{BASE_URL}/signup", data={"username": "testuser", "password": "password123"})
    print(f"   Status: {r.status_code} (Expected redirect to dashboard)")
    
    # 2. Login
    print("2. Logging in...")
    r = SESSION.post(f"{BASE_URL}/login", data={"username": "testuser", "password": "password123"})
    if r.status_code == 200 and "Dashboard" in r.text:
        print("   Success: Logged in and reached Dashboard.")
    else:
        print(f"   Failed: {r.status_code}")

def check_protected_routes():
    # 3. Check Metrics API
    print("3. Checking Analytics API...")
    r = SESSION.get(f"{BASE_URL}/api/metrics")
    if r.status_code == 200:
        data = r.json()
        print(f"   Success: Got metrics for {list(data['comparison'].keys())}")
    else:
        print(f"   Failed: {r.status_code}")

    # 4. Check History API
    print("4. Checking History API...")
    r = SESSION.get(f"{BASE_URL}/api/history")
    if r.status_code == 200:
        data = r.json()
        print(f"   Success: Got history for {list(data.keys())}")
    else:
        print(f"   Failed: {r.status_code}")

if __name__ == "__main__":
    try:
        register_and_login()
        check_protected_routes()
    except Exception as e:
        print(f"Error: {e}")
