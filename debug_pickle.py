import pickle
import os

files = [
    'model/v5_history.pckl',
    'model/v7_history.pckl',
    'model/v8_history.pckl'
]

for fpath in files:
    print(f"Checking {fpath}...")
    if not os.path.exists(fpath):
        print("  - File not found")
        continue
        
    try:
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            print(f"  - Loaded successfully. Type: {type(data)}")
            if isinstance(data, dict):
                print(f"  - Keys: {list(data.keys())}")
                # print(f"  - Content: {data}")
    except Exception as e:
        print(f"  - Error loading pickle: {e}")
