import requests
import base64
import os

URL = "http://127.0.0.1:5000/detect"
img_path = "testImages/1.jpg"

if not os.path.exists(img_path):
    print(f"Error: {img_path} not found.")
    exit(1)

try:
    with open(img_path, "rb") as f:
        files = {"file": f}
        print(f"Sending request to {URL}...")
        response = requests.post(URL, files=files)
    
    if response.status_code == 200:
        data = response.json()
        print("Success!")
        print(f"Detections found: {len(data['detections'])}")
        for det in data['detections']:
            print(f" - {det['label']} ({det['confidence']})")
        
        # Save output image
        if 'image' in data:
            img_data = base64.b64decode(data['image'])
            with open("test_output.jpg", "wb") as out:
                out.write(img_data)
            print("Output saved to test_output.jpg")
    else:
        print(f"Error: Status Code {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Exception: {e}")
