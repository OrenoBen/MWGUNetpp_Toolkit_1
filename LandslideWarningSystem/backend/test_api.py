import requests
import time
import os
import h5py
import numpy as np

# Configuration
BASE_URL = "http://localhost:5000/api/analysis"
TEST_FILE_PATH = "test_image.h5"

def create_dummy_h5():
    """Create a dummy H5 file for testing if one doesn't exist"""
    if not os.path.exists(TEST_FILE_PATH):
        print("Creating dummy H5 file...")
        with h5py.File(TEST_FILE_PATH, 'w') as f:
            # Create random data (14 channels, 128x128)
            # Similar to Landslide4Sense format
            data = np.random.rand(128, 128, 14).astype(np.float32)
            f.create_dataset('img', data=data)
            print(f"Created {TEST_FILE_PATH}")

def test_workflow():
    create_dummy_h5()

    # 1. Upload Image
    print("\n[1] Uploading Image...")
    try:
        with open(TEST_FILE_PATH, 'rb') as f:
            files = {'file': (TEST_FILE_PATH, f, 'application/octet-stream')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            
        if response.status_code != 201:
            print(f"Upload failed: {response.text}")
            return
            
        upload_data = response.json()
        image_id = upload_data['image_id']
        print(f"Upload Success! Image ID: {image_id}")
    except Exception as e:
        print(f"Upload Error: {e}")
        return

    # 2. Start Prediction
    print("\n[2] Starting Prediction...")
    try:
        response = requests.post(f"{BASE_URL}/predict", json={'image_id': image_id})
        if response.status_code != 202:
            print(f"Prediction failed: {response.text}")
            return
            
        task_data = response.json()
        task_id = task_data['task_id']
        print(f"Prediction Started! Task ID: {task_id}")
    except Exception as e:
        print(f"Prediction Error: {e}")
        return

    # 3. Poll Task Status
    print("\n[3] Polling Task Status...")
    while True:
        response = requests.get(f"{BASE_URL}/tasks/{task_id}")
        if response.status_code != 200:
            print(f"Get Status failed: {response.text}")
            break
            
        status_data = response.json()
        status = status_data['status']
        print(f"Task Status: {status}")
        
        if status in ['completed', 'failed']:
            break
            
        time.sleep(1)

    # 4. Get Results
    if status == 'completed':
        print("\n[4] Getting Results...")
        response = requests.get(f"{BASE_URL}/results/{task_id}")
        if response.status_code == 200:
            results = response.json()
            print(f"Success! Found {len(results['features'])} features.")
            print("First feature (snippet):", str(results['features'][0])[:100] if results['features'] else "None")
        else:
            print(f"Get Results failed: {response.text}")
    else:
        print("\nTask failed, skipping result retrieval.")

if __name__ == "__main__":
    # Ensure backend is running before executing this
    try:
        # Simple health check
        requests.get("http://localhost:5000/")
        test_workflow()
    except requests.exceptions.ConnectionError:
        print("Error: Backend is not running. Please start 'python app.py' in backend directory first.")
