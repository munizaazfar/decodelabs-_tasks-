import os
import requests

FILES = {
    "MobileNetSSD_deploy.prototxt": "https://raw.githubusercontent.com/nikmart/pi-object-detection/master/MobileNetSSD_deploy.prototxt.txt",
    "MobileNetSSD_deploy.caffemodel": "https://huggingface.co/spaces/Imran606/cds/resolve/main/MobileNetSSD_deploy.caffemodel"
}

print("--- DOWNLOADING MOBILENET-SSD WEIGHTS ---")
for filename, url in FILES.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✓ Saved {filename}")
    else:
        print(f"✓ {filename} already exists.")