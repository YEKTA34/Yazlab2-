import os
import urllib.request
from config import SKAB_DIR, BATADAL_DIR

def create_dirs():
    os.makedirs(os.path.join(SKAB_DIR, "valve1"), exist_ok=True)
    os.makedirs(os.path.join(SKAB_DIR, "valve2"), exist_ok=True)
    os.makedirs(BATADAL_DIR, exist_ok=True)

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        return
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            with open(dest_path, 'wb') as f:
                f.write(response.read())
    except Exception as e:
        pass

def main():
    create_dirs()
    
    for i in range(1, 17):
        url = f"https://raw.githubusercontent.com/waico/SKAB/master/data/valve1/{i}.csv"
        dest = os.path.join(SKAB_DIR, "valve1", f"{i}.csv")
        download_file(url, dest)
        
    for i in range(1, 5):
        url = f"https://raw.githubusercontent.com/waico/SKAB/master/data/valve2/{i}.csv"
        dest = os.path.join(SKAB_DIR, "valve2", f"{i}.csv")
        download_file(url, dest)
        
    batadal_url = "http://www.batadal.net/data/BATADAL_dataset04.csv"
    batadal_dest = os.path.join(BATADAL_DIR, "BATADAL_dataset04.csv")
    download_file(batadal_url, batadal_dest)

if __name__ == "__main__":
    main()
