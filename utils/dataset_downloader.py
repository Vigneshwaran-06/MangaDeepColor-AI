import os, zipfile, requests
from tqdm import tqdm

def download_dataset(url, output_path):
    local_filename = os.path.join(output_path, "dataset.zip")
    os.makedirs(output_path, exist_ok=True)

    with requests.get(url, stream=True) as r:
        total = int(r.headers.get("content-length", 0))
        with open(local_filename, "wb") as f, tqdm(
            desc="Downloading", total=total, unit="B", unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(output_path)

    os.remove(local_filename)
