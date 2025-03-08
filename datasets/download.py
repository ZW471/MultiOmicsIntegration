import os
import requests
from tqdm import tqdm
import subprocess

def download_file(url, file_path):
    """
    Downloads a file from the given URL and saves it to the specified file_path.
    A progress bar is shown during the download.
    """
    # Ensure the directory exists. Create it if it doesn't.
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Send a HTTP request to the URL with stream enabled.
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024  # 1 KB

    # Open the file in write-binary mode and create a tqdm progress bar.
    with open(file_path, 'wb') as file, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=os.path.basename(file_path),
            ncols=80
    ) as progress:
        for data in response.iter_content(chunk_size):
            file.write(data)
            progress.update(len(data))



if __name__ == "__main__":
    # https://zenodo.org/records/6348128
    # Replace the URL with your dataset's URL.
    # select from 'LUNG-CITE', 'BM-CITE', 'PBMC-DOGMA', 'PBMC-Multiome', 'PBMC-TEA', 'Skin-SHARE'
    file_names = ['PBMC-DOGMA', 'PBMC-TEA']

    for file_name in file_names:
        url = f"https://zenodo.org/records/6348128/files/{file_name}.Rds?download=1"
        # Specify the full download path. For example, here it will save to a folder named 'downloads'
        file_path = f"data/{file_name}.Rds"

        download_file(url, file_path)
        print("Download completed!")

        print("Converting to python")
        subprocess.run(["Rscript", "mojitoo_data_to_py.R", file_name], check=True)


