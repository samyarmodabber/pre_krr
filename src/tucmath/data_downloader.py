import os
import requests
import gzip
from tqdm import tqdm


def download_dataset(directory, url, compressed_file_name, uncompressed_file_name):
    if not os.path.exists(directory):
        os.makedirs(directory)

    compressed_file_path = os.path.join(directory, compressed_file_name)
    uncompressed_file_path = os.path.join(directory, uncompressed_file_name)

    # Check if .csv file already exists
    if os.path.exists(uncompressed_file_path):
        print(f"The file {uncompressed_file_path} already exists.")
    else:
        # Download the file with a progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte

        with open(compressed_file_path, 'wb') as file, tqdm(
                desc="Downloading SUSY.csv.gz",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)

        print(
            f"Downloaded {compressed_file_name} successfully to {compressed_file_path}.")

        # Unzip the file with a progress bar
        with gzip.open(compressed_file_path, 'rb') as f_in:
            with open(uncompressed_file_path, 'wb') as f_out, tqdm(
                desc=f"Extracting {uncompressed_file_name}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in iter(lambda: f_in.read(block_size), b''):
                    bar.update(len(data))
                    f_out.write(data)

        print(
            f"Extracted {uncompressed_file_name} successfully to {uncompressed_file_path}.")

        # Delete the compressed file
        os.remove(compressed_file_path)
        print(f"Deleted {compressed_file_path}.")
