# PKRR Package

## Download Datasets

Run the following block in example01:

```py
from data_downloader import download_dataset
# URL of the SUSY dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz'

# Directory to save the dataset
directory = 'data'

# Path to save the downloaded file
compressed_file_name = 'SUSY.csv.gz'
uncompressed_file_name = 'SUSY.csv'

download_dataset(directory, url, compressed_file_name, uncompressed_file_name)
```
