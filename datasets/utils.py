import os
import tarfile

import wget

from datasets.url import URLS


def check_file_exists(path):
    if not path:
        return False
    return os.path.isfile(path)


def check_dir_exists(path):
    if not path:
        return False
    return os.path.isdir(path)


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def download_file(name, path):
    url = URLS.get(name, None)
    if not url:
        raise ValueError("url not found in url.py")
    d_path = os.path.join(path, name)
    if not check_file_exists(d_path):
        print(f"Downloading {name} to {d_path}...")
        wget.download(url, d_path)
    else:
        print("File already downloaded...")
    return d_path


def extract_file(file, path):
    print("Extracting...")
    with tarfile.open(file) as archive:
        archive.extractall(path)
