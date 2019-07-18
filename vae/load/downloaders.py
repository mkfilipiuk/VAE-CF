import os
from os.path import basename, normpath
import urllib.request
import requests
import tarfile
import tempfile
import zipfile
from tqdm import tqdm
import itertools

from glob import glob
from vae.config import DOWNLOAD, DOWNLOAD_GOOGLE_DRIVE_ID, DOWNLOAD_URL, DATASETS_DIR, ML_20M, ML_20M_ALT, LASTFM, NETFLIX, \
    PINTEREST, LOG


def download_lastfm():
    filepath = os.path.join(DATASETS_DIR, 'lastfm-dataset-360K.tar.gz')
    if not glob(DOWNLOAD[LASTFM]):
        download_file(DOWNLOAD_URL[LASTFM], filepath)

    LOG.info("Extracting")
    extract_file(filepath, DATASETS_DIR)
    os.rename(
        os.path.join(DATASETS_DIR, 'lastfm-dataset-360K'),
        os.path.join(DATASETS_DIR, 'lastfm'))


def download_movielens():
    filepath = os.path.join(DATASETS_DIR, ML_20M_ALT + '.zip')
    if not glob(filepath):
        download_file(DOWNLOAD_URL[ML_20M], filepath)

    LOG.info("Extracting")
    extract_file(filepath, DATASETS_DIR)


def download_netflix():
    filepath = os.path.join(DATASETS_DIR, NETFLIX + '.tar.gz')
    if not glob(filepath):
        download_file(DOWNLOAD_URL[NETFLIX], filepath)

    LOG.info("Extracting 1/2")
    extract_file(filepath, tempfile.gettempdir())
    LOG.info("Extracting 2/2")
    extract_file(
        os.path.join(tempfile.gettempdir(), 'download', 'training_set.tar'),
        DATASETS_DIR)
    os.rename(os.path.join(DATASETS_DIR, 'training_set'), DOWNLOAD[NETFLIX])


def download_pinterest():
    filepath = os.path.join(DATASETS_DIR, PINTEREST + '.zip')
    if not glob(filepath):
        download_file_from_google_drive(DOWNLOAD_GOOGLE_DRIVE_ID[PINTEREST], filepath)
    LOG.info("Extracting")
    extract_file(filepath, DATASETS_DIR)
    os.rename(
        os.path.join(DATASETS_DIR, 'pinterest_iccv'), DOWNLOAD[PINTEREST])


def download_file(url, filename):
    if not os.path.isdir(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

    u = urllib.request.urlopen(url)
    with open(filename, 'wb') as f:
        meta = u.info()
        if (meta.get_all("Content-Length")):
            file_size = int(meta.get_all("Content-Length")[0])
            pbar = tqdm(
                total=file_size,
                desc=basename(normpath(filename)),
                unit='B',
                unit_scale=True)

            file_size_dl = 0
            block_sz = 8192
            while True:
                buff = u.read(block_sz)
                if not buff:
                    break
                pbar.update(len(buff))
                file_size_dl += len(buff)
                f.write(buff)
            pbar.close()
        else:
            LOG.warning("No content length information")
            file_size_dl = 0
            block_sz = 8192
            for cyc in itertools.cycle('/–\\|'):
                buff = u.read(block_sz)
                if not buff:
                    break
                print(cyc, end='\r')
                file_size_dl += len(buff)
                f.write(buff)


def download_file_from_google_drive(file_id: str, filename):
    if not os.path.isdir(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, filename):
        CHUNK_SIZE = 8192

        if 'Content-Length' in response.headers.keys():
            filesize = response.headers['Content-Length']
            pbar = tqdm(
                total=filesize,
                desc=basename(normpath(filename)),
                unit='B',
                unit_scale=True)

            with open(filename, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

            pbar.close()
        else:
            LOG.warning("No content length information")
            with open(filename, "wb") as f:
                for chunk, cyc in zip(
                        response.iter_content(CHUNK_SIZE),
                        itertools.cycle('/–\\|')):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        print(cyc, end='\r')

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, filename)


def extract_file(path, to_directory):
    """
    Extract file
    :param path: Path to compressed file
    :param to_directory: Directory that is going to store extracte files
    """
    if (path.endswith("tar.gz")):
        tar = tarfile.open(path, "r:gz")
        tar.extractall(path=to_directory)
        tar.close()
    elif (path.endswith("tar")):
        tar = tarfile.open(path, "r:")
        tar.extractall(path=to_directory)
        tar.close()
    elif (path.endswith("zip")):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(to_directory)
    else:
        raise Exception(
            "Could not extract {} as no appropriate extractor is found".format(path))
