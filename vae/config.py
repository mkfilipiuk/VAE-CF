import os.path
import logging

# root = location of this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# downloaded data
DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')

# downloaded and processed data
BIN_DATA_DIR = os.path.join(DATA_DIR, 'binary')

# the different datasets
ML_20M = 'ml-20m'
ML_20M_ALT = 'ml-20m_alt'
NETFLIX = 'netflix'
LASTFM = 'lastfm'
PINTEREST = 'pinterest'
DATASETS = [ML_20M, NETFLIX, LASTFM, PINTEREST, ML_20M_ALT]

# download urls to different datasets
DOWNLOAD_URL = {
    ML_20M: 'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
    NETFLIX: 'https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz',
    LASTFM: 'http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz',
}

# id of google drive file
DOWNLOAD_GOOGLE_DRIVE_ID = {
    PINTEREST: '0B0l8Lmmrs5A_REZXanM3dTN4Y28'
}

# dic to downloads
DOWNLOAD = {d: os.path.join(DATASETS_DIR, d) for d in DATASETS}

# dic to binary files
BIN_DATA = {d: os.path.join(BIN_DATA_DIR, 'bin_{}_matrix.npz'.format(d)) for d in DATASETS}

LOG = logging.Logger("VAE")
