import bson
import os
import pandas as pd
import scipy.sparse as sp
from glob import glob
from vae.load.downloaders import download_movielens, download_netflix, download_lastfm
from vae.config import BIN_DATA, BIN_DATA_DIR, ML_20M, ML_20M_ALT, NETFLIX, LASTFM, PINTEREST, DOWNLOAD, LOG
from tqdm import tqdm


def parse_movielens(threshold=4, **kwargs):
    if os.path.isfile(BIN_DATA[ML_20M_ALT]):
        LOG.info("Already processed, skipping.")
        return

    source_file = os.path.join(DOWNLOAD[ML_20M_ALT], "ratings.csv")
    if not glob(source_file):
        download_movielens()

    LOG.info("Parsing movielens.")
    df = pd.read_csv(source_file)
    df.drop('timestamp', axis=1, inplace=True)
    df["rating"] = make_feedback_implicit(df["rating"], threshold)

    map_user_id = {u: i for i, u in enumerate(df.userId.unique())}
    map_movie_id = {m: i for i, m in enumerate(df.movieId.unique())}

    m_sp = sp.csr_matrix(
        (df.rating,
         ([map_user_id[u] for u in df.userId],
          [map_movie_id[m] for m in df.movieId])),
        shape=(len(map_user_id), len(map_movie_id))
    )

    m_sp.eliminate_zeros()
    save_as_npz(m_sp, BIN_DATA[ML_20M_ALT])


def parse_netflix(threshold=3, **kwargs):
    if os.path.isfile(BIN_DATA[NETFLIX]):
        LOG.info("Already processed, skipping.")
        return

    files = glob(os.path.join(DOWNLOAD[NETFLIX], '*'))
    if not files:
        download_netflix()

    LOG.info("Parsing netflix")
    users = get_users(files)
    map_user_id = {u: i for i, u in enumerate(users)}

    csr_rows = []
    csr_columns = []
    csr_data = []

    LOG.info("Parsing netflix, step 2/2")
    for movie_id, file_path in tqdm(enumerate(files)):
        df = pd.read_csv(file_path, names=['User', 'Rating', 'Date'])
        df.drop(0, inplace=True)

        df['Rating'] = make_feedback_implicit(df['Rating'], threshold)

        rows = [map_user_id[user] for user in df['User']]
        columns = [movie_id] * len(rows)
        data = list(df['Rating'])

        assert len(rows) == len(columns) and len(columns) == len(data)

        csr_rows += rows
        csr_columns += columns
        csr_data += data

    m_sp = sp.csr_matrix((csr_data, (csr_rows, csr_columns)), shape=(len(users), len(files)))
    m_sp.eliminate_zeros()
    save_as_npz(m_sp, BIN_DATA[NETFLIX])


def parse_lastfm(**kwargs):
    if os.path.isfile(BIN_DATA[LASTFM]):
        LOG.info("Already processed, skipping.")
        return

    data_file = 'usersha1-artmbid-artname-plays.tsv'
    source_file = os.path.join(DOWNLOAD[LASTFM], data_file)
    if not glob(source_file):
        download_lastfm()

    LOG.info("Parsing lastfm")
    df = pd.read_csv(source_file, delimiter='\t', names=["User", "Artist id", "Artist name", "Plays"], dtype=str)

    artist_column = list(zip([str(i) for i in df['Artist id']], [str(i) for i in df['Artist name']]))
    user_column = df['User']

    map_artist_id = {artist: i for i, artist in enumerate(sorted(set(artist_column)))}
    map_user_id = {user: i for i, user in enumerate(sorted(set(user_column)))}

    user_ids = [map_user_id[user] for user in user_column]
    artist_ids = [map_artist_id[artist] for artist in artist_column]

    m_sp = sp.csr_matrix(([1] * df.shape[0], (user_ids, artist_ids)), shape=(len(map_user_id), len(map_artist_id)))
    save_as_npz(m_sp, BIN_DATA[LASTFM])


def parse_pinterest(**kwargs):
    if os.path.isfile(BIN_DATA[PINTEREST]):
        LOG.info("Already processed, skipping.")
        return

    data_file = 'subset_iccv_board_pins.bson'
    source_file = os.path.join(DOWNLOAD[PINTEREST], data_file)
    if not glob(source_file):
        raise Exception("Cannot find pinterest dataset")

    LOG.info("Parsing pinterest")

    with open(source_file, 'rb') as f:
        bsob = bson.decode_all(f.read())

    map_id_pin = dict()
    map_pin_id = dict()
    map_board_id = dict()
    map_id_board = dict()
    pins = 0

    board_pin_pairs = []
    for i, board in enumerate(bsob):
        map_id_board[i] = board
        map_board_id[board['board_id']] = i
        for pin in board['pins']:
            if (pin not in map_pin_id):
                map_pin_id[pin] = pins
                map_id_pin[pins] = pin
                pins += 1
            board_pin_pairs.append((map_board_id[board['board_id']], map_pin_id[pin]))
    boards = [board for (board, pin) in board_pin_pairs]
    pins = [pin for (board, pin) in board_pin_pairs]

    m_sp = sp.csr_matrix(([1] * len(boards), (boards, pins)), shape=(len(map_board_id), len(map_pin_id)))

    save_as_npz(m_sp, BIN_DATA[PINTEREST])


def save_as_npz(m_sp, path):
    if not os.path.isdir(BIN_DATA_DIR):
        os.makedirs(BIN_DATA_DIR)
    sp.save_npz(path, m_sp)


def make_feedback_implicit(feedback, threshold):
    return [1 if rating >= threshold else 0 for rating in feedback]


def get_users(files):
    users = set()
    for i, file_path in tqdm(enumerate(files)):
        df = pd.read_csv(file_path, names=['User', 'Rating', 'Date'])
        df.drop(0, inplace=True)
        users.update(set(df['User']))
    return list(sorted(users))
