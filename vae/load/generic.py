from vae.config import NETFLIX, ML_20M, LASTFM, PINTEREST, BIN_DATA, DATASETS, LOG, DOWNLOAD, ML_20M_ALT
from vae.load.parsers import parse_netflix, parse_movielens, parse_lastfm, parse_pinterest
from vae.load.preprocessing import load_data, load_and_parse_ML_20M


def load_dataset(dataset: str, *args, **kwargs):
    """
    Generic data loader.
    :param dataset: name of dataset to be loaded
    :return: 5 csr_matrices {train, valid_in, valid_out, test_in, test_out}
    """
    assert dataset in DATASETS, "Wrong dataset name"

    if dataset == ML_20M:
        out = load_and_parse_ML_20M(DOWNLOAD[ML_20M], *args, **kwargs)
        LOG.info("Done")
        return out

    handler_map_parse = {
        NETFLIX: parse_netflix,
        ML_20M_ALT: parse_movielens,
        LASTFM: parse_lastfm,
        PINTEREST: parse_pinterest
    }
    handler_map_parse[dataset]()

    out = load_data(BIN_DATA[dataset], *args, **kwargs)
    LOG.info("Done")
    return out
