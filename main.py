import argparse
import toml
import warnings
import logging
from libs.CrimePredictionApp import CrimePredictionApp

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='config.toml', help="Primary configuration toml input")
    args = parser.parse_args()

    config = toml.load(args.file)

    try:
        app = CrimePredictionApp.from_dict(config)
        requests = CrimePredictionApp.build_requests_from_dict(config)
        app._main(requests=requests)
    except Exception as e:
        logging.log("Unhandled Error", exc_info=e)


