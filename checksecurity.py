from cromwell_tools.cromwell_auth import CromwellAuth
from cromwell_tools import api
import argparse
import json


def check(config):
    try:
        with open(config, 'r') as f:
            config = json.load(f)
            CROMWELL_URL = f'http://{config["CROMWELL_HOST"]}:{config["CROMWELL_PORT"]}'
            OUTPUT_DIR = config["OUTOUT_DIRECTORY"]
            OUTPUT_NAME = config["OUTPUT_NAME"]
    except:
        return "Can't find config"

    auth = CromwellAuth.harmonize_credentials(url=CROMWELL_URL)
    apiResults = api.query({'status': 'Running'}, auth)

    print(apiResults.json()['results'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', dest='config', type=str,
                        default="config.json")

    args = parser.parse_args()

    print(check(config=args.config))
