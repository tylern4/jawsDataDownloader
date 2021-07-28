import pandas as pd
from cromwell_tools.cromwell_auth import CromwellAuth
from cromwell_tools import api
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime


COLUMNS = ['end', 'id', 'name', 'start', 'status', 'submission']


@np.vectorize
def getInputSizeBytes(cromwell_id, auth):
    """
    Loop through the input files from a job and add together the total bytes.
    """

    # Try to get the metadata if not log the error
    try:
        metaData = api.metadata(cromwell_id, auth).json()
    except Exception as e:
        logging.exception(f"Cannot get metaData : {e}")

    # Check if there are input files in the metadata if not return nan
    try:
        submittedFiles = json.loads(
            metaData.json()['submittedFiles']['inputs'])
    except KeyError:
        return np.nan

    # Set total bytes to 0
    total = 0
    # Loop through the submittedFiles:inputs
    for key, fileNames in submittedFiles.items():
        # Hack to put single input file into list
        if not isinstance(fileNames, list):
            fileNames = [fileNames]
        # Loop through files
        for file in fileNames:
            # Get file stats from the path
            try:
                total += Path(file).stat().st_size
            except (FileNotFoundError, TypeError, PermissionError):
                # Ignore file if we have common errors
                pass
            except Exception as e:
                # Log uncommon errors
                logging.exception(f"Cannot get file stats : {e}")
    # return total sum of the size
    return total


@np.vectorize
def getInputCompressed(cromwell_id, auth):
    """
    Check file extenstions to determine if any of the files were compressed.
    """
    # Try to get the metadata if not log the error
    try:
        metaData = api.metadata(cromwell_id, auth)
    except Exception as e:
        logging.exception(f"Cannot get metaData : {e}")

    # Check if there are input files in the metadata if not return noInputs
    try:
        submittedFiles = json.loads(
            metaData.json()['submittedFiles']['inputs'])
    except KeyError:
        return 'noInputs'

    # Loop through the submittedFiles:inputs
    for key, fileNames in submittedFiles.items():
        # Hack to put single input file into list
        if not isinstance(fileNames, list):
            fileNames = [fileNames]
        # Loop through files
        for file in fileNames:
            # Skip if it's not a string
            if not isinstance(file, str):
                continue
            # Split the directory struture
            # and get the last value -> file
            filename = file.split("/")[-1]
            # Split the filename struture
            # and get the last value -> file-extension
            # Check if file-extension is bz2/gz the common compression formats
            if filename.split('.')[-1] in ['bz2', 'gz']:
                # If any file is compressed return compressed
                return 'compressed'

    return 'uncompressed'


@np.vectorize
def getMetaDataInfo(cromwell_id, auth):
    """
    Loop through the input files from a job and add together the total bytes.
    """

    # Set up data to return

    # Try to get the metadata if not log the error
    try:
        metaData = api.metadata(cromwell_id, auth).json()
    except Exception as e:
        logging.exception(f"Cannot get metaData : {e}")

    # Setup majority of the data
    try:
        # Get the name of the run
        name = list(metaData['calls'].keys())[0]
        data = metaData['calls'][name][0]['runtimeAttributes']
        data['id'] = cromwell_id
    except (IndexError, KeyError):
        # print(metaData)
        return {'id': cromwell_id}

    # Check if there are input files in the metadata if not return nan
    try:
        submittedFiles = json.loads(
            metaData['submittedFiles']['inputs'])
    except KeyError:
        data['input_size_bytes'] = np.nan
        # data['inputs'] = "none"
        data['input_compressed'] = False
        return data

    # Set total bytes to 0
    total = 0
    compressed = False
    # Loop through the submittedFiles:inputs
    for key, fileNames in submittedFiles.items():
        # Hack to put single input file into list
        if not isinstance(fileNames, list):
            fileNames = [fileNames]
        # Loop through files
        for file in fileNames:
            # Skip if it's not a string
            if not isinstance(file, str):
                continue
            # Split the directory struture
            # and get the last value -> file
            filename = file.split("/")[-1]
            # Split the filename struture
            # and get the last value -> file-extension
            # Check if file-extension is bz2/gz the common compression formats
            if filename.split('.')[-1] in ['bz2', 'gz']:
                # If any file is compressed return compressed
                compressed = True
            try:
                total += Path(file).stat().st_size
            except (FileNotFoundError, TypeError, PermissionError):
                # Ignore file if we have common errors
                pass
            except Exception as e:
                # Log uncommon errors
                logging.exception(f"Cannot get file stats : {e}")
    # return total sum of the size
    data['input_size_bytes'] = total
    data['input_compressed'] = compressed
    return data


def update(config: str = "services/config.json") -> str:
    try:
        with open(config, 'r') as f:
            config = json.load(f)
            CROMWELL_URL = f'http://{config["CROMWELL_HOST"]}:{config["CROMWELL_PORT"]}'
            OUTPUT_DIR = config["OUTOUT_DIRECTORY"]
            OUTPUT_NAME = config["OUTPUT_NAME"]
    except:
        return "Can't find config"

    # Authenticate with Cromwell with no Auth
    auth = CromwellAuth.harmonize_credentials(url=CROMWELL_URL)
    apiResults = api.query({}, auth)
    # Create dataframe from resulting json file
    df = pd.DataFrame(apiResults.json()['results'])
    # Drop some columns
    df = df[COLUMNS]
    df.to_csv(f"{OUTPUT_DIR}/{OUTPUT_NAME}_{datetime.now()}.csv")
    return f'New file at: {OUTPUT_DIR}/{OUTPUT_NAME}_{datetime.now()}.csv'


if __name__ == "__main__":
    update()
