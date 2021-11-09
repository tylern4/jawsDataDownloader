from collections import deque
from datetime import datetime
import pandas as pd
from cromwell_tools.cromwell_auth import CromwellAuth
from cromwell_tools import api
import numpy as np
import json
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)


COLUMNS = ['end', 'id', 'name', 'start', 'status', 'submission']


@np.vectorize
def getInputSizeBytes(cromwell_id, auth):
    """
    Loop through the input files from a job and add together the total bytes.
    """
    # return np.nan
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
        for fil in fileNames:
            # Get file stats from the path
            try:
                total += Path(fil).stat().st_size
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
        for fil in fileNames:
            # Skip if it's not a string
            if not isinstance(fil, str):
                continue
            # Split the directory struture
            # and get the last value -> file
            filename = fil.split("/")[-1]
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
        for fil in fileNames:
            # Skip if it's not a string
            if not isinstance(fil, str):
                continue
            # Split the directory struture
            # and get the last value -> file
            filename = fil.split("/")[-1]
            # Split the filename struture
            # and get the last value -> file-extension
            # Check if file-extension is bz2/gz the common compression formats
            if filename.split('.')[-1] in ['bz2', 'gz']:
                # If any file is compressed return compressed
                compressed = True
            try:
                total += Path(fil).stat().st_size
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


@np.vectorize
def getMemory(mem, memory):
    if isinstance(mem, str):
        mem = mem.replace("G", "")
        return float(mem)
    elif isinstance(memory, str):
        memory = memory.replace("GB", "")
        return float(memory)
    else:
        return np.nan


def update(config: str = "config.json") -> str:
    try:
        with open(config, 'r') as f:
            config = json.load(f)
            CROMWELL_URL = f'http://{config["CROMWELL_HOST"]}:{config["CROMWELL_PORT"]}'
            OUTPUT_DIR = config["OUTOUT_DIRECTORY"]
            OUTPUT_NAME = config["OUTPUT_NAME"]
    except:
        return "Can't find config"

    start = datetime.now()
    # Authenticate with Cromwell with no Auth
    auth = CromwellAuth.harmonize_credentials(url=CROMWELL_URL)
    # Check if we have the file already
    fileName = f"{OUTPUT_DIR}/{OUTPUT_NAME}.csv"
    fileExists = Path(fileName).exists()
    TIME = None
    # TIME = "2021-09-14T15:04:45.825Z"
    # If csv exists
    if fileExists:
        with open(fileName, 'r') as fil:
            # Read the last line of the file
            # and get the last start time in file to read from
            TIME = deque(fil, 1)[0].split(',')[5]

        logging.info(f"Reading from {TIME}")

    if TIME is not None:
        apiResults = api.query({'status': 'Succeeded', 'start': TIME}, auth)
    else:
        apiResults = api.query({'status': 'Succeeded'}, auth)

    totalResultsCount = apiResults.json()['totalResultsCount']
    if totalResultsCount == 0:
        logging.info(
            f"No new results between {TIME} and {datetime.now()}")
        exit()
    else:
        logging.info(f"Downloading {totalResultsCount} results")

    # Create dataframe from resulting json file
    cromwellData = pd.DataFrame(apiResults.json()['results'])

    # Only keep these columns
    COLUMNS = ['name', 'id', 'submission', 'start', 'end']
    cromwellData = cromwellData[COLUMNS]
    # cromwellData['start']

    metaData = getMetaDataInfo(cromwellData.id, auth)
    metaData = pd.DataFrame.from_records(metaData)

    # Looks like the column name changed at some point
    # This merges them back into the memory column
    try:
        metaData['memory'] = getMemory(metaData.mem, metaData.memory)
    except AttributeError:
        metaData['memory'] = getMemory(None, metaData.memory)

    try:
        metaData['docker'] = metaData['docker'].fillna("none")
    except KeyError:
        metaData['docker'] = ["none" for _ in range(metaData.shape[0])]

    META_COLUMNS = ['poolname', 'shared', 'nwpn', 'cluster', 'cpu', 'constraint',
                    'node', 'account', 'time', 'qos', 'memory', 'id', 'input_size_bytes',
                    'input_compressed', 'docker']
    try:
        metaData = metaData[META_COLUMNS]
    except Exception as e:
        print(e)

    data = pd.merge(left=cromwellData, right=metaData,
                    left_on='id', right_on='id')
    try:
        data = data[~data.account.isnull()]
        data['submission'] = np.where(
            data.submission.isnull(), data.start, data.submission)
        data['shared'] = data['shared'].astype(bool)
    except Exception as e:
        print(e)
    # cromwell returns from newest to oldest
    # Reverse this for easier reading in date later
    data = data.iloc[::-1]

    # If the file exists append else just write out everything
    if fileExists:
        data.to_csv(fileName,  mode='a', header=False)
    else:
        data.to_csv(fileName)

    logging.info(f"total time {datetime.now()-start}")
    return data.shape[0]


if __name__ == "__main__":
    update()
