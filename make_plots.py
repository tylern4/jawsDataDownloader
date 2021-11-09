import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import functools
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, date2num

import seaborn as sns
sns.set_context("notebook", rc={"lines.linewidth": 2})
sns.set_theme()

from multiprocessing import Pool

@np.vectorize
def getTimeSeconds(time):
    time = time.split(":")
    time = int(time[0])*60*60 + int(time[1])*60 + int(time[2])
    return time

@np.vectorize
def loadData(filepath: str) -> pd.DataFrame:
    filename = filepath.split("/")[-1]
    siteConfig = filename.split("_")[0].split(".")[0]

    parse_dates = ['end', 'start', 'submission']
    data = pd.read_csv(filepath,
                   index_col=0,
                   parse_dates=parse_dates)
    data['queue_time'] = (data.start - data.submission) / np.timedelta64(1, 's')
    data['runtime'] = (data.end - data.start) / np.timedelta64(1, 's')
    data['requested_runtime'] = getTimeSeconds(data.time)

    data['siteConfig'] = [siteConfig for _ in range(data.shape[0])]
    data['constraint'] = data.constraint.fillna("None")

    return data

############################################
# From Brandon Cook LDMS examples
def union_categories_dtype(a,b):
    """For efficeint/correct merge operations pandas categorical
    types have to match"""
    s1, s2 = pd.unique(a), pd.unique(b)
    cats = pd.unique(np.concatenate([s1,s2]))
    return pd.api.types.CategoricalDtype(categories=cats)

@functools.lru_cache
def read_slurm_steps(path, exclude_steps=["batch", "extern"],
                     columns=["jobidraw", "step", "ProducerName"]):
    """Use an LRU cache because many per-job files will map to a single
    slurm_steps.parquet file
    """

    df = pd.read_parquet(path, columns=columns)
    df = df[~df['step'].isin(exclude_steps)]
    df['step'] = df['step'].astype('category')
    return df

def slurm_path_from_jobpath(path):
    """Path of Slurm step file that corresponds to a per-job parquet file"""

    root = path.parent.parent.parent.parent
    name = ".".join(path.name.split(".")[0:2] + ["slurm_steps", "parquet"])
    return root / "slurm" / name

def slurm_path_from_bulkpath(path):
    """Path of Slurm step file that corresponds to bulk data file"""
    s = str(path)
    s = s.replace("data", "slurm").replace(".parquet", ".slurm_steps.parquet")
    return Path(s)

def slurm_path(path):
    if "data" in str(path):
        return slurm_path_from_bulkpath(path)
    elif "jobs" in str(path):
        return slurm_path_from_jobpath(path)
    else:
        raise LookupError("Couldn't determine type of input file")

def read_parquet_with_steps(path, columns=None):
    """Read a parquet file and merge step column back in"""
    df = pd.read_parquet(path, columns=columns)
    steps = read_slurm_steps(slurm_path(path))
    t = union_categories_dtype(df['ProducerName'], steps['ProducerName'])
    df['ProducerName'] = df['ProducerName'].astype(t)
    steps['ProducerName'] = steps['ProducerName'].astype(t)
    return df.merge(steps, on=['jobidraw', 'ProducerName']).reset_index(drop=True)

def read_job(slurm_cluster, jobid, sampler, output_root="/global/cscratch1/sd/usgweb/ldms_output", columns=None, with_steps=False):
    """Load a dataframe for a specific job and sampler"""
    path = Path(Path(output_root),
                Path("jobs"),
                Path(f"{slurm_cluster}"),
                Path(f"{jobid}"))
    parquet_files = path.glob(f"*{sampler}*.parquet")

    if with_steps:
        dfs = (read_parquet_with_steps(fn, columns) for fn in parquet_files)
    else:
        dfs = (pd.read_parquet(fn, columns=columns) for fn in parquet_files)

    df = pd.concat(dfs).reset_index(drop=True)

    # pd.concat does not preserve categorical type when the categories are different
    if with_steps:
        df['step'] = df['step'].astype('category')
    df['ProducerName'] = df['ProducerName'].astype('category')
    return df

############################################


def plot_vertical(_data, set_log=False):
    data = _data[1]
    rolling = 120
    try:
        proctstat = read_job("cori", data.JobID, "procstat_Hsw64", "/global/cscratch1/sd/usgweb/ldms_output").set_index('Time')
        proctstat = proctstat[['ProducerName','idle','user']]

        df_mem = read_job("cori", data.JobID, "meminfo").set_index('Time').sort_index()
        df_mem['MemUsed'] = df_mem['MemTotal'] - df_mem['MemFree'] - df_mem['Buffers'] - df_mem['Cached']
        df_mem['MemUsed'] *= 1e-6 # KB -> GB

        df_lustre = read_job("cori", data.JobID, "lustre_llite").set_index('Time').sort_index()

    except ValueError:
        # print("Not Working")
        return None

    fig, ax = plt.subplots(3,1,figsize=[12,6], sharex=True)
    fig.subplots_adjust(wspace=0, hspace=0.05)
    date_form = DateFormatter("%H:%M")
    for _ax in ax.flatten():
        _ax.xaxis.set_major_formatter(date_form)
        _ax.grid(True)
        # Cut axis at Slurm start and end
        _ax.set_xlim(left=data.Start+datetime.timedelta(seconds=rolling)*0.9,
                            right=data.End+datetime.timedelta(seconds=rolling)*1.1)
        if set_log:
            _ax.set_yscale('symlog')
        try:
            pass
            # Cromwell start and end lines
            #_ax.axvline(data.start+datetime.timedelta(seconds=rolling), c='#007AC8')
            #_ax.axvline(data.end+datetime.timedelta(seconds=rolling), c='#007AC8')

            # Slurm start and end lines
            #_ax.axvline(data.Start+datetime.timedelta(seconds=rolling), c='#CC5500')
            #_ax.axvline(data.End+datetime.timedelta(seconds=rolling), c='#CC5500')
        except:
            print("Not Working ax")
            pass
    try:
        overtime = (data.End - data.end.tz_convert(None)).seconds
        hours = overtime//3600
        minutes = (overtime//60)%60
        if hours == 0:
            fig.suptitle(f'{data.JobName}\nOver time {minutes:02d} min.')
        else:
            fig.suptitle(f'{data.JobName}\nOver time {hours:02d} hrs {minutes:02d} min')

    except:
        fig.suptitle(f'{data.JobName}')

    idle = proctstat['idle'].rolling(rolling).apply(lambda x: x.iloc[-1] - x.iloc[0])
    user = proctstat['user'].rolling(rolling).apply(lambda x: x.iloc[-1] - x.iloc[0])
    user = 100*user/(user+idle)
    sns.lineplot(x=proctstat.index, y=user, ax=ax[0], legend=False, ci='sd')

    ax[0].set_ylim(0, 110)
    ax[0].set_ylabel('CPU Utilization')
    ax[0].axhline(100*data.AllocCPUS/64, dashes=(5, 2, 1, 2), c='r')
    ax[0].axhline(100*data.ReqCPUS/64, dashes=(1, 2, 1, 2))

    sns.lineplot(x=df_mem.index, y=df_mem.MemUsed.rolling(rolling).mean(), ax=ax[1])
    ax[1].set_ylim(0, 130)
    ax[1].axhline(data.ReqMemNode * 1e-9, dashes=(1, 2, 1, 2))


    read_bytes = (df_lustre['client.read_bytes.rate#llite.snx11168'] * 1e-6).rolling(rolling).mean()
    write_bytes = (df_lustre['client.write_bytes.rate#llite.snx11168'] * 1e-6).rolling(rolling).mean()
    sns.lineplot(x=df_lustre.index, y=read_bytes, ax=ax[2])
    sns.lineplot(x=df_lustre.index, y=write_bytes, ax=ax[2])
    ax[2].set_ylabel('Transfer Rate Bytes')
    try:
        top = int(1.5*np.max([np.max(read_bytes), np.max(write_bytes)]))
    except ValueError:
        top = 0
    if top != 0:
        ax[2].set_ylim(0, top)
    print(f"trying to save plots/{data.Account}/{data.Group}/{data.User}/{data.JobName}_{data.Time}.png")
    Path(f'plots/{data.Account}/{data.Group}/{data.User}').mkdir(parents=True, exist_ok=True)
    plt.savefig(f"plots/{data.Account}/{data.Group}/{data.User}/{data.JobName}_{data.Time}.png")
    plt.close(fig)

    ####################
    fig, ax = plt.subplots(3,1,figsize=[12,6], sharex=True)
    fig.subplots_adjust(wspace=0, hspace=0.05)
    date_form = DateFormatter("%H:%M")
    for _ax in ax.flatten():
        _ax.xaxis.set_major_formatter(date_form)
        _ax.grid(True)
        if set_log:
            _ax.set_yscale('symlog')
        try:
            # Cut axis at Slurm start and Cromwell end
            _ax.set_xlim(left=data.Start+datetime.timedelta(seconds=rolling)*0.9,
                            right=data.end+datetime.timedelta(seconds=rolling)*1.1)

        except:
            pass

    fig.suptitle(data.JobName)

    idle = proctstat['idle'].rolling(rolling).apply(lambda x: x.iloc[-1] - x.iloc[0])
    user = proctstat['user'].rolling(rolling).apply(lambda x: x.iloc[-1] - x.iloc[0])
    user = user/(user+idle)
    sns.lineplot(x=proctstat.index, y=user, ax=ax[0], legend=False, ci='sd')

    ax[0].set_ylabel('CPU Utilization')
    ax[0].axhline(100*data.ReqCPUS/64, dashes=(1, 2, 1, 2))

    sns.lineplot(x=df_mem.index, y=df_mem.MemUsed.rolling(rolling).mean(), ax=ax[1])
    ax[1].axhline(data.ReqMemNode * 1e-9, dashes=(1, 2, 1, 2))


    read_bytes = (df_lustre['client.read_bytes.rate#llite.snx11168'] * 1e-6).rolling(rolling).mean()
    write_bytes = (df_lustre['client.write_bytes.rate#llite.snx11168'] * 1e-6).rolling(rolling).mean()
    sns.lineplot(x=df_lustre.index, y=read_bytes, ax=ax[2])
    sns.lineplot(x=df_lustre.index, y=write_bytes, ax=ax[2])
    ax[2].set_ylabel('Transfer Rate Bytes')
    try:
        top = int(1.5*np.max([np.max(read_bytes), np.max(write_bytes)]))
    except ValueError:
        top = 0
    if top != 0:
        ax[2].set_ylim(0, top)

    plt.savefig(f"plots/{data.Account}/{data.Group}/{data.User}/zoomed_{data.JobName}_{data.Time}.png")
    plt.close(fig)

    return data.JobName





if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--jaws', action=argparse.BooleanOptionalAction,
                        help='Load in jaws cromwell data or not.',
                        default=False,
                        )

    parser.add_argument('--threads',
                        help='Number of threads for plotting.',
                        default=2)

    args = parser.parse_args()

    print("loading slurm data")
    con = sqlite3.connect("/global/cscratch1/sd/tylern/genepool.sqlite3")
    cur = con.cursor()

    keepColumns = 'JobID,ArrayTaskID,Partition,JobName,User,"Group",Account,State,Timelimit,Elapsed,Time,Submit,Start,End,Priority,ConsumedEnergy,ReqCPUS,AllocCPUS,CPUTime,TotalCPU,UserCPU,SystemCPU,ReqMemNode,ReqMemCPU'
    parse_dates = ['Time','Submit','Start','End']
    slurm_data = pd.read_sql(f"SELECT {keepColumns} FROM allocations WHERE JobID > 44229702", con, parse_dates=parse_dates)
    if args.jaws:
        print("loading cromwell data")
        files = ["/global/cfs/cdirs/nstaff/tylern/jawsData/coriDev.csv",
             "/global/cfs/cdirs/nstaff/tylern/jawsData/coriProd.csv",
             "/global/cfs/cdirs/nstaff/tylern/jawsData/coriStaging.csv"
             ]

        loadedDF = loadData(files)
        cromwell_data = pd.concat(loadedDF)

        split_ = lambda x : x.split('_')[-1]
        slurm_data['cromwell_id'] = slurm_data.JobName.apply(split_)
        print("merging data")
        all_data = cromwell_data.merge(slurm_data, left_on='id', right_on='cromwell_id')
    else:
        all_data = slurm_data
    print("starting pool")

    with Pool(int(args.threads)) as p:
        p.map(plot_vertical, all_data.iterrows())

