#!/usr/bin/env bash

# source activate cromwell

cd /global/cfs/cdirs/nstaff/tylern/jawsDataDownloader

RUNTIME=`date +"%Y-%m-%d"`

PYTHONBIN=/global/homes/t/tylern/.conda/envs/cromwell/bin/python
SITES=(cori_prod cori_dev cori_staging nmdc)

for site in "${SITES[@]}";
do
    echo "======== ${RUNTIME} ======== ${site} ======== " >> logs/${site}.log
    $PYTHONBIN update.py --config ${site}.json >> logs/${site}.log 2>&1
    echo "======== ${RUNTIME} ======== ${site} ======== " >> logs/${site}.log
done

cd /global/cfs/cdirs/nstaff/tylern/slurmInfo
export PYTHONPATH=/opt/mods/lib/python3.6/site-packages:/opt/ovis/lib/python3.6/site-packages
export PYTHONUSERBASE=/global/homes/t/tylern/.local/cori/3.8-anaconda-2020.11
export PYTHON_DIR=/usr/common/software/python/3.8-anaconda-2020.11
export CONDA_EXE=/usr/common/software/python/3.8-anaconda-2020.11/bin/conda
export CONDA_PYTHON_EXE=/usr/common/software/python/3.8-anaconda-2020.11/bin/python
export SLURM2SQL=/global/homes/t/tylern/.local/cori/3.8-anaconda-2020.11/bin/slurm2sql
$SLURM2SQL --history-resume -u genepool.sqlite3 -- --allusers -r genepool,genepool_shared >> logs/genepool.log 2>&1
$SLURM2SQL --history-resume -u nmdc.sqlite3 -- --user nmdcda >> logs/nmdc.log 2>&1
$SLURM2SQL --history-resume -u kbase.sqlite3 -- --user kbaserun >> logs/kbase.log 2>&1
