# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Get jobs currently being processed on clusters
# ---

# ### Import modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle

import numpy as np
import pandas as pd

import subprocess

# #########################################################
from methods import get_df_jobs_paths
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Script inputs

TEST = True

# +
if os.environ["COMPENV"] != "wsl":
    TEST = False

if verbose:
    print("TEST:", TEST)
# -

# ### Read data objects

df_paths = get_df_jobs_paths()

# + active=""
#
#
# -

# ### Parse currently running/pending jobs from `jobs_mine` command

# +
compenv = os.environ["COMPENV"]
scripts_dir = os.environ["sc"]

if TEST:
    compenv = "sherlock"


if compenv == "sherlock":
    bash_comm = "python %s/08_slurm_jobs/jobs.py" % scripts_dir
elif compenv == "slac":
    bash_comm = "/usr/local/bin/bjobs -w"
elif compenv == "nersc":
    bash_comm = "/global/homes/f/flores12/usr/bin/queues"
else:
    bash_comm = ""


if verbose:
    print("bash_comm:", "\n", bash_comm, sep="")
# -

if not TEST:
    result = subprocess.run(
        bash_comm.split(" "),
        stdout=subprocess.PIPE,
        )

    output_list = result.stdout.decode('utf-8').splitlines()
    if verbose:
        print(output_list)

# + jupyter={"source_hidden": true}
if TEST:
    output_list = [
        'No prev_jobs.txt found. Writing prev_jobs.txt now',
        '+----------+-----------+----+----------+-------+---------------------------------------------------------------------------------------------------------------------------------------------------+',
        '|  Job ID  | Partition | ST | Run Time | Qtime | Path                                                                                                                                              |',
        '+----------+-----------+----+----------+-------+---------------------------------------------------------------------------------------------------------------------------------------------------+',
        '| 17512866 | owners,ir | PD |   0.0    |  0.7  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/7ic1vt7pz4/010/active_site__37/01_attempt/_01    |',
        '| 17512840 |    iric   | R  |   0.3    |  0.4  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/8l919k6s7p/1-100/active_site__50/01_attempt/_01  |',
        '| 17513849 | owners,ir | PD |   0.0    |  0.2  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/8p937183bh/2-1-11/active_site__33/01_attempt/_01 |',
        '| 17512889 | owners,ir | PD |   0.0    |  0.7  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/926dnunrxf/010/active_site__48/01_attempt/_01    |',
        '| 17512874 | owners,ir | PD |   0.0    |  0.7  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/bgcpc2vabf/010/active_site__64/01_attempt/_01    |',
        '| 17512851 | owners,ir | PD |   0.0    |  0.7  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/cq7smr6lvj/20-3/active_site__49/01_attempt/_01   |',
        '| 17512897 | owners,ir | PD |   0.0    |  0.7  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/nscdbpmdct/1-102/active_site__28/01_attempt/_01  |',
        '| 17512835 | owners,ir | PD |   0.0    |  0.7  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/v2blxebixh/2-10/active_site__67/01_attempt/_01   |',
        '| 17512860 | owners,ir | PD |   0.0    |  0.7  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/zimixdvdxd/2-1-11/active_site__56/01_attempt/_01 |',
        '| 17512854 | owners,ir | PD |   0.0    |  0.7  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/zimixdvdxd/2-1-11/active_site__61/01_attempt/_01 |',
        '| 17354096 |   owners  | R  |   12.7   |  3.3  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_slabs/run_oh_covered/out_data/dft_jobs/mrbine8k72/010/oh/active_site__33/02_attempt/_04    |',
        '| 17232073 |   owners  | R  |   12.7   |  2.4  | /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_slabs/run_oh_covered/out_data/dft_jobs/mrbine8k72/010/oh/active_site__34/01_attempt/_03    |',
        '+----------+-----------+----+----------+-------+---------------------------------------------------------------------------------------------------------------------------------------------------+',
        ]

# +
PROJ_irox_oer = os.environ["PROJ_irox_oer"]

if TEST:
    PROJ_irox_oer = "/scratch/users/flores12/PROJ_IrOx_OER"

# +
# #########################################################
data_dict_list = []
# #########################################################
for line_i in output_list:
    if PROJ_irox_oer in line_i:
        for split_j in line_i.split(" "):
            if PROJ_irox_oer in split_j:
                path_parsed_j = split_j

                find_ind = path_parsed_j.find(PROJ_irox_oer)
                path_short_j = path_parsed_j[find_ind + len(PROJ_irox_oer) + 1:]

                # #########################################
                data_dict_i = dict()
                # #########################################
                data_dict_i["compenv"] = compenv
                data_dict_i["path"] = path_parsed_j
                data_dict_i["path_short"] = path_short_j
                # #########################################
                data_dict_list.append(data_dict_i)
                # #########################################

# #########################################################
df = pd.DataFrame(data_dict_list)
# #########################################################
# -

# ### Getting job_id by mapping path to `df_jobs_paths`

# +
# #########################################################
data_dict_list = []
# #########################################################
for index_i, row_i in df.iterrows():
    # #####################################################
    path_short_i = row_i.path_short
    # #####################################################


    df_paths_i = df_paths[df_paths.path_rel_to_proj == path_short_i]

    df_paths_2_i = df_paths[df_paths.path_rel_to_proj__no_compenv == path_short_i]

    # assert df_paths_i.shape[0] == 1, "Must only be one"
    assert df_paths_i.shape[0] <= 1, "Must only be one, or 0 (not ideal)"


    if df_paths_i.shape[0] > 0:
        df_paths_i = df_paths_i
    elif df_paths_2_i.shape[0] > 0:
        # print("SDIFJIDS")
        df_paths_i = df_paths_2_i


    if df_paths_i.shape[0] > 0:
        row_paths_i = df_paths_i.iloc[0]

        job_id_i = row_paths_i.name

        # #################################################
        data_dict_i = dict()
        # #################################################
        data_dict_i["job_id"] = job_id_i
        # #################################################
        data_dict_i.update(row_i.to_dict())
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

# #########################################################
df_jobs_on_clus = pd.DataFrame(data_dict_list)

if len(data_dict_list) == 0:
    df_jobs_on_clus = pd.DataFrame(columns=["job_id", ])
else:
    df_jobs_on_clus = df_jobs_on_clus.set_index("job_id", drop=False)
# #########################################################

# +
# assert False
# -

# ### Writing dataframe to file

# +
out_data_dir_rel_to_proj = os.path.join(
    "dft_workflow/cluster_scripts",
    "out_data",
    )

directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    out_data_dir_rel_to_proj)

# print("directory:", directory)

if not os.path.exists(directory):
    os.makedirs(directory)

# Pickling data ###########################################
file_path = os.path.join(
    directory, "df_jobs_on_clus__%s.pickle" % compenv)
with open(file_path, "wb") as fle:
    pickle.dump(df_jobs_on_clus, fle)
# #########################################################
# -

# ### Syncing dataframe file to Dropbox

dir_rel_to_dropbox = os.path.join(
    "01_norskov/00_git_repos/PROJ_IrOx_OER",
    out_data_dir_rel_to_proj,
    )

# +
rclone_comm_i = "rclone copy %s %s:%s" % (file_path, os.environ["rclone_dropbox"], dir_rel_to_dropbox)

rclone_comm_list_i = [i for i in rclone_comm_i.split(" ") if i != ""]

result = subprocess.run(
    rclone_comm_list_i,
    stdout=subprocess.PIPE)
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("get_jobs_running.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
