# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# + jupyter={"source_hidden": true}
import os
import sys

import copy
import shutil
from pathlib import Path
from contextlib import contextmanager

# import pickle; import os

import pickle
import  json

import pandas as pd
import numpy as np

from ase import io

import plotly.graph_objects as go

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis import local_env

# #########################################################
from misc_modules.pandas_methods import drop_columns

from methods import read_magmom_comp_data

import os
print(os.getcwd())
import sys

from IPython.display import display

import pandas as pd
pd.set_option("display.max_columns", None)
pd.options.display.max_colwidth = 20
# pd.set_option('display.max_rows', None)

# #########################################################
from methods import (
    get_df_jobs_paths,
    get_df_dft,
    get_df_job_ids,
    get_df_jobs,
    get_df_jobs_data,
    get_df_slab,
    get_df_slab_ids,
    get_df_jobs_data_clusters,
    get_df_jobs_anal,
    get_df_slabs_oh,
    get_df_init_slabs,
    get_df_magmoms,
    get_df_ads,
    get_df_atoms_sorted_ind,
    get_df_rerun_from_oh,
    get_df_slab_simil,
    get_df_active_sites,
    get_df_features_targets,
    )
from methods import (
    get_other_job_ids_in_set,
    read_magmom_comp_data,
    )
from methods import get_df_coord

from ase.visualize import view

# + jupyter={"source_hidden": true}
df_dft = get_df_dft()
df_job_ids = get_df_job_ids()
df_jobs = get_df_jobs(exclude_wsl_paths=True)
df_jobs_data = get_df_jobs_data(exclude_wsl_paths=True)
df_jobs_data_clusters = get_df_jobs_data_clusters()
df_slab = get_df_slab()
df_slab_ids = get_df_slab_ids()
df_jobs_anal = get_df_jobs_anal()
df_jobs_paths = get_df_jobs_paths()
df_slabs_oh = get_df_slabs_oh()
df_init_slabs = get_df_init_slabs()
df_magmoms = get_df_magmoms()
df_ads = get_df_ads()
df_atoms_sorted_ind = get_df_atoms_sorted_ind()
df_rerun_from_oh = get_df_rerun_from_oh()
magmom_data_dict = read_magmom_comp_data()
df_slab_simil = get_df_slab_simil()
df_active_sites = get_df_active_sites()
df_features_targets = get_df_features_targets()


# + jupyter={"source_hidden": true}
def display_df(df, df_name, display_head=True, num_spaces=3):
    print(40 * "*")
    print(df_name)
    print("df_i.shape:", df_i.shape)
    print(40 * "*")

    if display_head:
        display(df.head())

    print(num_spaces * "\n")

df_list = [
    ("df_dft", df_dft),
    ("df_job_ids", df_job_ids),
    ("df_jobs", df_jobs),
    ("df_jobs_data", df_jobs_data),
    ("df_jobs_data_clusters", df_jobs_data_clusters),
    ("df_slab", df_slab),
    ("df_slab_ids", df_slab_ids),
    ("df_jobs_anal", df_jobs_anal),
    ("df_jobs_paths", df_jobs_paths),
    ("df_slabs_oh", df_slabs_oh),
    ("df_magmoms", df_magmoms),
    ("df_ads", df_ads),
    ("df_atoms_sorted_ind", df_atoms_sorted_ind),
    ("df_rerun_from_oh", df_rerun_from_oh),
    ("df_slab_simil", df_slab_simil),
    ("df_active_sites", df_active_sites),
    ]

# for name_i, df_i in df_list:
#     display_df(df_i, name_i)

# print("")
# print("")

# for name_i, df_i in df_list:
#     display_df(
#         df_i,
#         name_i,
#         display_head=False,
#         num_spaces=0)
# -

# ### Script Inputs

# +
# ('sherlock', 'telibose_95', 'oh', 35.0, 1)

# +
# dft_jobs/nd919pnr6q/111/01_attempt

# +
names_i = [
    # ('slac', 'votafefa_68', 38.0),
    # ('sherlock', 'kegidafu_92', 66.0),
    # ('slac', 'bikoradi_95', 65., )
    # ('sherlock', 'ramufalu_44', 54.0, )
    
    ('sherlock', 'telibose_95', 54.0, ),

    ]

# ('slac', 'bikoradi_95', 'o', 65, 1, False)
# ('sherlock', 'kegidafu_92', 'oh', 66.0, 1, True)

# job_id = "melifiwi_93"
job_id = "toberotu_75"

# ('slac', 'waloguhe_35', 65.0),
# ('sherlock', 'kesekodi_38', 50.0),
# ('slac', 'votafefa_68', 38.0),

# +
from methods import get_other_job_ids_in_set

df_oer_set = get_other_job_ids_in_set(job_id, df_jobs=df_jobs, oer_set=True)

# # get_other_job_ids_in_set?

# +
df = df_oer_set[["compenv", "slab_id", "ads", "active_site", "att_num", ]]

idx = pd.MultiIndex.from_tuples(
    [tuple(x) for x in df.to_records(index=False)]
    )

unique_idx = idx.unique()

long_indices = unique_idx.tolist()
# -

df_jobs_anal.loc[long_indices]

# +
# assert False

# +
df_ind = df_features_targets.index.to_frame()

df = df_ind
df = df[
    (df["compenv"] == "slac") &
    (df["slab_id"] == "bikoradi_95") &
    # (df["active_site"] == 65.) &
    [True for i in range(len(df))]
    ]
df

# +
from methods import create_name_str_from_tup

df_feat_i = df_features_targets.loc[names_i]

out_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "__temp__/oer_sets",
    )
try:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
except:
    tmp = 42

job_ids = []
for index_i, row_i in df_feat_i.iterrows():

    atoms_list = []
    ads_list = ["o", "oh", "bare", ]
    # ads_list = ["o", "bare", ]
    # ads_list = ["oh", ]
    for ads_i in ads_list:
        job_id_i = row_i["data"]["job_id_" + ads_i][""]
        # print("job_id_i:", job_id_i)
        job_ids.append(job_id_i)

        row_paths_i = df_jobs_paths.loc[job_id_i]

        path_dir = os.path.join(
            os.environ["PROJ_irox_oer_gdrive"],
            row_paths_i.gdrive_path,
            )
        path_0 = os.path.join(path_dir, "final_with_calculator.json")
        path_1 = os.path.join(path_dir, "out.cif")

        atoms_i = io.read(path_0)
        atoms_list.append(atoms_i)

        out_dir = os.path.join(
            os.environ["PROJ_irox_oer"],
            "__temp__/oer_sets",
            create_name_str_from_tup(index_i),
            )
        print(out_dir)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        shutil.copy(
            path_1,
            os.path.join(
                out_dir,
                ads_i + ".cif"
                )
            )

    view(atoms_list)
# -
print("job_ids:", job_ids)

# ### Print out local paths of jobs

# job_ids = ["pibuvule_81", ]
# job_ids = ["toberotu_75", ]
job_ids = ["wawehamu_56", ]

for job_id_i in job_ids:
    row_paths_i = df_jobs_paths.loc[job_id_i]
    gdrive_path_i = row_paths_i.gdrive_path

    full_path_i = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        gdrive_path_i)
    print(full_path_i)

# +
# frag_i = "slac/8p8evt9pcg/220/bare/active_site__24/01_attempt"
# frag_i = "slac/8p8evt9pcg/220/bare/active_site__24"
# frag_i = "8p8evt9pcg/220/bare"
# frag_i = "slac/8p8evt9pcg/220/bare/active_site__"
# frag_i = "slac/8p8evt9pcg/220/oh"
# frag_i = "dft_jobs/sherlock/v2blxebixh/2-10/bare/active_site__49/01_attempt"
# frag_i = "dft_workflow/run_slabs/run_oh_covered/out_data/dft_jobs/slac/b5cgvsb16w/111/oh/active_site__71/01_attempt/_02"
# frag_i = "dft_workflow/run_slabs/run_oh_covered/out_data/dft_jobs/zimixdvdxd/010/oh/active_site__26/00_attempt/_01"
# frag_i = "zimixdvdxd/010/oh/active_site__26"
# frag_i = "dft_workflow/run_slabs/run_oh_covered/out_data/dft_jobs/slac/zimixdvdxd/010/oh/active_site__26/00_attempt/_01"

# frag_i = "dft_jobs/nd919pnr6q/111/01_attempt"
frag_i = "nd919pnr6q/111"

for job_id_i, row_i in df_jobs_paths.iterrows():
    gdrive_path_i = row_i.gdrive_path
    if frag_i in gdrive_path_i:
        print(job_id_i)
        print(gdrive_path_i)
        print("")

# +
df_jobs_paths

df_jobs.loc["seladuri_58"]

('sherlock', 'likeniri_51', 'o', 'NaN', 1)

# + active=""
#
#
#
#
# -

import os
import json

# +
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer_gdrive"],
    "dft_workflow/run_slabs/run_o_covered/out_data/dft_jobs/slac",
    "8p937183bh/10-12/active_site__38/01_attempt/_01")

# #########################################################
path_i = os.path.join(
    root_path_i,
    "out_data/init_magmoms.json")
with open(path_i, "r") as fle:
    init_magmoms = json.load(fle)

# #########################################################
path_i = os.path.join(
    root_path_i,
    "init.traj")
atoms_init = io.read(path_i)

# #########################################################
path_i = os.path.join(
    root_path_i,
    "OUTCAR")
atoms_outcar = io.read(path_i, index=":")

# +
atoms_0 = atoms_outcar[0]
magmoms = atoms_0.get_magnetic_moments()

for atom, magmom_i in zip(atoms_0, magmoms):
    print(
        atom.index, "|",
        atom.symbol, "|",
        # atom.magmom, "|",
        magmom_i, "|",
        atom.position,
        )
# -

np.sum(magmoms)
# magmoms

# +
# # io.read?

# +
# # MAGMOM = 1*1.0540 1*1.0010 1*-0.3030 1*-0.9940 1*-0.3480 1*0.3280 1*0.0070 1*-0.0020 1*-0.0370 1*-0.0240 1*-0.0270 1*-0.0440 1*-    0.0520 1*-0.0250 1*-0.0340 1*-0.0420 1*-0.0170 1*-0.0300 1*-0.0120 1*-0.0400 1*0.0050 1*-0.0070 1*0.0670 1*-0.0320 1*0.2310 1*-0.    0070 1*-0.0130 1*-0.0010 1*0.8900 1*-0.8390 1*-0.0530 1*-0.1740 1*-0.0140 1*-0.2610 1*-0.2180 1*-0.1320 1*0.0410 1*0.2170 1*-0.      0180

# 1.0540
# 1.0010
# -0.3030
# -0.9940
# -0.3480
# 0.3280
# 0.0070
# -0.0020
# -0.0370
# -0.0240
# -0.0270
# -0.0440
# -0.0520
# -0.0250
# -0.0340
# -0.0420
# -0.0170
# -0.0300
# -0.0120
# -0.0400
# 0.0050
# -0.0070
# 0.0670
# -0.0320
# 0.2310
# -0.0070
# -0.0130
# -0.0010
# 0.8900
# -0.8390
# -0.0530
# -0.1740
# -0.0140
# -0.2610
# -0.2180
# -0.1320
# 0.0410
# 0.2170
# -0.0180
# -

for atom in atoms_init:
    print(
        atom.index, "|",
        atom.symbol, "|",
        atom.magmom, "|",
        atom.position,
        # atom.position, "|",
        )

# atoms_init.get_magnetic_moments()
atoms_init.get_initial_magnetic_moments()

np.array(init_magmoms)
