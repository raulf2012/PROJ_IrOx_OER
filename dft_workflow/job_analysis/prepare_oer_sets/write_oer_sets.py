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

# # Writing OER sets to file for
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import json

import pandas as pd
import numpy as np

# #########################################################
from methods import (
    get_df_features_targets,
    get_df_jobs,
    get_df_jobs_paths,
    get_df_atoms_sorted_ind,
    )
from methods import create_name_str_from_tup
from methods import get_df_jobs_paths, get_df_jobs_data

# #########################################################
from local_methods import write_other_jobs_in_set
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Read Data

# +
df_jobs = get_df_jobs()
df_jobs_paths = get_df_jobs_paths()
df_features_targets = get_df_features_targets()
df_atoms = get_df_atoms_sorted_ind()

df_jobs_paths = get_df_jobs_paths()
df_jobs_data = get_df_jobs_data()
# -

df_atoms = df_atoms.set_index("job_id")

# + active=""
#
#
#
# -

# ### Main loop | writing OER sets

# +
# # TEMP

# name_i = ('slac', 'wufulafe_03', 58.0)
# df_features_targets = df_features_targets.loc[[name_i]]

# +
# # TEMP
# print(111 * "TEMP | ")

# indices = [
#     # ('slac', 'relovalu_12', 24.0),

#     ('sherlock', 'sifebelo_94', 61.0),
#     # ('sherlock', 'sifebelo_94', 62.0),

#     ]

# df_features_targets = df_features_targets.loc[indices]

# +
# for name_i, row_i in df_features_targets.iterrows():

iterator = tqdm(df_features_targets.index, desc="1st loop")
for i_cnt, index_i in enumerate(iterator):
    row_i = df_features_targets.loc[index_i]

    #  if verbose:
    # print(name_i)

    # #####################################################
    job_id_o_i = row_i.data.job_id_o.iloc[0]
    job_id_bare_i = row_i.data.job_id_bare.iloc[0]
    job_id_oh_i = row_i.data.job_id_oh.iloc[0]
    # #####################################################

    if job_id_bare_i is None:
        continue

    oh_exists = False
    if job_id_oh_i is not None:
        oh_exists = True

    # #####################################################
    df_atoms__o = df_atoms.loc[job_id_o_i]
    df_atoms__bare = df_atoms.loc[job_id_bare_i]

    # #####################################################
    atoms__o = df_atoms__o.atoms_sorted_good
    atoms__bare = df_atoms__bare.atoms_sorted_good

    if oh_exists:
        df_atoms__oh = df_atoms.loc[job_id_oh_i]
        atoms__oh = df_atoms__oh.atoms_sorted_good

    # #########################################################
    # #########################################################
    # dir_name = create_name_str_from_tup(name_i)
    dir_name = create_name_str_from_tup(index_i)

    dir_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/prepare_oer_sets",
        "out_data/oer_group_files",
        dir_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    # #####################################################
    atoms__o.write(
        os.path.join(dir_path, "atoms__o.traj"))

    atoms__o.write(
        os.path.join(dir_path, "atoms__o.cif"))

    atoms__bare.write(
        os.path.join(dir_path, "atoms__bare.traj"))
    atoms__bare.write(
        os.path.join(dir_path, "atoms__bare.cif"))

    if oh_exists:
        atoms__oh.write(
            os.path.join(dir_path, "atoms__oh.traj"))
        atoms__oh.write(
            os.path.join(dir_path, "atoms__oh.cif"))


    # #####################################################
    data_dict_to_write = dict(
        job_id_o=job_id_o_i,
        job_id_bare=job_id_bare_i,
        job_id_oh=job_id_oh_i,
        )

    data_path = os.path.join(dir_path, "data.json")
    with open(data_path, "w") as outfile:
        json.dump(data_dict_to_write, outfile, indent=2)


    # #####################################################
    # Write other jobs in OER set
    write_other_jobs_in_set(
        job_id_bare_i,
        dir_path=dir_path,
        df_jobs=df_jobs, df_atoms=df_atoms,
        df_jobs_paths=df_jobs_paths,
        df_jobs_data=df_jobs_data,
        )
# -

import ase
ase.__version__

atoms__o

assert False

# + active=""
#
#
# -

# # Writing top systems to file ROUGH TEMP

# +
# TOP SYSTEMS

if False:
# if True:
    df_features_targets = df_features_targets.loc[
        [

            ("slac", "tefovuto_94", 16.0),
#             slac__nifupidu_92__032
#             sherlock__bihetofu_24__036

            ('slac', 'hobukuno_29', 16.0),
            ('sherlock', 'ramufalu_44', 56.0),
            ('slac', 'nifupidu_92', 32.0),
            ('sherlock', 'bihetofu_24', 36.0),
            ('slac', 'dotivela_46', 32.0),
            ('slac', 'vovumota_03', 33.0),
            ('slac', 'ralutiwa_59', 32.0),
            ('sherlock', 'bebodira_65', 16.0),
            ('sherlock', 'soregawu_05', 62.0),
            ('slac', 'hivovaru_77', 26.0),
            ('sherlock', 'vegarebo_06', 50.0),
            ('slac', 'ralutiwa_59', 30.0),
            ('sherlock', 'kamevuse_75', 49.0),
            ('nersc', 'hesegula_40', 94.0),
            ('slac', 'fewirefe_11', 39.0),
            ('sherlock', 'vipikema_98', 60.0),
            ('slac', 'gulipita_22', 48.0),
            ('sherlock', 'rofetaso_24', 48.0),
            ('slac', 'runopeno_56', 32.0),
            ('slac', 'magiwuni_58', 26.0),
            ]
        ]

    for name_i, row_i in df_features_targets.iterrows():

        # #####################################################
        job_id_o_i = row_i.data.job_id_o.iloc[0]
        job_id_bare_i = row_i.data.job_id_bare.iloc[0]
        job_id_oh_i = row_i.data.job_id_oh.iloc[0]
        # #####################################################

        oh_exists = False
        if job_id_oh_i is not None:
            oh_exists = True

        # #####################################################
        df_atoms__o = df_atoms.loc[job_id_o_i]
        df_atoms__bare = df_atoms.loc[job_id_bare_i]

        # #####################################################
        atoms__o = df_atoms__o.atoms_sorted_good
        atoms__bare = df_atoms__bare.atoms_sorted_good

        if oh_exists:
            df_atoms__oh = df_atoms.loc[job_id_oh_i]
            atoms__oh = df_atoms__oh.atoms_sorted_good

        # #########################################################
        # #########################################################
        dir_name = create_name_str_from_tup(name_i)

        dir_path = os.path.join(
            os.environ["PROJ_irox_oer"],
            "dft_workflow/job_analysis/prepare_oer_sets",
            "out_data/top_overpot_sys")
            # dir_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # atoms__o.write(
        #     os.path.join(dir_path, dir_name + "_o.cif"))

        # atoms__bare.write(
        #     os.path.join(dir_path, dir_name + "_bare.cif"))

        if oh_exists:
            atoms__oh.write(
                os.path.join(dir_path, dir_name + "_oh.cif"))
# -

# # MISC | Writing random cifs to file to open in VESTA

# +
df_subset = df_features_targets.sample(n=6)

if False:
    for name_i, row_i in df_subset.iterrows():
        tmp = 42

        job_id_oh_i = row_i[("data", "job_id_oh", "", )]


        # # #####################################################
        # job_id_o_i = row_i.data.job_id_o.iloc[0]
        # job_id_bare_i = row_i.data.job_id_bare.iloc[0]
        # job_id_oh_i = row_i.data.job_id_oh.iloc[0]
        # # #####################################################

        # if job_id_bare_i is None:
        #     continue

        oh_exists = False
        if job_id_oh_i is not None:
            oh_exists = True

        # # #####################################################
        # df_atoms__o = df_atoms.loc[job_id_o_i]
        # df_atoms__bare = df_atoms.loc[job_id_bare_i]

        # # #####################################################
        # atoms__o = df_atoms__o.atoms_sorted_good
        # atoms__bare = df_atoms__bare.atoms_sorted_good

        if oh_exists:
            df_atoms__oh = df_atoms.loc[job_id_oh_i]
            atoms__oh = df_atoms__oh.atoms_sorted_good

        # #########################################################
        # #########################################################
        file_name_i = create_name_str_from_tup(name_i)
        print(file_name_i)

        dir_path = os.path.join(
            os.environ["PROJ_irox_oer"],
            "dft_workflow/job_analysis/prepare_oer_sets",
            "out_data/misc_cif_files_oh")
            # dir_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


        # #####################################################
        # atoms__o.write(
        #     os.path.join(dir_path, "atoms__o.traj"))

        # atoms__o.write(
        #     os.path.join(dir_path, "atoms__o.cif"))

        # atoms__bare.write(
        #     os.path.join(dir_path, "atoms__bare.traj"))
        # atoms__bare.write(
        #     os.path.join(dir_path, "atoms__bare.cif"))

        if oh_exists:
            atoms__oh.write(
                os.path.join(dir_path, file_name_i + ".cif"))

                # os.path.join(dir_path, "atoms__oh.traj"))

            # atoms__oh.write(
            #     os.path.join(dir_path, "atoms__oh.cif"))
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("write_oer_sets.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# import os
# print(os.getcwd())
# import sys

# import pickle

# pd.set_option('display.max_columns', None)
# # pd.set_option('display.max_rows', None)
