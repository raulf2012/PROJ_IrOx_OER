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

# # Setup initial *O slabs to run
# ---

# # Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

import json
import pickle
from shutil import copyfile

import numpy as np
import pandas as pd

from ase import io

from tqdm.notebook import tqdm
from IPython.display import display

# #########################################################
from methods import (
    get_df_slab,
    get_df_jobs,
    )

from proj_data import metal_atom_symbol

# #########################################################
from dft_workflow_methods import (
    get_job_spec_dft_params,
    get_job_spec_scheduler_params,
    submit_job,
    calc_wall_time)
# -

# # Script Inputs

# +
# Slac queue to submit to
slac_sub_queue = "suncat3"  # 'suncat', 'suncat2', 'suncat3'

# COMPENV to submit to
# compenv_i = "slac"
# compenv_i = "sherlock"
compenv_i = "nersc"
# -

# # Read Data

# +
# #########################################################
df_slab = get_df_slab()
df_slab = df_slab.set_index("slab_id")
df_slab_i = df_slab

# #########################################################
df_jobs = get_df_jobs()
# -

# ### Read `df_slabs_to_run` from `create_slabs.ipynb`, used to mark priority slabs

# +
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs",
    "out_data")

# #########################################################
import pickle; import os
path_i = os.path.join(
    directory,
    "df_slabs_to_run.pickle")
with open(path_i, "rb") as fle:
    df_slabs_to_run = pickle.load(fle)
# #########################################################


indices_not_good = []
for i_cnt, row_i in df_slabs_to_run.iterrows():
    df = df_slab_i
    df = df[
        (df["bulk_id"] == row_i.bulk_id) &
        (df["facet"] == row_i.facet_str) &
        [True for i in range(len(df))]
        ]
    if df.shape[0] == 0:
        indices_not_good.append(i_cnt)

df_slabs_to_run.loc[
    indices_not_good
    ]
# -


# # Selecting Slabs to Run

# +
# Dropping slabs that have been previously done
df_jobs_i = df_jobs[df_jobs.ads == "o"]
df_slab_i = df_slab_i.drop(
    df_jobs_i.slab_id.unique()
    )

# Doing only phase 2 slabs for now
df_slab_i = df_slab_i[df_slab_i.phase == 2]

# #########################################################
# Selecting smallest slabs
# df_slab_i = df_slab_i[df_slab_i.num_atoms < 80]

# print("Just doing XRD facets for now")
# df_slab_i = df_slab_i[df_slab_i.source == "xrd"]
# -

# ### Filtering down to best slabs, no layered, all octahedra, 0.3 eV/atom above hull cutoff

# +
good_slabs = []
for slab_id_i, row_i in df_slab_i.iterrows():
    # ####################################################
    bulk_id_i = row_i.bulk_id
    facet_i = row_i.facet
    # ####################################################

    # print("")
    # print(bulk_id_i, slab_id_i)

    df = df_slabs_to_run
    df = df[
        (df["bulk_id"] == bulk_id_i) &
        (df["facet_str"] == facet_i) &
        [True for i in range(len(df))]
        ]
    if df.shape[0] > 0:
        # print("Good")
        good_slabs.append(slab_id_i)

    # elif df.shape[0] == 0:
    #     print("Bad")

df_slab_i = df_slab_i.loc[
    good_slabs
    ]

# +
df = df_slab_i
df = df[
    (df["num_atoms"] <= 100) &
    # (df[""] == "") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]
df_slab_i = df

df_slab_i = df_slab_i.sort_values("num_atoms", ascending=False)
# -

df_slab_i

df_slab_i.index.tolist()

df_slab_i = df_slab_i.loc[
    [

        'legofufi_61',
        'gekawore_16',
        'mitilaru_63',

        # 'winomuvi_99',
        # 'letapivu_80',
        # 'giworuge_14',

        # 'lirilapa_78',
        # 'wakidowo_59',

        # 'kererape_22',
        # 'nekelele_74',
        # 'pebitiru_79',

        ]
    ]

df_slab_i

assert False

# # Setting up the job folders

# +
data_dict_list = []
for i_cnt, row_i in df_slab_i.iterrows():
    data_dict_i = dict()

    # #####################################################
    slab_id = row_i.name
    bulk_id = row_i.bulk_id
    facet = row_i.facet
    slab_final = row_i.slab_final
    num_atoms = row_i.num_atoms
    loop_time = row_i.loop_time
    iter_time_i = row_i.iter_time_i
    # #####################################################

    attempt = 1
    rev = 1


    # Checking if job dir exists for other comp. envs. (it shouldn't)
    job_exists_in_another_compenv = False
    path_already_exists = False
    for compenv_j in ["slac", "sherlock", "nersc", ]:
        
        path_j = os.path.join(
            os.environ["PROJ_irox_oer_gdrive"],
            "dft_workflow/run_slabs/run_o_covered/out_data/dft_jobs",
            compenv_j,
            bulk_id,
            facet,
            str(attempt).zfill(2) + "_attempt",
            "_" + str(rev).zfill(2)
            )
        if os.path.exists(path_j) and compenv_j == compenv_i:
            path_already_exists = True
            print("This path already exists", path_j)

        elif os.path.exists(path_j):
            job_exists_in_another_compenv = True
            print("Job exists in another COMPENV", path_j)

    good_to_go = True
    if job_exists_in_another_compenv:
        good_to_go = False
    if path_already_exists:
        good_to_go = False


    if good_to_go:
        path_i = os.path.join(
            os.environ["PROJ_irox_oer_gdrive"],
            "dft_workflow/run_slabs/run_o_covered/out_data/dft_jobs",
            compenv_i,
            bulk_id,
            facet,
            str(attempt).zfill(2) + "_attempt",
            "_" + str(rev).zfill(2)
            )

        print(path_i)
        if os.path.exists(path_i):
            print("TEMP | This path already exists and it shouldn't", path_i)

        if not os.path.exists(path_i):
            os.makedirs(path_i)


        # #####################################################
        # Copy dft script to job folder
        # #####################################################
        copyfile(
            os.path.join(
                os.environ["PROJ_irox_oer"],
                "dft_workflow/dft_scripts/slab_dft.py"
                ),
            os.path.join(
                path_i,
                "model.py",
                ),
            )

        copyfile(
            os.path.join(
                os.environ["PROJ_irox_oer"],
                "dft_workflow/dft_scripts/slab_dft.py"
                ),
            os.path.join(
                path_i,
                "slab_dft.py",
                ),
            )

        # #####################################################
        # Copy atoms object to job folder
        # #####################################################
        slab_final.write(
            os.path.join(path_i, "init.traj")
            )

        # #####################################################
        data_dict_i["slab_id"] = slab_id
        data_dict_i["bulk_id"] = bulk_id
        data_dict_i["facet"] = facet
        data_dict_i["slab_final"] = slab_final
        data_dict_i["num_atoms"] = num_atoms
        data_dict_i["attempt"] = attempt
        data_dict_i["rev"] = rev
        data_dict_i["path_i"] = path_i
        # #####################################################
        data_dict_list.append(data_dict_i)
        # #####################################################


# #########################################################
df_jobs_new = pd.DataFrame(data_dict_list)
df_jobs_new = df_jobs_new.set_index("slab_id")
# #########################################################
# -

# # Assigning job specific DFT parameters

# +
data_dict_list = []
for i_cnt, row_i in df_jobs_new.iterrows():
    data_dict_i = dict()
    # #####################################################
    slab_id = row_i.name
    num_atoms = row_i.num_atoms
    path_i =row_i.path_i
    # #####################################################

    dft_params_dict = get_job_spec_dft_params(
        compenv=compenv_i,
        slac_sub_queue="suncat3",
        )

    # #####################################################
    data_dict_i["slab_id"] = slab_id
    data_dict_i["dft_params"] = dft_params_dict
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

df_dft_params = pd.DataFrame(data_dict_list)
df_dft_params = df_dft_params.set_index("slab_id")



# #########################################################
# Writing DFT params to job directory
for slab_id, row_i in df_dft_params.iterrows():
    # #####################################################
    dft_params = row_i.dft_params
    # #####################################################
    row_slab_i = df_jobs_new.loc[slab_id]
    path_i = row_slab_i.path_i
    # #####################################################

    with open(os.path.join(path_i, "dft-params.json"), "w+") as fle:
        json.dump(dft_params, fle, indent=2, skipkeys=True)
# -

# # Setting initial magnetic moments

data_dict_list = []
for i_cnt, row_i in df_jobs_new.iterrows():
    # #####################################################
    atoms = row_i.slab_final
    path_i =row_i.path_i
    # #####################################################

    z_positions = atoms.positions[:, 2]
    z_max = z_positions.max()

    O_magmom=0.2
    M_magmom=0.6
    magmoms_i = []
    for atom in atoms:
        z_pos = atom.position[2]
        dist_from_top = z_max - z_pos
        # print(z_max - z_pos)

        if dist_from_top < 4:
            if atom.symbol == "O":
                magmom_i = O_magmom
            else:
                magmom_i = M_magmom
            magmoms_i.append(magmom_i)
        else:
            magmoms_i.append(0.)

    data_path = os.path.join(path_i, "magmoms.json")
    with open(data_path, "w") as outfile:
        json.dump(magmoms_i, outfile, indent=2)

print("Paths of new jobs:")
tmp = [print(i) for i in df_jobs_new.path_i.tolist()]

# #########################################################
print(20 * "# # ")
print("All done!")
print("setup_dft.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#

# + jupyter={"source_hidden": true}
# df_slab_i

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_slab_i

# + jupyter={"source_hidden": true}
# df_slab_i = df_slab_i.iloc[[0]]

# + jupyter={"source_hidden": true}
# assert False
