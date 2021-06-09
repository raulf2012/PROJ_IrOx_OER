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

# # Setup DOS/Bader Jobs for Finished OER Sets
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

from pathlib import Path
import shutil
from shutil import copyfile
import json

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
# pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100

# #########################################################
from methods import (
    get_df_jobs,
    get_df_features_targets,
    get_df_jobs_data,
    get_df_atoms_sorted_ind,
    get_df_jobs_paths,
    )
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
df_features_targets = get_df_features_targets()
df_features_targets_i = df_features_targets

df_jobs = get_df_jobs()

df_jobs_data = get_df_jobs_data()

df_atoms = get_df_atoms_sorted_ind()

df_paths = get_df_jobs_paths()
# -

# ### Preprocess `df_features_targets`

# +
# df_features_targets_i = df_features_targets_i[
#     df_features_targets_i.data.all_done == True]

df = df_features_targets_i
df = df[
    (df["data", "all_done"] == True) &
    (df["data", "from_oh__o"] == True) &
    # (df["data", "from_oh__bare"] == True) &
    [True for i in range(len(df))]
    ]
df_features_targets_i = df
# -

# ### Figuring which systems to process

# +
# #########################################################
data_dict_dict = dict()
indices_to_process = []
# #########################################################
for i_cnt, (ind_i, row_i) in enumerate(df_features_targets_i.iterrows()):

    # #####################################################
    compenv_i = ind_i[0]
    slab_id_i = ind_i[1]
    active_site_i = ind_i[2]
    # #####################################################
    job_id_o_i = row_i[("data", "job_id_o", "", )]
    # #####################################################


    # #####################################################
    row_jobs_i = df_jobs.loc[job_id_o_i]
    # #####################################################
    active_site_i = row_jobs_i.active_site
    bulk_id_i = row_jobs_i.bulk_id
    facet_i = row_jobs_i.facet
    # #####################################################

    # #####################################################
    row_data_i = df_jobs_data.loc[job_id_o_i]
    # #####################################################
    rerun_from_oh_i = row_data_i.rerun_from_oh
    # #####################################################

    assert rerun_from_oh_i, "filtering by all_done should mean that all *O are rerun from *OH"
    # TEMP
    # print(active_site_i)

    assert active_site_i != "NaN", "Active site should be number, rerun from *OH so should have one"

    # #####################################################
    # Creating new directories
    path_new_i = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        "dft_workflow/run_dos_bader",
        "run_o_covered/out_data/dft_jobs",
        compenv_i, bulk_id_i, facet_i,
        "active_site__" + str(int(active_site_i)),
        str(1).zfill(2) + "_attempt",
        "_01",
        # "_01.tmp",
        )

    my_file = Path(path_new_i)
    path_does_not_exist = False
    if not my_file.is_dir():
        if verbose:
            print("i_cnt:", i_cnt, "index_i:", ind_i)
            # print("path_new_i:", path_new_i)

        path_does_not_exist = True

    if path_does_not_exist:
        indices_to_process.append(ind_i)

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i["path_new"] = path_new_i
    data_dict_i["job_id_o"] = job_id_o_i
    data_dict_i["att_num"] = 1
    # #####################################################
    data_dict_dict[ind_i] = data_dict_i
    # #####################################################

# #########################################################
df_features_targets_i_2 = df_features_targets_i.loc[indices_to_process]
# #########################################################
# -

df_features_targets_i_2.shape

# +
# assert False
# -

# ### Main Loop

# +
# #########################################################
data_dict_list = []
# #########################################################
for ind_i, row_i in df_features_targets_i_2.iterrows():

    # #####################################################
    compenv_i = ind_i[0]
    slab_id_i = ind_i[1]
    active_site_i = ind_i[2]
    # #####################################################
    data_dict_prev = data_dict_dict[ind_i]
    # #####################################################
    path_new_i = data_dict_prev["path_new"]
    job_id_o_i = data_dict_prev["job_id_o"]
    att_num_i = data_dict_prev["att_num"]
    # #####################################################

    # #####################################################
    row_jobs_i = df_jobs.loc[job_id_o_i]
    # #####################################################
    active_site_i = row_jobs_i.active_site
    bulk_id_i = row_jobs_i.bulk_id
    facet_i = row_jobs_i.facet
    att_num_orig_i = row_jobs_i.att_num
    # #####################################################

    # #####################################################
    # atoms_name_i = (compenv_i, slab_id_i, "o", active_site_i, att_num_orig_i, )
    atoms_name_i = ("oer_adsorbate", compenv_i, slab_id_i, "o", active_site_i, att_num_orig_i, )
    row_atoms_i = df_atoms.loc[atoms_name_i]
    # #####################################################
    atoms_i = row_atoms_i.atoms_sorted_good
    was_sorted_i = row_atoms_i.was_sorted
    # #####################################################

    # #####################################################
    row_paths_i = df_paths.loc[job_id_o_i]
    # #####################################################
    gdrive_path_i = row_paths_i.gdrive_path
    # #####################################################

    if was_sorted_i:
        magmoms_sorted_good = row_atoms_i.magmoms_sorted_good
        magmoms_i = magmoms_sorted_good
    else:
        magmoms_i = atoms_i.get_magnetic_moments()


    if not os.path.exists(path_new_i):
        os.makedirs(path_new_i)

        # #################################################
        # Copy dft script to job folder
        copyfile(
            os.path.join(
                os.environ["PROJ_irox_oer"],
                "dft_workflow/dft_scripts/dos_scripts",
                "model_dos.py",
                ),
            os.path.join(path_new_i, "model.py"),
            )
        copyfile(
            os.path.join(
                os.environ["PROJ_irox_oer"],
                "dft_workflow/dft_scripts/dos_scripts",
                "model_dos.py",
                ),
            os.path.join(path_new_i, "model_dos.py"),
            )

        # #################################################
        # Copy dos__calc_settings.py to job folder
        copyfile(
            os.path.join(
                os.environ["PROJ_irox_oer"],
                "dft_workflow/dft_scripts/dos_scripts",
                "dos__calc_settings.py",
                ),
            os.path.join(path_new_i, "dos__calc_settings.py"),
            )


        atoms_i.set_initial_magnetic_moments(magmoms_i)
        atoms_i.write(os.path.join(path_new_i, "init.traj"))

        num_atoms_i = atoms_i.get_global_number_of_atoms()


        # ---------------------------------------
        # Moving dft-params.json file to new dir
        file__dft_params = os.path.join(
            os.environ["PROJ_irox_oer_gdrive"],
            gdrive_path_i,
            "dft-params.json",
            )

        copyfile(
            file__dft_params,
            os.path.join(path_new_i, "dft-params.json"),
            )

        # #################################################
        data_dict_i["compenv"] = compenv_i
        data_dict_i["slab_id"] = slab_id_i
        data_dict_i["bulk_id"] = bulk_id_i
        data_dict_i["att_num"] = att_num_i
        data_dict_i["rev_num"] = 1
        data_dict_i["facet"] = facet_i
        data_dict_i["slab"] = atoms_i
        data_dict_i["num_atoms"] = num_atoms_i
        data_dict_i["path_new"] = path_new_i
        data_dict_i["job_id_orig"] = job_id_o_i
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

        data_path = os.path.join(path_new_i, "data_dict.json")
        with open(data_path, "w") as outfile:
            json.dump(dict(job_id_orig=job_id_o_i), outfile, indent=2)



# #########################################################
df = pd.DataFrame(data_dict_list)
# #########################################################


# if True:
#     shutil.rmtree(path_new_i)
# -

path_new_i

# + active=""
#
#
#
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("setup_dos_bader.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_features_targets_i.loc[
#     # ('nersc', 'gekawore_16', 84.0)
#     # ('nersc', 'gekawore_16', 81.0)
#     ('nersc', 'giworuge_14', 81.0)
#     ]["data"]

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_features_targets_i.loc[
#     # ('nersc', 'gekawore_16', 84.0)
#     ('nersc', 'gekawore_16', 81.0)
#     ]

# + jupyter={"source_hidden": true}
# df_features_targets_i.loc[
#     # ('nersc', 'gekawore_16', 84.0)
#     # ('nersc', 'gekawore_16', 81.0)
#     ('nersc', 'giworuge_14', 81.0)
#     ]
