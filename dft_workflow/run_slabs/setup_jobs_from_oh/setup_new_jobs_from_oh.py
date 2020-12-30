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

# # Setup new jobs to resubmit *O (and possibly *) from *OH to achieve better magmom matching
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import copy
import json
from shutil import copyfile

import numpy as np
import pandas as pd
# pd.set_option("display.max_columns", None)
# pd.set_option('display.max_rows', None)
pd.options.display.max_colwidth = 130

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_anal,
    get_df_oer_groups,
    get_df_jobs_oh_anal,
    get_df_rerun_from_oh,
    get_df_atoms_sorted_ind,
    )
from methods import get_df_coord
from methods import get_df_jobs_paths
from methods import get_df_jobs_data
from methods import get_other_job_ids_in_set
from methods import get_df_slab

# #########################################################
from dft_workflow_methods import get_job_spec_dft_params
# -
from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# # Read Data

# +
df_jobs = get_df_jobs()

df_oer_groups = get_df_oer_groups()

df_jobs_oh_anal = get_df_jobs_oh_anal()

df_rerun_from_oh = get_df_rerun_from_oh()
df_rerun_from_oh_i = df_rerun_from_oh

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_jobs_paths = get_df_jobs_paths()

df_jobs_data = get_df_jobs_data()

df_slab = get_df_slab()

# + active=""
#
#
# -

df_rerun_from_oh

# +
# assert False
# -

# ### Only setting up jobs for slab phase > 1

# +
df_slab_i = df_slab[df_slab.phase > 1]

if verbose:
    print(df_rerun_from_oh_i.shape[0])

df_rerun_from_oh_i = df_rerun_from_oh_i.loc[
    df_rerun_from_oh_i.slab_id.isin(df_slab_i.index)
    ]
if verbose:
    print(df_rerun_from_oh_i.shape[0])

df_rerun_from_oh_i = df_rerun_from_oh_i[df_rerun_from_oh_i.rerun_from_oh == True]
if verbose:
    print(df_rerun_from_oh_i.shape[0])
# -

df_rerun_from_oh_i = df_rerun_from_oh_i.reset_index()

df_rerun_from_oh_i

# +
# #########################################################
data_dict_dict = dict()
indices_to_process = []
# #########################################################
for i_cnt, row_i in df_rerun_from_oh_i.iterrows():
    data_dict_i = dict()
    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    active_site_i = row_i.active_site
    job_id_most_stable_i = row_i.job_id_most_stable
    # #####################################################

    # #####################################################
    row_jobs_i = df_jobs.loc[job_id_most_stable_i]
    # #####################################################
    att_num_i = row_jobs_i.att_num
    # #####################################################

    # #####################################################
    # Getting the att_num values for *O  and bare jobs so new ones can be assigned 
    df_jobs_i = df_jobs[
        (df_jobs.compenv == compenv_i) & \
        (df_jobs.slab_id == slab_id_i) & \
        [True for i in range(len(df_jobs))]
        ]
    df_jobs_o_i = df_jobs_i[
        (df_jobs_i.ads == "o")
        ]

    df_jobs_bare_i = df_jobs_i[
        (df_jobs_i.ads == "bare") & \
        (df_jobs_i.active_site == active_site_i) & \
        [True for i in range(len(df_jobs_i))]
        ]

    df_jobs_data_bare_i = df_jobs_data.loc[
        df_jobs_bare_i.index
        ]

    df_restart_from_oh_i = df_jobs_data_bare_i[df_jobs_data_bare_i.rerun_from_oh == True]

    job_ids_from_oh_restarted_jobs = []
    for job_id_j in df_restart_from_oh_i.index.tolist():

        tmp = get_other_job_ids_in_set(
            job_id_j,
            df_jobs=df_jobs,
            )
        job_ids_in_set = tmp.job_id.tolist()
        job_ids_from_oh_restarted_jobs.extend(job_ids_in_set)

    df_jobs_bare_i_2 = df_jobs_bare_i.drop(labels=job_ids_from_oh_restarted_jobs)

    unique_att_nums_bare = list(df_jobs_bare_i_2.att_num.unique())
    new_att_num_bare = np.max(unique_att_nums_bare) + 1

    # print("new_att_num_bare:", new_att_num_bare )



    unique_bulk_ids = list(df_jobs_i.bulk_id.unique())
    mess_i = "iSSJfi"
    assert len(unique_bulk_ids) == 1, mess_i
    bulk_id_i = unique_bulk_ids[0]

    unique_facets = list(df_jobs_i.facet.unique())
    mess_i = "iSSJfi"
    assert len(unique_facets) == 1, mess_i
    facet_i = unique_facets[0]


    # #####################################################
    # Creating new directories
    new_o_path = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        "dft_workflow/run_slabs",
        "run_o_covered/out_data/dft_jobs",
        compenv_i, bulk_id_i, facet_i,
        "active_site__" + str(int(active_site_i)),
        str(1).zfill(2) + "_attempt",
        "_01",
        )


    new_bare_path = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        "dft_workflow/run_slabs",
        "run_bare_oh_covered/out_data/dft_jobs",
        compenv_i, bulk_id_i, facet_i,
        "bare",
        "active_site__" + str(int(active_site_i)),
        str(new_att_num_bare).zfill(2) + "_attempt",
        "_01",
        )


    from pathlib import Path

    my_file = Path(new_o_path)
    o_path_does_not_exist = False
    if not my_file.is_dir():
        o_path_does_not_exist = True
        
    my_file = Path(new_bare_path)
    bare_path_does_not_exist = False
    if not my_file.is_dir():
        bare_path_does_not_exist = True
        
    if o_path_does_not_exist or bare_path_does_not_exist:
        indices_to_process.append(i_cnt)


    data_dict_i["new_bare_path"] = new_bare_path
    data_dict_i["new_o_path"] = new_o_path
    data_dict_dict[i_cnt] = data_dict_i

df_rerun_from_oh_i_2 = df_rerun_from_oh_i.loc[indices_to_process]

# +
# df_rerun_from_oh_i_2 = df_rerun_from_oh_i_2.iloc[[0]]
# -

print(
    df_rerun_from_oh_i_2.shape[0],
    " new jobs are being set up",
    sep="")

# +
# df_rerun_from_oh_i_2

# +
# assert False
# -

# ### Create directories and initialize

# #########################################################
data_dict_list_o = []
data_dict_list_bare = []
# #########################################################
for i_cnt, row_i in df_rerun_from_oh_i_2.iterrows():
    # print("i_cnt:", i_cnt)

    data_dict_o_i = dict()
    data_dict_bare_i = dict()

    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    active_site_i = row_i.active_site
    job_id_most_stable_i = row_i.job_id_most_stable
    # #####################################################

    # #########################################################
    row_jobs_i = df_jobs.loc[job_id_most_stable_i]
    # #########################################################
    att_num_i = row_jobs_i.att_num
    # #########################################################

    # #########################################################
    # #########################################################
    # #########################################################

    # #########################################################
    idx_i = pd.IndexSlice[compenv_i, slab_id_i, "oh", active_site_i, att_num_i]
    row_atoms_i = df_atoms_sorted_ind.loc[idx_i, :]
    # #########################################################
    atoms_i = row_atoms_i.atoms_sorted_good
    magmoms_sorted_good_i = row_atoms_i.magmoms_sorted_good
    # #########################################################

    if atoms_i.calc is None:
        if magmoms_sorted_good_i is not None:
            atoms_i.set_initial_magnetic_moments(magmoms_sorted_good_i)
        else:
            print("Not good there should be something here")
    else:
        atoms_i.set_initial_magnetic_moments(
            atoms_i.get_magnetic_moments()
            )

    # #########################################################
    from local_methods import get_bare_o_from_oh
    bare_o_out_dict =  get_bare_o_from_oh(
        compenv=compenv_i,
        slab_id=slab_id_i,
        active_site=active_site_i,
        att_num=att_num_i,
        atoms=atoms_i,
        )
    atoms_bare = bare_o_out_dict["atoms_bare"]
    atoms_O = bare_o_out_dict["atoms_O"]

    write_atoms = True
    if write_atoms:
        atoms_i.write("__temp__/oh.traj")
        atoms_bare.write("__temp__/bare.traj")
        atoms_O.write("__temp__/o.traj")

    # #########################################################
    # Getting the att_num values for *O  and bare jobs so new ones can be assigned 
    df_jobs_i = df_jobs[
        (df_jobs.compenv == compenv_i) & \
        (df_jobs.slab_id == slab_id_i) & \
        # (df_jobs.active_site == active_site_i) & \
        [True for i in range(len(df_jobs))]
        ]
    df_jobs_o_i = df_jobs_i[
        (df_jobs_i.ads == "o")
        ]

    # unique_att_nums_o = list(df_jobs_o_i.att_num.unique())
    # new_att_num_o = np.max(unique_att_nums_o) + 1

    df_jobs_data_o_i = df_jobs_data.loc[
        df_jobs_o_i.index
        ]

    df_restart_from_oh_i = df_jobs_data_o_i[df_jobs_data_o_i.rerun_from_oh == True]

    job_ids_from_oh_restarted_jobs = []
    for job_id_j in df_restart_from_oh_i.index.tolist():

        tmp = get_other_job_ids_in_set(
            job_id_j,
            df_jobs=df_jobs,
            )
        job_ids_in_set = tmp.job_id.tolist()
        job_ids_from_oh_restarted_jobs.extend(job_ids_in_set)

    df_jobs_o_i_2 = df_jobs_o_i.drop(labels=job_ids_from_oh_restarted_jobs)

    unique_att_nums_o = list(df_jobs_o_i_2.att_num.unique())
    new_att_num_o = np.max(unique_att_nums_o) + 1








    # #####################################################
    data_dict_i = data_dict_dict[i_cnt]
    # #####################################################
    new_o_path = data_dict_i["new_o_path"]
    new_bare_path = data_dict_i["new_bare_path"]
    # #####################################################

    print(new_o_path)
    print(new_bare_path)








    if not os.path.exists(new_o_path):
        os.makedirs(new_o_path)

        # #################################################
        # Copy dft script to job folder
        copyfile(
            os.path.join(os.environ["PROJ_irox_oer"], "dft_workflow/dft_scripts/slab_dft.py"),
            os.path.join(new_o_path, "model.py"),
            )

        # #################################################
        # Copy atoms object to job folder
        atoms_O.write(
            os.path.join(new_o_path, "init.traj")
            )
        num_atoms_i = atoms_O.get_global_number_of_atoms()

        # #################################################
        data_dict_o_i["compenv"] = compenv_i
        data_dict_o_i["slab_id"] = slab_id_i
        data_dict_o_i["bulk_id"] = bulk_id_i
        data_dict_o_i["att_num"] = new_att_num_o
        data_dict_o_i["rev_num"] = 1
        data_dict_o_i["active_site"] = "NaN"
        data_dict_o_i["facet"] = facet_i
        data_dict_o_i["slab"] = atoms_O
        data_dict_o_i["num_atoms"] = num_atoms_i
        # data_dict_i["path_i"] = path_i
        data_dict_o_i["path_full"] = new_o_path
        # #############################################
        data_dict_list_o.append(data_dict_o_i)
        # #############################################

    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################



    if not os.path.exists(new_bare_path):
        os.makedirs(new_bare_path)

        # #############################################
        # Copy dft script to job folder
        copyfile(
            os.path.join(os.environ["PROJ_irox_oer"], "dft_workflow/dft_scripts/slab_dft.py"),
            os.path.join(new_bare_path, "model.py"),
            )

        # #############################################
        # Copy atoms object to job folder
        atoms_bare.write(
            os.path.join(new_bare_path, "init.traj")
            )
        num_atoms_i = atoms_bare.get_global_number_of_atoms()

        # #############################################
        data_dict_bare_i["compenv"] = compenv_i
        data_dict_bare_i["slab_id"] = slab_id_i
        data_dict_bare_i["bulk_id"] = bulk_id_i
        data_dict_bare_i["att_num"] = new_att_num_o
        data_dict_bare_i["rev_num"] = 1
        data_dict_bare_i["active_site"] = "NaN"
        data_dict_bare_i["facet"] = facet_i
        data_dict_bare_i["slab"] = atoms_bare
        data_dict_bare_i["num_atoms"] = num_atoms_i
        data_dict_bare_i["path_full"] = new_bare_path
        # #############################################
        data_dict_list_bare.append(data_dict_bare_i)
        # #############################################






    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # #####################################################
    # Writing data_dict to mark that these were rerun from *OH
    data_dict_out = dict(rerun_from_oh=True)

    # #####################################################
    slac_sub_queue_i = "suncat3"
    dft_params_i = get_job_spec_dft_params(
        compenv=compenv_i,
        slac_sub_queue=slac_sub_queue_i,
        )
    dft_params_i["ispin"] = 2

    if os.path.exists(new_o_path):
        # #################################################
        with open(os.path.join(new_o_path, "data_dict.json"), "w+") as fle:
            json.dump(data_dict_out, fle, indent=2, skipkeys=True)
        # #################################################
        with open(os.path.join(new_o_path, "dft-params.json"), "w+") as fle:
            json.dump(dft_params_i, fle, indent=2, skipkeys=True)
        # #################################################

    if os.path.exists(new_bare_path):
        # #################################################
        with open(os.path.join(new_bare_path, "data_dict.json"), "w+") as fle:
            json.dump(data_dict_out, fle, indent=2, skipkeys=True)
        # #################################################
        with open(os.path.join(new_bare_path, "dft-params.json"), "w+") as fle:
            json.dump(dft_params_i, fle, indent=2, skipkeys=True)
        # #################################################

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("setup_new_jobs_from_oh.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
