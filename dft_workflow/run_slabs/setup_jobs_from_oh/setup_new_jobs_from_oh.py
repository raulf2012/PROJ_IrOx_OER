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

# # Setup new jobs to resubmit *O (and possibly *) from *OH to achieve better magmom matching
# ---

# # Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

import copy

import numpy as np
import pandas as pd

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

# +
import json

from dft_workflow_methods import get_job_spec_dft_params

from shutil import copyfile
# -

# # Read Data

# + jupyter={"source_hidden": true}
df_jobs = get_df_jobs()

df_oer_groups = get_df_oer_groups()

df_jobs_oh_anal = get_df_jobs_oh_anal()

df_rerun_from_oh = get_df_rerun_from_oh()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_jobs_paths = get_df_jobs_paths()

# + active=""
#
#

# +
df_rerun_from_oh_i = df_rerun_from_oh[df_rerun_from_oh.rerun_from_oh == True]

print(5 * "TEMP ")
df_rerun_from_oh_i = df_rerun_from_oh_i.iloc[[-1]]

# #########################################################
data_dict_list_o = []
data_dict_list_bare = []
# #########################################################
for i_cnt, row_i in df_rerun_from_oh_i.iterrows():
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

    unique_att_nums_o = list(df_jobs_o_i.att_num.unique())
    new_att_num_o = np.max(unique_att_nums_o) + 1

    df_jobs_oh_i = df_jobs_i[
        (df_jobs_i.ads == "bare") & \
        (df_jobs_i.active_site == active_site_i) & \
        [True for i in range(len(df_jobs_i))]
        ]
    unique_att_nums_bare = list(df_jobs_oh_i.att_num.unique())
    new_att_num_bare = np.max(unique_att_nums_bare) + 1

    unique_bulk_ids = list(df_jobs_i.bulk_id.unique())
    mess_i = "iSSJfi"
    assert len(unique_bulk_ids) == 1, mess_i
    bulk_id_i = unique_bulk_ids[0]

    unique_facets = list(df_jobs_i.facet.unique())
    mess_i = "iSSJfi"
    assert len(unique_facets) == 1, mess_i
    facet_i = unique_facets[0]



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



    # #########################################################
    # Creating new directories
    new_o_path = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        "dft_workflow/run_slabs",
        "run_o_covered/out_data/dft_jobs",
        compenv_i, bulk_id_i, facet_i,
        str(new_att_num_o).zfill(2) + "_attempt",
        "_01",
        )


    if not os.path.exists(new_o_path):
        os.makedirs(new_o_path)

        # #############################################
        # Copy dft script to job folder
        copyfile(
            os.path.join(os.environ["PROJ_irox_oer"], "dft_workflow/dft_scripts/slab_dft.py"),
            os.path.join(new_o_path, "model.py"),
            )

        # #############################################
        # Copy atoms object to job folder
        atoms_O.write(
            os.path.join(new_o_path, "init.traj")
            )
        num_atoms_i = atoms_O.get_global_number_of_atoms()

        # #############################################
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

        slac_sub_queue_i = "suncat2"
        dft_params_i = get_job_spec_dft_params(
            compenv=compenv_i,
            slac_sub_queue=slac_sub_queue_i,
            )
        dft_params_i["ispin"] = 2

        # #################################################
        with open(os.path.join(new_bare_path, "dft-params.json"), "w+") as fle:
            json.dump(dft_params_i, fle, indent=2, skipkeys=True)
        # #################################################
        with open(os.path.join(new_o_path, "dft-params.json"), "w+") as fle:
            json.dump(dft_params_i, fle, indent=2, skipkeys=True)
# -

atoms_i.get_initial_magnetic_moments()


# +
atoms_i.get_initial_magnetic_moments()


# atoms_i.set_initial_magnetic_moments(magmoms_sorted_good_i)

atoms_i.set_initial_magnetic_moments(
    atoms_i.get_magnetic_moments()
    )
# -

atoms_i.get_initial_magnetic_moments()

# + active=""
#
#
