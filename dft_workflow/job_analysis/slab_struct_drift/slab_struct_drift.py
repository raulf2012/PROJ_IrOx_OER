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

# # Structural similarity of before/after relaxation
# ---
# This will allow us to quantify the degree of structural drift

# # Import Modules

from pathlib import Path

from local_methods import get_ave_drift__wrapper

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

# import pickle
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ase import io

from IPython.display import display

# #########################################################
from StructurePrototypeAnalysisPackage.ccf import cal_ccf_d

# #########################################################
from methods import get_df_jobs, get_df_jobs_data
from methods import get_df_init_slabs
from methods import get_df_jobs_anal
from methods import get_df_jobs_paths
from methods import get_df_atoms_sorted_ind

# #########################################################
from local_methods import get_ccf
from local_methods import get_all_ccf_data
from local_methods import remove_constrained_atoms
from local_methods import check_ccf_data_present
from local_methods import get_ave_drift
# -

# # Script Inputs

# +
verbose = True

r_cut_off = 10
r_vector = np.arange(0.06, 10, 0.02)

# r_cut_off = 10
# r_vector = np.arange(1, 10, 0.005)

# r_cut_off = 30
# r_vector = np.arange(0, 30, 0.01)
# -

# # Read Data

# +
df_jobs = get_df_jobs()
df_jobs_i = df_jobs

df_jobs_data = get_df_jobs_data()

df_init_slabs = get_df_init_slabs()

df_jobs_anal = get_df_jobs_anal()

df_jobs_paths = get_df_jobs_paths()


df_atoms_sorted_ind = get_df_atoms_sorted_ind()
# -

df_jobs_i = df_jobs_i.drop(columns=[
    "job_id",
    "facet",
    "num_revs",
    "compenv_origin",
    "submitted",
    ])

# +
# # TEMP

# # ('sherlock', 'kesekodi_38', 'o', 'NaN', 1)

# # Exploded job
# # name_i = ("slac", "relovalu_12", "oh", 24.0, 2, )

# # name_i = ("nersc", "kalisule_45", "bare", 73.0, 1, )

# # Most disimilar slab, not the one that exploded weirdly enough
# # name_i = ("nersc", "horikovi_77", "o", "NaN", 1, )

# name_i = ('slac', 'relovalu_12', 'oh', 24.0, 2)

# df = df_jobs
# df = df[
#     (df["compenv"] == name_i[0]) &
#     (df["slab_id"] == name_i[1]) &
#     (df["ads"] == name_i[2]) &
#     (df["active_site"] == name_i[3]) &
#     (df["att_num"] == name_i[4]) &
#     [True for i in range(len(df))]
#     ]
# df_jobs_i = df

# +
# df_jobs_i

# +
# assert False

# +
group_cols = ["compenv", "slab_id", "ads", "active_site", "att_num", ]
grouped = df_jobs_i.groupby(group_cols)
# #########################################################
groups_list = []
for i_cnt, (name_i, group_i) in enumerate(grouped):
    groups_list.append(group_i)

len(groups_list)
# -

# #########################################################
group_cols = ["compenv", "slab_id", "ads", "active_site", "att_num", ]
grouped = df_jobs_i.groupby(group_cols)
# #########################################################
for i_cnt, (name_i, group_i) in enumerate(grouped):

    init_true = check_ccf_data_present(
        name_tup=name_i,
        init_or_final="init",
        intact=True)

    init_false = check_ccf_data_present(
        name_tup=name_i,
        init_or_final="init",
        intact=False)

    final_true = check_ccf_data_present(
        name_tup=name_i,
        init_or_final="final",
        intact=True)

    final_false = check_ccf_data_present(
        name_tup=name_i,
        init_or_final="final",
        intact=False)

    all_files_present = False
    if init_false and init_true and final_false and final_true:
        all_files_present = True

# #####################################################
row_atoms_sorted_i = df_atoms_sorted_ind[
    df_atoms_sorted_ind.job_id == job_id_max]
# row_atoms_sorted_i = row_atoms_sorted_i.iloc[0]
# # #####################################################
# atoms_final_sorted = row_atoms_sorted_i.atoms_sorted_good
# failed_to_sort_i = row_atoms_sorted_i.failed_to_sort
# # #####################################################

job_id_max

# +
# #########################################################
data_dict_list = []
# #########################################################
group_cols = ["compenv", "slab_id", "ads", "active_site", "att_num", ]
grouped = df_jobs_i.groupby(group_cols)
# #########################################################
num_groups_processed = 0
for i_cnt, (name_i, group_i) in enumerate(grouped):
    # print(name_i)
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    name_dict_i = dict(zip(group_cols, name_i))
    # #####################################################


    # #########################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #########################################################

    # #########################################################
    row_anal_i = df_jobs_anal.loc[name_i]
    # #########################################################
    job_completely_done_i = row_anal_i.job_completely_done
    # #########################################################

    # print(job_completely_done_i)

    if job_completely_done_i:
        num_groups_processed += 1

        # print("TEMP | ")
        # if num_groups_processed > 1:
        #     break

        # #################################################
        # Findind mix/max job_ids
        rev_max = group_i.rev_num.max()
        rev_min = 1


        if rev_max != group_i.shape[0]:
            print("s9sdufjs9f09sui", name_i)
            continue

        row_max_i = group_i[group_i.rev_num == rev_max]
        assert row_max_i.shape[0] == 1, "ISDJFISj"
        row_max_i = row_max_i.iloc[0]
        job_id_max = row_max_i.name

        row_min_i = group_i[group_i.rev_num == rev_min]
        assert row_min_i.shape[0] == 1, "ISDJFISj"
        row_min_i = row_min_i.iloc[0]
        job_id_min = row_min_i.name

        # #################################################
        row_data_i = df_jobs_data.loc[job_id_max]
        # #################################################
        atoms_final = row_data_i.final_atoms
        # #################################################

        # #####################################################
        row_atoms_sorted_i = df_atoms_sorted_ind[
            df_atoms_sorted_ind.job_id == job_id_max]
        row_atoms_sorted_i = row_atoms_sorted_i.iloc[0]
        # #####################################################
        atoms_final_sorted = row_atoms_sorted_i.atoms_sorted_good
        failed_to_sort_i = row_atoms_sorted_i.failed_to_sort
        # #####################################################

        # #################################################
        row_init_min_i = df_init_slabs[df_init_slabs.job_id_min == job_id_min]
        row_init_min_i = row_init_min_i.iloc[0]
        # #################################################
        atoms_init = row_init_min_i.init_atoms
        # #################################################

        atoms_init_non_constr = remove_constrained_atoms(atoms_init)
        atoms_final_non_constr = remove_constrained_atoms(atoms_final)


        # #################################################
        out_ccf_data_dict = get_all_ccf_data(
            atoms_init=atoms_init,
            atoms_final=atoms_final,
            atoms_init_part=atoms_init_non_constr,
            atoms_final_part=atoms_final_non_constr,
            name_i=name_i,
            r_cut_off=r_cut_off,
            r_vector=r_vector,
            )
        # #################################################
        ccf_init = out_ccf_data_dict["ccf_init"]
        ccf_init_2 = out_ccf_data_dict["ccf_init_2"]
        ccf_final = out_ccf_data_dict["ccf_final"]
        ccf_final_2 = out_ccf_data_dict["ccf_final_2"]
        # #################################################

        d_i = cal_ccf_d(ccf_init, ccf_final)
        d_i_2 = cal_ccf_d(ccf_init_2, ccf_final_2)

        ave_dist_pa = None
        if not failed_to_sort_i:
            ave_dist_pa = get_ave_drift__wrapper(
                atoms_init=atoms_init,
                atoms_final=atoms_final_sorted,
                name_i=name_i,
                )


        # #################################################
        data_dict_i.update(name_dict_i)
        # #################################################
        data_dict_i["job_id_min"] = job_id_min
        data_dict_i["job_id_max"] = job_id_max
        data_dict_i["init_final_simil"] = d_i
        data_dict_i["init_final_simil_part"] = d_i_2
        data_dict_i["ave_dist_pa"] = ave_dist_pa
        data_dict_i["atoms_init_part"] = atoms_init_non_constr
        data_dict_i["atoms_final_part"] = atoms_final_non_constr
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

# #########################################################
df_struct_drift = pd.DataFrame(data_dict_list)

# df_struct_drift = df_struct_drift.sort_values(
#     "init_final_simil", ascending=False)
# df_struct_drift = df_struct_drift.sort_values("init_final_simil_part", ascending=False)
df_struct_drift = df_struct_drift.sort_values("ave_dist_pa", ascending=False)

df_struct_drift = df_struct_drift.set_index(
    ["compenv", "slab_id", "ads", "active_site", "att_num", ])
# #########################################################
# -

ave_dist_pa

# +
# row_atoms_sorted_i

# + active=""
#
#
#
#
#
#
#
#
#
# -

assert False

# # Writing pairs to file for viewing

# +
dir_path = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/slab_struct_drift",
    "out_data/most_dissimilar_pairs")

shutil.rmtree(dir_path)

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# +
df_struct_drift_i = df_struct_drift.iloc[0:15]

# df_struct_drift_i = df_struct_drift

files_to_open = []
for i_cnt, (name_i, row_i) in enumerate(df_struct_drift_i.iterrows()):
    # #####################################################
    job_id_min_i = row_i.job_id_min
    job_id_max_i = row_i.job_id_max
    # #####################################################
    atoms_init_part_i = row_i.atoms_init_part
    atoms_final_part_i = row_i.atoms_final_part
    # #####################################################

    row_paths_min_i = df_jobs_paths.loc[job_id_min_i]
    row_paths_max_i = df_jobs_paths.loc[job_id_max_i]

    gdrive_path_min_i = row_paths_min_i.gdrive_path
    gdrive_path_max_i = row_paths_max_i.gdrive_path

    init_path = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        gdrive_path_min_i,
        "init.traj")

    final_path = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        gdrive_path_max_i,
        "out.cif")

    name_tup = name_i
    name_list = []
    for i in name_tup:
        if type(i) == int or type(i) == float:
            name_list.append(str(int(i)))
        elif type(i) == str:
            name_list.append(i)
        else:
            name_list.append(str(i))

    dir_name = "__".join(name_list)
    dir_name = str(i_cnt).zfill(3) + "__" + dir_name

    dir_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/slab_struct_drift",
        "out_data/most_dissimilar_pairs",
        dir_name)


    # #####################################################
    # final_cif_path = os.path.join(final_path, "final.cif")
    my_file = Path(final_path)
    if my_file.is_file():

        my_file = Path(dir_path)
        if not my_file.is_dir():
            os.makedirs(dir_path)

        shutil.copyfile(
            final_path,
            os.path.join(dir_path, str(i_cnt).zfill(3) + "_final.cif"))

        atoms_init = io.read(init_path)
        atoms_init.write(
            os.path.join(dir_path, str(i_cnt).zfill(3) + "_init.cif")
            )

        atoms_init_part_i.write(
            os.path.join(dir_path, str(i_cnt).zfill(3) + "_part_init.cif")
            )
        atoms_final_part_i.write(
            os.path.join(dir_path, str(i_cnt).zfill(3) + "_part_final.cif")
            )


        files_to_open.append(
            os.path.join(dir_path, str(i_cnt).zfill(3) + "_part_init.cif"),
            )
        files_to_open.append(
            os.path.join(dir_path, str(i_cnt).zfill(3) + "_part_final.cif"),
            )
# -

print("VESTA \\")
for i in files_to_open:
    tmp = 42
    print(
        i,
        "\\",
        )

df_struct_drift_i.iloc[0:20]

assert False

# +
# final_path

# +
# Simple Plotly Plot
import plotly.graph_objs as go

# x_array = [0, 1, 2, 3]
y_array = df_struct_drift_i.init_final_simil_part

trace = go.Scatter(
    # x=x_array,
    y=y_array,
    )
data = [trace]

fig = go.Figure(data=data)
fig.show()
# -

assert False

# + active=""
#
#

# +
# # Pickling data ###########################################
# import os; import pickle
# directory = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/creating_slabs/slab_similarity",
#     "out_data")
# if not os.path.exists(directory): os.makedirs(directory)
# with open(os.path.join(directory, "df_slab_simil.pickle"), "wb") as fle:
#     pickle.dump(df_slab_simil, fle)
# # #########################################################

# +
# from methods import get_df_slab_simil
# df_slab_simil = get_df_slab_simil()
# -

assert False

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# import os
# import sys

# import pickle
# from pathlib import Path
# import itertools

# import numpy as np
# import pandas as pd

# from StructurePrototypeAnalysisPackage.ccf import struc2ccf
# from StructurePrototypeAnalysisPackage.ccf import cal_ccf_d

# + jupyter={"source_hidden": true}
# # slab_id = None
# slab = atoms_init
# name_tup = name_i
# init_or_final = "init"
# verbose = True
# r_cut_off = r_cut_off
# r_vector = r_vector


# def get_ccf(
#     slab=None,
#     name_tup=None,
#     init_or_final=None,
#     verbose=None,
#     r_cut_off=None,
#     r_vector=None,


#     # slab=None,
#     # verbose=True,
#     # r_cut_off=None,
#     # r_vector=None,
#     ):
# """
# """
# #| - get_ccf_i
# # #####################################################
# global os
# global pickle






# name_list = []
# for i in name_tup:
#     if type(i) == int or type(i) == float:
#         name_list.append(str(int(i)))
#     elif type(i) == str:
#         name_list.append(i)
#     else:
#         name_list.append(str(i))
# name_i = "__".join(name_list)
# name_i += "___" + init_or_final + ".pickle"







# directory = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "dft_workflow/job_analysis/slab_struct_drift",
#     "out_data/ccf_files")


# file_path_i = os.path.join(directory, name_i)
# my_file = Path(file_path_i)
# if my_file.is_file():
#     if verbose:
#         print("File exists already")

#     # #################################################
#     # import pickle; import os
#     # path_i = os.path.join(
#     #     os.environ["PROJ_irox_oer"],
#     #     "workflow/creating_slabs/slab_similarity",
#     #     file_path_i)
#     with open(file_path_i, "rb") as fle:
#         ccf_i = pickle.load(fle)
#     # #################################################

# else:
#     ccf_i = struc2ccf(slab, r_cut_off, r_vector)


#     # Pickling data ###################################
#     if not os.path.exists(directory): os.makedirs(directory)
#     with open(file_path_i, "wb") as fle:
#         pickle.dump(ccf_i, fle)
#     # #################################################

# # # return(ccf_i)
# #__|

# + jupyter={"source_hidden": true}
# # get_ccf?
# ccf_init

# + jupyter={"source_hidden": true}
# grouped = df_slab.groupby(["bulk_id"])
# for bulk_id_i, group_i in grouped:
#     for slab_id_j, row_j in group_i.iterrows():
#         # #####################################################
#         slab_final_j = row_j.slab_final
#         # #####################################################

#         ccf_j = get_ccf(
#             slab_id=slab_id_j,
#             slab_final=slab_final_j,
#             r_cut_off=r_cut_off,
#             r_vector=r_vector,
#             verbose=False)

# + jupyter={"source_hidden": true}
# verbose_local = False
# # #########################################################
# data_dict_list = []
# # #########################################################
# grouped = df_slab.groupby(["bulk_id"])
# for bulk_id_i, group_i in grouped:
#     # #####################################################
#     data_dict_i = dict()
#     # #####################################################

#     if verbose_local:
#         print("slab_id:", bulk_id_i)

#     D_ij = get_D_ij(group_i, slab_id=bulk_id_i)
#     ident_slab_pairs_i = get_identical_slabs(D_ij)

#     # print("ident_slab_pairs:", ident_slab_pairs_i)

#     ids_to_remove = []
#     for ident_pair_i in ident_slab_pairs_i:
#         # Checking if any id already added to `id_to_remove` is in a new pair
#         for i in ids_to_remove:
#             if i in ident_pair_i:
#                 print("This case needs to be dealt with more carefully")
#                 break

#         ident_pair_2 = np.sort(ident_pair_i)
#         ids_to_remove.append(ident_pair_2[0])

#     num_ids_to_remove = len(ids_to_remove)

#     if verbose_local:
#         print("ids_to_remove:", ids_to_remove)

#     # #####################################################
#     data_dict_i["bulk_id"] = bulk_id_i
#     data_dict_i["slab_ids_to_remove"] = ids_to_remove
#     data_dict_i["num_ids_to_remove"] = num_ids_to_remove
#     # #####################################################
#     data_dict_list.append(data_dict_i)
#     # #####################################################

# + jupyter={"source_hidden": true}
df_slab_simil = pd.DataFrame(data_dict_list)

df_slab_simil

# + jupyter={"source_hidden": true}
# tmp.index

# df_struct_drift.shape

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# def remove_constrained_atoms(atoms):
#     """
#     """
#     #| - remove_constrained_atoms

#     # atoms_new = copy.deepcopy(atoms_final)
#     atoms_new = copy.deepcopy(atoms)

#     indices_to_remove = atoms_new.constraints[0].index

#     mask = []
#     for atom in atoms_new:
#         if atom.index in indices_to_remove:
#             mask.append(True)
#         else:
#             mask.append(False)
#     del atoms_new[mask]

#     return(atoms_new)
#     #__|

# + jupyter={"source_hidden": true}
# atoms_init_non_constr = remove_constrained_atoms(atoms_init)
# atoms_final_non_constr = remove_constrained_atoms(atoms_final)

# + jupyter={"source_hidden": true}
# atoms_init_non_constr.write("tmp_init.cif")
# atoms_final_non_constr.write("tmp_final.cif")

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# # nersc	gubufafu_74	o	NaN	1

# df = df_jobs
# df = df[
#     (df["compenv"] == "nersc") &
#     (df["slab_id"] == "gubufafu_74") &
#     (df["ads"] == "o") &
#     (df["active_site"] == "NaN") &
#     (df["att_num"] == 1) &
#     [True for i in range(len(df))]
#     ]
# df_jobs_i = df

# + jupyter={"source_hidden": true}
# # This is the job that exploded

# # slac	relovalu_12	oh	24.0	2

# df = df_jobs
# df = df[
#     (df["compenv"] == "slac") &
#     (df["slab_id"] == "relovalu_12") &
#     (df["ads"] == "oh") &
#     (df["active_site"] == 24.) &
#     (df["att_num"] == 2) &
#     [True for i in range(len(df))]
#     ]
# df_jobs_i = df

# + jupyter={"source_hidden": true}
# # TEMP


# df = df_jobs
# df = df[
#     (df["compenv"] == "sherlock") &
#     (df["slab_id"] == "kesekodi_38") &
#     (df["ads"] == "o") &
#     (df["active_site"] == "NaN") &
#     (df["att_num"] == 1) &
#     [True for i in range(len(df))]
#     ]
# df_jobs_i = df

# + jupyter={"source_hidden": true}
# out_ccf_data_dict = get_all_ccf_data(
#     atoms_init=atoms_init,
#     atoms_final=atoms_final,
#     atoms_init_part=atoms_init_non_constr,
#     atoms_final_part=atoms_final_non_constr,
#     name_i=name_i,
#     r_cut_off=r_cut_off,
#     r_vector=r_vector,
#     )

# + jupyter={"source_hidden": true}
# # #####################################################
# row_atoms_sorted_i = df_atoms_sorted_ind[
#     df_atoms_sorted_ind.job_id == job_id_max]
# row_atoms_sorted_i = row_atoms_sorted_i.iloc[0]
# # #####################################################
# atoms_final_sorted = row_atoms_sorted_i.atoms_sorted_good
# # #####################################################

# + jupyter={"source_hidden": true}
# atoms_init
# atoms_final_sorted

# + jupyter={"source_hidden": true}
# atoms_final = atoms_final_sorted

# # np.all(atoms_init.cell == atoms_final.cell)

# lattice_cells_equal = np.allclose(
#     atoms_init.cell,
#     atoms_final.cell,
#     )

# atoms_init.cell

# atoms_final_sorted.cell

# df_struct_drift
