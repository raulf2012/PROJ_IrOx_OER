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

# # Compute similarity of constructed *O IrOx slabs
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import pickle
# from pathlib import Path

import numpy as np

# #########################################################
# from StructurePrototypeAnalysisPackage.ccf import struc2ccf
# from StructurePrototypeAnalysisPackage.ccf import struc2ccf, cal_ccf_d
from StructurePrototypeAnalysisPackage.ccf import cal_ccf_d

# #########################################################
from methods import get_df_slab

# #########################################################
from local_methods import get_ccf
# -

# # Script Inputs

# +
verbose = True

r_cut_off = 10
r_vector = np.arange(1, 10, 0.02)
# -

# # Read Data

df_slab = get_df_slab()

# # TEMP | Filtering down `df_slab`

df_slab = df_slab[df_slab.bulk_id ==  "mjctxrx3zf"]

# + active=""
#
#
# -

# # Looping through slabs and computing CCF

# +
grouped = df_slab.groupby(["bulk_id"])
for name_i, group_i in grouped:
    tmp = 42
    # print(name_i, ": ", group_i.shape[0], sep="")

# #########################################################
# name_i = "mjctxrx3zf"
# group_i = grouped.get_group(name_i)


for slab_id_j, row_j in group_i.iterrows():

    # #####################################################
    # row_j = group_i.iloc[0]
    # #####################################################
    slab_final_j = row_j.slab_final
    # #####################################################

    ccf_j = get_ccf(
        slab_id=slab_id_j,
        slab_final=slab_final_j,
        r_cut_off=r_cut_off,
        r_vector=r_vector,
        verbose=False)
# -

# # Constructing D_ij matrix

grouped = df_slab.groupby(["bulk_id"])
for name_i, group_i in grouped:
    tmp = 42

for slab_id_j, row_j in group_i.iterrows():
    print(40 * "*")
    for slab_id_k, row_k in group_i.iterrows():
        print(30 * "-")
        ccf_j = get_ccf(slab_id=slab_id_j, verbose=False)
        ccf_k = get_ccf(slab_id=slab_id_k, verbose=False)

        tmp = cal_ccf_d(ccf_j, ccf_k)
        print(tmp)



# +
# from StructurePrototypeAnalysisPackage.ccf import (
#     struc2ccf,
#     cal_ccf_d,
#     cal_inter_atomic_d,
#     d2ccf,
#     weight_f,
#     pearson_cc,
#     gaussian_f,
#     element_tag,
#     cell_range,
#     count_atoms_dict,
#     )

# from StructurePrototypeAnalysisPackage.ccf import struc2ccf, cal_ccf_d
# -



# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# def get_ccf_i(slab_id=None, ):
#     """
#     """
#     slab_id_i = slab_id

#     directory = "out_data/ccf_files"
#     name_i = slab_id_i + ".pickle"
#     # print("os:", os)
#     file_path_i = os.path.join(directory, name_i)

#     my_file = Path(file_path_i)
#     if my_file.is_file():
#         if verbose:
#             print("File exists already")

#         # #################################################
#         import pickle; import os
#         path_i = os.path.join(
#             os.environ["PROJ_irox_oer"],
#             "workflow/creating_slabs/slab_similarity",
#             file_path_i)
#         with open(path_i, "rb") as fle:
#             ccf_i = pickle.load(fle)
#         # #################################################
#     else:
#         ccf_j = struc2ccf(slab_final_j, r_cut_off, r_vector)


#         # Pickling data ###################################
#         if not os.path.exists(directory): os.makedirs(directory)
#         with open(file_path_i, "wb") as fle:
#             pickle.dump(ccf_j, fle)
#         # #################################################
