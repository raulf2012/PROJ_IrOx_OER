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

import numpy as np

from local_methods import get_octa_vol, get_octa_geom

# +
# #########################################################
import pickle; import os
path_i = os.path.join(
    # os.environ[""],
    "__temp__",
    "data.pickle")
with open(path_i, "rb") as fle:
    out_data = pickle.load(fle)
# #########################################################

# out_data.keys()

df_coord_i = out_data["df_coord_i"]
active_site_j = out_data["active_site_j"]

atoms = out_data["atoms"]

# +
print("active_site_j:", active_site_j)

atoms.write("__temp__/tmp.cif")

# +
# df_coord_i=None
# active_site_j=None
# atoms=None
# verbose=False

# +
# def get_octa_geom(
#     df_coord_i=None,
#     active_site_j=None,
#     atoms=None,
#     verbose=False,
#     ):
"""
"""
#| - get_octa_geom
out_dict = dict(
    active_o_metal_dist=None,
    ir_o_mean=None,
    ir_o_std=None,
    )

row_coord_i = df_coord_i[df_coord_i.structure_index == active_site_j]
row_coord_i = row_coord_i.iloc[0]

nn_info_i = row_coord_i.nn_info

ir_nn = nn_info_i[0]
ir_coord = ir_nn["site"].coords

active_o_has_1_neigh = True
if len(nn_info_i) != 1:
    print("Need to return NaN")
    active_o_has_1_neigh = False
else:
    #| - Calculating the distance between the active O atom and Ir
    atom_active_o = atoms[active_site_j]

    diff_list = atom_active_o.position - ir_coord
    dist_i = (np.sum([i ** 2 for i in diff_list])) ** (1 / 2)

    # ir_nn["site"].coords

    out_dict["active_o_metal_dist"] = dist_i
    #__|

if active_o_has_1_neigh:
    ir_site_index = ir_nn["site_index"]

    row_coord_j = df_coord_i[df_coord_i.structure_index == ir_site_index]
    row_coord_j = row_coord_j.iloc[0]

    nn_info_ir = row_coord_j.nn_info

    if len(nn_info_ir) != 6:
        print("Pass NaN")
    else:
        ir_o_distances = []
        for nn_j in nn_info_ir:
            diff_list = nn_j["site"].coords - ir_coord
            dist_i = (np.sum([i ** 2 for i in diff_list])) ** (1 / 2)
            ir_o_distances.append(dist_i)

        ir_o_mean = np.mean(ir_o_distances)
        ir_o_std = np.std(ir_o_distances)
        out_dict["ir_o_mean"] = ir_o_mean
        out_dict["ir_o_std"] = ir_o_std

# return(out_dict)
#__|
# -

out_dict

# +
ir_site_index = ir_nn["site_index"]

row_coord_j = df_coord_i[df_coord_i.structure_index == ir_site_index]
row_coord_j = row_coord_j.iloc[0]

nn_info_ir = row_coord_j.nn_info

if len(nn_info_ir) != 6:
    print("Pass NaN")
else:
    ir_o_distances = []
    for nn_j in nn_info_ir:
        diff_list = nn_j["site"].coords - ir_coord
        dist_i = (np.sum([i ** 2 for i in diff_list])) ** (1 / 2)
        ir_o_distances.append(dist_i)

    ir_o_mean = np.mean(ir_o_distances)
    ir_o_std = np.std(ir_o_distances)
    out_dict["ir_o_mean"] = ir_o_mean
    out_dict["ir_o_std"] = ir_o_std
# -



# +
# octa_geom_dict = get_octa_geom(
#     df_coord_i=df_coord_i,
#     active_site_j=active_site_j,
#     atoms=atoms,
#     verbose=False,
#     )
# octa_geom_dict
# -

assert False

# + jupyter={"source_hidden": true}
# row_coord_i = df_coord_i[df_coord_i.structure_index == active_site_j]
# row_coord_i = row_coord_i.iloc[0]

# nn_info_i = row_coord_i.nn_info

# if len(nn_info_i) != 1:
#     print("Need to return NaN")

# # nn_info_i[0]["site"].coords

# # dir(nn_info_i[0]["site"])
# # row_coord_i
# # df_coord_i
# # print(atom_active_o.position)
# # print(nn_info_i[0]["site"].coords)

# atom_active_o = atoms[active_site_j]

# # dir(atom_active_o)

# diff_list = atom_active_o.position - nn_info_i[0]["site"].coords
# dist_i = (np.sum([i ** 2 for i in diff_list])) ** (1 / 2)

# dist_i

# + active=""
#
#
#
#
#
#
#

# +
# vol_i = get_octa_vol(
#     df_coord_i=df_coord_i,
#     active_site_j=active_site_j,
#     )
# vol_i
# -

# df_coord_i=None
# active_site_j=None
verbose=False

# +
# def get_octa_vol(
#     df_coord_i=None,
#     active_site_j=None,
#     verbose=False,
#     ):
"""
"""
#| - get_octa_vol
# #########################################################
row_coord_i = df_coord_i[df_coord_i.structure_index == active_site_j]
row_coord_i = row_coord_i.iloc[0]
# #########################################################
nn_info_i = row_coord_i.nn_info
# #########################################################

if len(nn_info_i) != 1:
    volume=None
    # return(None)
    
# mess_i = "Must  only have one row  here"
# assert len(nn_info_i) == 1, mess_i

nn_info_i = nn_info_i[0]

metal_index_i = nn_info_i["site_index"]


# #########################################################
# #########################################################
# #########################################################


row_coord_i = df_coord_i[df_coord_i.structure_index == metal_index_i]
row_coord_i = row_coord_i.iloc[0]

nn_info = row_coord_i.nn_info

#
if len(nn_info) != 6:
    if verbose:
        print("This active site Ir doesn't have 6 nearest neighbors")
    volume=None
    # return(volume)

coord_list = []
for nn_j in nn_info:
    site_j = nn_j["site"]
    coords_j = site_j.coords
    coord_list.append(coords_j)

# coord_list

import numpy as np
from scipy.spatial import ConvexHull

# points = np.array([[....]])
volume = ConvexHull(coord_list).volume

# return(volume)
#__|

# +
nn_info

volume
# -

volume

nn_info_i

coord_list

nn_info

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# # def get_octa_vol(
# #     df_coord_i=None,
# #     active_site_j=None,
# #     verbose=False,
# #     ):
# """
# """
# #| - get_octa_vol
# # #########################################################
# row_coord_i = df_coord_i[df_coord_i.structure_index == active_site_j]
# row_coord_i = row_coord_i.iloc[0]
# # #########################################################
# nn_info_i = row_coord_i.nn_info
# # #########################################################

# mess_i = "Must  only have one row  here"
# assert len(nn_info_i) == 1, mess_i

# nn_info_i = nn_info_i[0]

# metal_index_i = nn_info_i["site_index"]


# # #########################################################
# # #########################################################
# # #########################################################


# row_coord_i = df_coord_i[df_coord_i.structure_index == metal_index_i]
# row_coord_i = row_coord_i.iloc[0]

# nn_info = row_coord_i.nn_info

# # 
# if nn_info != 6:
#     if verbose:
#         print("This active site Ir doesn't have 6 nearest neighbors")
#     return(None)

# coord_list = []
# for nn_j in nn_info:
#     site_j = nn_j["site"]
#     coords_j = site_j.coords
#     coord_list.append(coords_j)

# # coord_list

# import numpy as np
# from scipy.spatial import ConvexHull

# # points = np.array([[....]])
# volume = ConvexHull(coord_list).volume

# # return(volume)
# #__|
