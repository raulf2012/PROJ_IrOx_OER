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

# # Import Modules

# + jupyter={"outputs_hidden": true}
import os
print(os.getcwd())
import sys

# from pathlib import Path
# import copy
import pickle

# import json

import numpy as np
import pandas as pd

from ase import io

# # from tqdm import tqdm
from tqdm.notebook import tqdm

# # #########################################################


# # #########################################################
# from misc_modules.pandas_methods import drop_columns
# from misc_modules.misc_methods import GetFriendlyID
# from ase_modules.ase_methods import view_in_vesta

# # #########################################################
from methods import (
    # get_df_dft,
    # symmetrize_atoms,
    # get_structure_coord_df,
    # remove_atoms,
    get_df_slab,
    )

from proj_data import metal_atom_symbol

# #########################################################
from local_methods import (
    mean_O_metal_coord,
    get_all_active_sites,
    get_unique_active_sites,
    process_rdf,
    compare_rdf_ij,
    return_modified_rdf,
    create_interp_df,
    )

# # from local_methods import calc_surface_area
# -

# # Read Data

# + jupyter={"outputs_hidden": true}
df_slab = get_df_slab()

df_slab = df_slab.set_index("slab_id")
# -

# # Create Directories

# + jupyter={"outputs_hidden": true}
directory = "out_data"
if not os.path.exists(directory):
    os.makedirs(directory)

# + jupyter={"outputs_hidden": true}
row_i = df_slab.loc["tagilahu_40"]

# + jupyter={"outputs_hidden": true}
# #####################################################
slab = row_i.slab_final
slab.write("out_data/temp.cif")

slab_id = row_i.name
bulk_id = row_i.bulk_id
facet = row_i.facet
num_atoms = row_i.num_atoms
# #####################################################

active_sites = get_all_active_sites(
    slab=slab,
    slab_id=slab_id,
    bulk_id=bulk_id,
    )

# + jupyter={"outputs_hidden": true}
# active_sites = [62, 63, 64, 66, 67, 68]
# active_sites = [63, ]


# + jupyter={"outputs_hidden": true}
# assert False

# + jupyter={"outputs_hidden": true}
directory = "out_plot/__temp__"
if not os.path.exists(directory):
    os.makedirs(directory)

# + jupyter={"outputs_hidden": true}
slab=slab
active_sites=active_sites
bulk_id=bulk_id
facet=facet
slab_id=slab_id
metal_atom_symbol=metal_atom_symbol

# + jupyter={"outputs_hidden": true}
# def get_unique_active_sites(
# slab=None,
# active_sites=None,
# bulk_id=None,
# facet=None,
# slab_id=None,
# metal_atom_symbol=None,
# ):
"""
"""
#| - get_unique_active_sites

df_coord_slab_i = get_df_coord(slab_id=slab_id, mode="slab")

df_coord_bulk_i = get_df_coord(bulk_id=bulk_id, mode="bulk")


# #########################################################
custom_name_pre = bulk_id + "__" + facet + "__" + slab_id

df_rdf_dict = dict()
for i in active_sites:

    print("active_site:", i)
    df_rdf_i = process_rdf(
        atoms=slab,
        active_site_i=i,
        df_coord_slab_i=df_coord_slab_i,
        metal_atom_symbol=metal_atom_symbol,
        custom_name=custom_name_pre,
        TEST_MODE=True,
        )
    # df_rdf_i = df_rdf_i.rename(columns={" g(r)": "g"})
    df_rdf_dict[i] = df_rdf_i


# # #########################################################
# diff_rdf_matrix = np.empty((len(active_sites), len(active_sites), ))
# diff_rdf_matrix[:] = np.nan
# for i_cnt, active_site_i in enumerate(active_sites):
#     df_rdf_i = df_rdf_dict[active_site_i]

#     for j_cnt, active_site_j in enumerate(active_sites):
#         df_rdf_j = df_rdf_dict[active_site_j]

#         diff_i = compare_rdf_ij(
#             df_rdf_i=df_rdf_i,
#             df_rdf_j=df_rdf_j,
#             )

#         diff_rdf_matrix[i_cnt, j_cnt] = diff_i

# # #########################################################
# df_rdf_ij = pd.DataFrame(diff_rdf_matrix, columns=active_sites)
# df_rdf_ij.index = active_sites


# # #########################################################
# import copy

# active_sites_cpy = copy.deepcopy(active_sites)


# diff_threshold = 0.3
# duplicate_active_sites = []
# for active_site_i in active_sites:

#     if active_site_i in duplicate_active_sites:
#         continue

#     for active_site_j in active_sites:
#         if active_site_i == active_site_j:
#             continue

#         diff_ij = df_rdf_ij.loc[active_site_i, active_site_j]
#         if diff_ij < diff_threshold:
#             try:
#                 active_sites_cpy.remove(active_site_j)
#                 duplicate_active_sites.append(active_site_j)
#             except:
#                 pass

# active_sites_unique = active_sites_cpy

# # #########################################################
# import plotly.express as px
# import plotly.graph_objects as go

# active_sites_str = [str(i) for i in active_sites]
# fig = go.Figure(data=go.Heatmap(
#     z=df_rdf_ij.to_numpy(),
#     x=active_sites_str,
#     y=active_sites_str,
#     # type="category",
#     ))

# fig["layout"]["xaxis"]["type"] = "category"
# fig["layout"]["yaxis"]["type"] = "category"

# # fig.show()

# directory = "out_plot/rdf_heat_maps"
# if not os.path.exists(directory):
#     os.makedirs(directory)

# from plotting.my_plotly import my_plotly_plot

# # file_name = "rdf_heat_maps/ " + custom_name_pre + "_rdf_diff_heat_map"
# file_name = "__temp__/ " + custom_name_pre + "_rdf_diff_heat_map"
# my_plotly_plot(
#     figure=fig,
#     # plot_name="rdf_heat_maps/rdf_diff_heat_map",
#     plot_name=file_name,

#     write_html=True,
#     write_png=False,
#     png_scale=6.0,
#     write_pdf=False,
#     write_svg=False,
#     try_orca_write=False,
#     )

# # return(active_sites_unique)
# #__|

# + jupyter={"outputs_hidden": true}
custom_name_pre

# + jupyter={"outputs_hidden": true}
# active_site_i = 62
active_site_i = 63
active_site_j = 66

df_rdf_i = df_rdf_dict[active_site_i]
df_rdf_j = df_rdf_dict[active_site_j]

# + jupyter={"outputs_hidden": true}
# Pickling data ###########################################
# out_dict = dict()
# out_dict["TEMP"] = None

import os; import pickle
path_i = os.path.join(
    os.environ["HOME"],
    "__temp__",
    "temp.pickle")
with open(path_i, "wb") as fle:
    pickle.dump((df_rdf_i, df_rdf_j), fle)
# #########################################################

# + jupyter={"outputs_hidden": true}
def test_rdf_opt(dx, df_rdf_i, df_rdf_j, chunks_to_edit):
    """
    """
    df_rdf_j_new = return_modified_rdf(
        df_rdf=df_rdf_j,
        # chunk_to_edit=0,
        chunks_to_edit=chunks_to_edit,
        # dx=-0.04,
        dx=dx,
        )

    df_rdf_i = df_rdf_i
    df_rdf_j = df_rdf_j_new

    # #########################################################
    r_combined = np.sort((df_rdf_j.r.tolist() + df_rdf_i.r.tolist()))
    r_combined = np.sort(list(set(r_combined)))

    df_interp_i = create_interp_df(df_rdf_i, r_combined)
    df_interp_j = create_interp_df(df_rdf_j, r_combined)

    diff_i = compare_rdf_ij(
        df_rdf_i=df_interp_i,
        df_rdf_j=df_interp_j)

    print("dx:", dx, " | ","diff_i:", diff_i)
    print("")

    return(diff_i)


# + jupyter={"outputs_hidden": true}
def constraint_bounds(dx, df_rdf_i, df_rdf_j, chunks_to_edit):
    # out = -0.2 + np.abs(dx)
    out = +0.05 - np.abs(dx)
    return(out)


# + jupyter={"outputs_hidden": true}
from scipy.optimize import minimize

data_dict_list = []
# for peak_i in range(0, 10):
for peak_i in range(0, 1):
    data_dict_i = dict()

    data_dict_i["peak"] = peak_i

    arguments = (df_rdf_i, df_rdf_j, peak_i)

    cons = ({
        "type": "ineq",
        "fun": constraint_bounds,
        "args": arguments,
        })

    initial_guess = 0

    result = minimize(
        # obj,
        test_rdf_opt,
        initial_guess,
        method="SLSQP",
        args=arguments,
        constraints=cons,
        )

    print(40 * "*")
    print(result)
    dx_i = result["x"][0]
    data_dict_i["dx"] = dx_i
    print(40 * "*")


    data_dict_list.append(data_dict_i)

# + jupyter={"outputs_hidden": true}
df = pd.DataFrame(data_dict_list)

df

# + jupyter={"outputs_hidden": true}
df_rdf_j_new = return_modified_rdf(
    df_rdf=df_rdf_j,

    # chunks_to_edit=0,
    chunks_to_edit=df.peak.tolist(),

    # dx=-0.04,
    # dx=-0.03332953,
    dx=df.dx.tolist(),
    )

import plotly.graph_objs as go
data = []

trace = go.Scatter(
    x=df_rdf_i.r,
    y=df_rdf_i.g,
    name="df_rdf_i",
    )
data.append(trace)

trace = go.Scatter(
    x=df_rdf_j_new.r,
    y=df_rdf_j_new.g,
    name="df_rdf_j_new",
    )
data.append(trace)

trace = go.Scatter(
    x=df_rdf_j.r,
    y=df_rdf_j.g,
    name="df_rdf_j",
    )
data.append(trace)

fig = go.Figure(data=data)
from plotting.my_plotly import my_plotly_plot
file_name = "__temp__/modified_and_opt_rdf_plots"
my_plotly_plot(
    figure=fig,
    plot_name=file_name,
    write_html=True)
fig.show()

# + jupyter={"outputs_hidden": true}

df_rdf_i = df_rdf_i
df_rdf_j = df_rdf_j_new

# #########################################################
r_combined = np.sort((df_rdf_j.r.tolist() + df_rdf_i.r.tolist()))
r_combined = np.sort(list(set(r_combined)))

df_interp_i = create_interp_df(df_rdf_i, r_combined)
df_interp_j = create_interp_df(df_rdf_j, r_combined)

diff_i = compare_rdf_ij(
    df_rdf_i=df_interp_i,
    df_rdf_j=df_interp_j)

diff_i

# + jupyter={"outputs_hidden": true}
print(diff_i)

# + jupyter={"outputs_hidden": true}
assert False

# + jupyter={"outputs_hidden": true}
tmp = {
    -0.04: 0.5574390754357897,
    -0.03: 0.5421194254988151,
    -0.02: 0.5866178898201364,
    -0.01: 0.6260045841988724,
    +0.00: 0.6653912750495825,
    +0.01: 0.6901818880303157,
    +0.05: 0.748109677289929,
    +0.10: 0.756948547426282,
    }

# + jupyter={"outputs_hidden": true}
# # Pickling data ###########################################
# # out_dict = dict()
# # out_dict["TEMP"] = None

# import os; import pickle
# path_i = os.path.join(
#     os.environ["HOME"],
#     "__temp__",
#     "temp_2.pickle")
# with open(path_i, "wb") as fle:
#     pickle.dump((df_rdf_i, df_rdf_j_new), fle)
# # #########################################################

# + jupyter={"outputs_hidden": true}
# from scipy.optimize import minimize

# arguments = (df_rdf_i, df_rdf_j, 0)

# def constraint_bounds():
#     out = -0.2 + np.abs(dx)
#     return(out)

# cons = ({
#     "type": "ineq",
#     "fun": constraint_bounds,
#     "args": arguments,
#     })

# initial_guess = 0

# result = minimize(
#     # obj,
#     test_rdf_opt,
#     initial_guess,
#     method="SLSQP",
#     args=arguments,
#     # constraints=cons,
#     )

# print(40 * "*")
# result
# print(40 * "*")

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
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

# + jupyter={"outputs_hidden": true}
assert False
# -

# # Import Modules

# + jupyter={"outputs_hidden": true}
import os
print(os.getcwd())
import sys

# from pathlib import Path
# import copy
import pickle

# import json

import numpy as np
# import pandas as pd

from ase import io

# # from tqdm import tqdm
# from tqdm.notebook import tqdm

# # #########################################################


# # #########################################################
# from misc_modules.pandas_methods import drop_columns
# from misc_modules.misc_methods import GetFriendlyID
# from ase_modules.ase_methods import view_in_vesta

# # #########################################################
from methods import (
    # get_df_dft,
    # symmetrize_atoms,
    # get_structure_coord_df,
    # remove_atoms,
    get_df_slab,
    )

from proj_data import metal_atom_symbol

# #########################################################
from local_methods import (
    mean_O_metal_coord,
    # analyse_local_coord_env, check_if_sys_processed,
    # remove_nonsaturated_surface_metal_atoms,
    # remove_noncoord_oxygens,
    # create_slab_from_bulk,
    # get_slab_thickness,
    # remove_highest_metal_atoms,
    # remove_all_atoms_above_cutoff,
    # create_final_slab_master,
    # constrain_slab,
    )

# # from local_methods import calc_surface_area

# + jupyter={"outputs_hidden": true}
assert False
# -

# # Read Data

# + jupyter={"outputs_hidden": true}
df_slab = get_df_slab()

df_slab = df_slab.set_index("slab_id")
# -

# # Create Directories

# + jupyter={"outputs_hidden": true}
directory = "out_data"
if not os.path.exists(directory):
    os.makedirs(directory)

# + active=""
#
#
#
#

# + jupyter={"outputs_hidden": true}
for i_cnt, row_i in df_slab.iterrows():
    tmp = 42

row_i

# + jupyter={"outputs_hidden": true}
assert False
# -

# # Testing on single row/structure

# + jupyter={"outputs_hidden": true}

# #########################################################
# row_i = df_slab.loc["honorupo_58"]
# row_i = df_slab.sample(n=1).iloc[0]

slab = row_i.slab_final
slab.write("out_data/temp.cif")

slab_id = row_i.name
bulk_id = row_i.bulk_id
facet = row_i.facet
num_atoms = row_i.num_atoms
# #########################################################

# #########################################################
from methods import get_df_coord

df_coord_slab_i = get_df_coord(slab_id=slab_id, mode="slab")
df_coord_bulk_i = get_df_coord(bulk_id=bulk_id, mode="bulk")

# #########################################################
def method(row_i, metal_elem=None):
    neighbor_count = row_i.neighbor_count
    elem_num = neighbor_count.get(metal_elem, None)
    return(elem_num)

df_i = df_coord_bulk_i
df_i["num_metal"] = df_i.apply(
    method, axis=1,
    metal_elem="Ir")

df_i = df_coord_slab_i
df_i["num_metal"] = df_i.apply(
    method, axis=1,
    metal_elem="Ir")

# #########################################################
# mean_O_metal_coord = mean_O_metal_coord(df_coord=df_coord_bulk_i)

dz = 4
positions = slab.positions

z_min = np.min(positions[:,2])
z_max = np.max(positions[:,2])

# #########################################################
active_sites = []
for atom in slab:
    if atom.symbol == "O":
        if atom.position[2] > z_max - dz:
            df_row_i = df_coord_slab_i[
                df_coord_slab_i.structure_index == atom.index]
            df_row_i = df_row_i.iloc[0]
            num_metal = df_row_i.num_metal

            if num_metal == 1:
                active_sites.append(atom.index)


# active_sites
