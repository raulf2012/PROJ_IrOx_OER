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

# # Further process slabs created in `create_slabs.ipynb`
# ---
#
#

# ### Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

from pathlib import Path
# from pathlib import Path
import time
from tqdm.notebook import tqdm

import json
import pickle

import pandas as pd
import numpy as np

import plotly.graph_objs as go

# #########################################################
from plotting.my_plotly import my_plotly_plot
from misc_modules.pandas_methods import reorder_df_columns

# #########################################################
from methods import get_structure_coord_df
from methods import (
    get_df_dft,
    get_slab_thickness,
    get_df_slab,
    )

from local_methods import (
    constrain_slab,
    resize_z_slab,
    calc_surface_area,
    repeat_xy,
    )


# -

# ### Read `df_slab` and `df_dft` dataframes

# +
df_slab = get_df_slab(mode="almost-final")

df_dft = get_df_dft()
# -

if "status" in df_slab.columns:
    df_slab = df_slab[df_slab.status != "Took too long"]
else:
    print("eh")
    df_slab = df_slab

# ### Create directories

# +
dir_root = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs")

directory = os.path.join(
    dir_root,
    "out_data/final_slabs_1")
if not os.path.exists(directory):
    os.makedirs(directory)

directory = os.path.join(
    dir_root,
    "out_data/final_slabs_2")
if not os.path.exists(directory):
    os.makedirs(directory)

directory = os.path.join(
    dir_root,
    "out_data/bulk_structures_temp")
if not os.path.exists(directory):
    os.makedirs(directory)
# -

# ### Main processing slabs

# +
# %%capture

# #########################################################
data_dict_list = []
# #########################################################
iterator = tqdm(df_slab.index.tolist(), desc="1st loop")
for i_cnt, slab_id_i in enumerate(iterator):
    row_i = df_slab.loc[slab_id_i]

    # if i_cnt > 100:
    #     break

    data_dict_i = dict()
    t0 = time.time()

    # #####################################################
    slab_id = row_i.name
    slab = row_i.slab_final
    bulk_id_i = row_i.bulk_id
    facet_i = row_i.facet
    facet_rank_i = row_i.facet_rank
    iter_time_i = row_i.iter_time_i
    phase_i = row_i.phase
    source_i = row_i.source
    # #####################################################

    slab_old = slab

    # slab_constrained = constrain_slab(atoms=slab)
    slab_final = resize_z_slab(atoms=slab, vacuum=15)
    slab_final.center()
    slab_final.wrap()


    # Repeat slab if needed
    min_len = 4
    out_dict = repeat_xy(
        atoms=slab_final,
        min_len_x=min_len,
        min_len_y=min_len)

    atoms_repeated = out_dict["atoms_repeated"]
    is_repeated = out_dict["is_repeated"]
    repeat_list = out_dict["repeat_list"]


    num_atoms_i = atoms_repeated.get_global_number_of_atoms()


    surf_a_i = calc_surface_area(atoms=atoms_repeated)

    slab_final = atoms_repeated  # <-------------------------------------------

    cell_mag_x = np.linalg.norm(slab_final.cell.array[0])
    cell_mag_y = np.linalg.norm(slab_final.cell.array[1])

    # #####################################################
    data_dict_i["slab_id"] = slab_id
    data_dict_i["bulk_id"] = bulk_id_i
    data_dict_i["facet"] = facet_i
    data_dict_i["facet_rank"] = facet_rank_i
    # -----------------------------------------------------
    data_dict_i["slab_final_old"] = slab_old
    data_dict_i["slab_final"] = slab_final
    data_dict_i["phase"] = phase_i
    data_dict_i["source"] = source_i
    # -----------------------------------------------------
    data_dict_i["num_atoms"] = num_atoms_i
    data_dict_i["num_atoms"] = num_atoms_i
    data_dict_i["surf_area"] = surf_a_i
    data_dict_i["cell_mag_x"] = cell_mag_x
    data_dict_i["cell_mag_y"] = cell_mag_y
    # -----------------------------------------------------
    data_dict_i["is_repeated"] = is_repeated
    data_dict_i["repeat_list"] = repeat_list
    # -----------------------------------------------------
    data_dict_i["loop_time"] = time.time() - t0
    data_dict_i["iter_time_i"] = iter_time_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

    file_name = row_i.bulk_id + "__" + row_i.name + "__" + row_i.facet + ".cif"
    slab_final.write("out_data/final_slabs_1/" + file_name)

    file_name = row_i.bulk_id + "__" + row_i.name + "__" + row_i.facet + ".traj"
    slab_final.write("out_data/final_slabs_1/" + file_name)

# #########################################################
df_slab_2 = pd.DataFrame(data_dict_list)
df_slab_2 = df_slab_2.set_index("slab_id", drop=False)

# df_slab_2.head()
# -

# ### Further analysis of slabs
#
#   * Get slab thickness
#   * Asign whether slab is unique or not based on slab similarity analysis

# +
from methods import get_df_slab_simil
df_slab_simil = get_df_slab_simil()

df_slab_simil_i = df_slab_simil[df_slab_simil.num_ids_to_remove > 0]

# #########################################################
ids_to_remove = df_slab_simil_i.slab_ids_to_remove.tolist()

ids_to_remove_flat = np.array([np.array(i) for i in ids_to_remove])
ids_to_remove_flat = np.hstack(ids_to_remove_flat)
ids_to_remove_flat = list(ids_to_remove_flat)

# +
# #########################################################
data_dict_list = []
# #########################################################
for slab_id_i, row_i in df_slab_2.iterrows():
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    slab_final = row_i.slab_final
    # #####################################################


    # Check whether slab is unique
    unique_slab_i = True
    if slab_id_i in ids_to_remove_flat:
        unique_slab_i = False

    # Computing slab thickness
    slab_thick_i = get_slab_thickness(atoms=slab_final)

    # from methods import get_df_coord
    # df_coord_i = get_df_coord(
    #     slab_id=slab_id,
    #     slab=slab_final,
    #     mode="slab")

    # from methods import remove_protruding_bottom_Os
    # # atoms = slab_final_i
    # dz = 0.75
    # angle_thresh = 30
    # atoms_out = remove_protruding_bottom_Os(
    #     atoms=slab_final,
    #     dz=dz,
    #     angle_thresh=angle_thresh,
    #     df_coord=df_coord_i,
    #     )

    # #####################################################
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["slab_thick"] = slab_thick_i
    data_dict_i["unique_slab"] = unique_slab_i
    # data_dict_i["slab_final"] = atoms_out
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


# #########################################################
df_slab_info = pd.DataFrame(data_dict_list)
df_slab_info = df_slab_info.set_index("slab_id")

df_slab_3 = pd.concat([
    df_slab_2,
    df_slab_info,
    ], axis=1)
# -

# ### Cleaning up dataframe

# +
cols_order = [
    "slab_id",
    "bulk_id",
    "facet",
    "slab_thick",
    "num_atoms",
    "slab_final",
    "loop_time",
    "iter_time_i",
    ]
df_slab_3 = reorder_df_columns(cols_order, df_slab_3)

df_slab_final = df_slab_3
df_slab_final = df_slab_final.set_index("slab_id", drop=False)
# -

# ### Save data

# Pickling data ###########################################
import os; import pickle
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_slab_final.pickle"), "wb") as fle:
    pickle.dump(df_slab_final, fle)
# #########################################################

# +
# (419, 18)
# (457, 18)
# (483, 18)

df_slab_final.shape

# +
from methods import update_df_slab_ids

update_df_slab_ids()
# -

# # Getting `df_coord` for final slab
#
# Needed because of cell xy repitiion

verbose_local = False
for i_cnt, row_i in df_slab_final.iterrows():

    # #####################################################
    slab_final = row_i.slab_final
    slab_id = row_i.slab_id
    # #####################################################

    if verbose_local:
        print(40 * "*")
        print("slab_id:", slab_id)

    file_name_i = slab_id + "_after_rep" + ".pickle"
    file_path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/creating_slabs/out_data/df_coord_files",
        file_name_i)

    my_file = Path(file_path_i)
    if not my_file.is_file():
        df_coord_slab_final = get_structure_coord_df(slab_final)
        with open(file_path_i, "wb") as fle:
            pickle.dump(df_coord_slab_final, fle)
    else:
        if verbose_local:
            print("Already computed")

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
# -

# # Checking that no slabs are less than 15 A in thickness

# +
df_slabs_too_thin = df_slab_3[df_slab_3.slab_thick < 15]

print("Number of slabs that are too thin:", "\n", df_slabs_too_thin.shape[0])

# df_slabs_too_thin[["slab_id", "bulk_id", "facet"]].to_csv("temp_slabs_too_thin.csv", index=False)
# -

# # Plotting Processing Speed vs Structure Size

# +
y_array = df_slab_2.iter_time_i / 60
x_array = df_slab_2.num_atoms

trace = go.Scatter(
    x=x_array,
    y=y_array,
    mode="markers",
    )

data = [trace]

fig = go.Figure(data=data)

fig.update_layout(
    title="Processing speed vs structure size (num atoms)",
    xaxis=dict(title=dict(text="Number of atoms")),
    yaxis=dict(
        title=dict(
            text="Processing time (min)",
            ),
        range=[-1, 12],
        ),
    )

fig.show()
# -

my_plotly_plot(
    figure=fig,
    plot_name="iter_speed_vs_num_atoms",
    write_html=True,
    write_png=False,
    png_scale=6.0,
    write_pdf=False,
    write_svg=False,
    try_orca_write=False,
    )

# # Writing the structures that are unique octahedras

# +
# #######################################################################
data_path = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs/selecting_bulks",
    "out_data/data.json")
with open(data_path, "r") as fle:
    data = json.load(fle)
# #######################################################################

bulk_ids__octa_unique = data["bulk_ids__octa_unique"]

# +
df = df_slab_2
df_i = df[df.bulk_id.isin(bulk_ids__octa_unique)]

file_names_list = []
for i_cnt, row_i in df_i.iterrows():
    slab = row_i.slab_final

    file_name_base = row_i.bulk_id + "__" + row_i.slab_id + "__" + row_i.facet
    file_names_list.append(file_name_base)

    file_name = file_name_base + ".cif"
    slab.write("out_data/final_slabs_2/" + file_name)

    file_name = file_name_base + ".traj"
    slab.write("out_data/final_slabs_2/" + file_name)
# -

# if True:
if False:
    file_names_list_i = [i + ".traj" for i in file_names_list]
    print("ase gui", *file_names_list_i)

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# # slab_id

# # df_coord_i =
# slab_id = "tetuwido_70"

# get_df_coord(
#     slab_id=slab_id,
#     slab=slab_final,
#     mode="slab", verbose=True)

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# print("TEMP")
# assert False
