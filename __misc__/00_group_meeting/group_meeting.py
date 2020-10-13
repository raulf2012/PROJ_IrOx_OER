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

# # Getting data collected for group meeting presentation
# ---

# +
import os
print(os.getcwd())
import sys

import copy

import plotly.graph_objs as go

# #########################################################
from methods import get_df_slab
from methods import get_df_active_sites
# -

# # Script Inputs

bulk_id_i = "8l919k6s7p"

# # Read Data

# +
# #########################################################
df_active_sites = get_df_active_sites()

# #########################################################
df_slab = get_df_slab()
df_slab_i = df_slab[df_slab.bulk_id == bulk_id_i]
# -

for slab_id_i, row_i in df_slab_i.iterrows():

    # #########################################################
    slab_final_i =  row_i.slab_final
    slab_id_i = row_i.slab_id
    facet_i = row_i.facet
    # #########################################################

    file_name_i = slab_id_i + "_" + facet_i + ".cif"

    slab_final_i.write(os.path.join("__temp__/slabs", file_name_i))

# # Writing active sites to cif

df_active_sites_i = df_active_sites[df_active_sites.bulk_id == bulk_id_i]

# +
slab_id_i = "lulidoka_21"

df_active_sites_j = df_active_sites_i.loc[[slab_id_i]]
for slab_id_i, row_i in df_active_sites_j.iterrows():

    # #########################################################
    active_sites_i = row_i.active_sites
    active_sites_unique_i = row_i.active_sites_unique
    # #########################################################

    # #########################################################
    row_slab_i = df_slab.loc[slab_id_i]
    # #########################################################
    slab_final_i = row_slab_i.slab_final
    # #########################################################


    slab_final_i_orig = copy.deepcopy(slab_final_i)
    slab_final_0 = copy.deepcopy(slab_final_i)
    slab_final_1 = copy.deepcopy(slab_final_i)


    # #########################################################
    for as_i in active_sites_i:
        active_atom = slab_final_0[as_i]
        active_atom.set("symbol", "N")
    slab_final_0.write("__temp__/active_sites/all_active_sites.cif")
    # #########################################################

    # #########################################################
    for as_unique_i in active_sites_unique_i:
        active_atom = slab_final_1[as_unique_i]
        active_atom.set("symbol", "N")
    slab_final_1.write("__temp__/active_sites/unique_active_sites.cif")
    # #########################################################

# +
slab_id_i = "lulidoka_21"

path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/enumerate_adsorption",
    "out_data/df_rdf_dict")

files_list = os.listdir(path_i)


candidate_files = []
for file_i in files_list:
    if slab_id_i in file_i:
        # print(file_i)
        candidate_files.append(file_i)

mess_i = "Must only have one (or zero) file that matches slab_id given"
assert len(candidate_files) <= 1, mess_i

file_i = candidate_files[0]

# #########################################################
import pickle; import os
path_i = os.path.join(path_i, file_i)
with open(path_i, "rb") as fle:
    df_rdf_dict_i = pickle.load(fle)
# #########################################################
# -

data = []
for active_site_i, df_rdf_i in df_rdf_dict_i.items():
    df_rdf = df_rdf_i

    x_array = df_rdf["r"]
    y_array = df_rdf["g"]

    trace = go.Scatter(
        x=x_array,
        y=y_array,
        name=str(active_site_i),
        )
    data.append(trace)

# +
fig = go.Figure(data=data)
# fig.show()

from plotting.my_plotly import my_plotly_plot

plot_dir = "."
out_plot_file = "./temp_plot"
my_plotly_plot(
    figure=fig,
    # plot_name=str(active_site_i).zfill(4) + "_rdf",
    plot_name=out_plot_file,
    write_html=True,
    write_png=False,
    png_scale=6.0,
    write_pdf=False,
    write_svg=False,
    try_orca_write=False,
    )

# +
# i
# j

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# from ase import Atom

# # Atom?
