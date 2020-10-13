# -*- coding: utf-8 -*-
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

# # OER Analysis notebook

# # Import Modules

# + jupyter={}
import os
print(os.getcwd())
import sys

sys.path.insert(
    0, os.path.join(
        os.environ["PROJ_irox"],
        "data"))

# #############################################################################
# Python Modules
import pickle

import numpy as np
import pandas as pd

import plotly.graph_objs as go

# #############################################################################
# My Modules
from oxr_reaction.oxr_rxn import ORR_Free_E_Plot
from oxr_reaction.oxr_plotting_classes.oxr_plot_volcano import Volcano_Plot

# #############################################################################
# Project Data
from proj_data_irox import (
    smart_format_dict,
    gas_molec_dict,
    scaling_dict_ideal,
    )

from methods import (
    get_df_ads,
    get_df_jobs_paths,
    get_df_jobs,
    get_df_jobs_anal,
    )

# #############################################################################
# Local Imports
from plotting.my_plotly import my_plotly_plot
# -

# # Script Inputs

save_plot = False
plot_exp_traces = True

# # Read Data

# + jupyter={}
# #########################################################
df_ads = get_df_ads()

df_ads = df_ads[~df_ads.g_oh.isna()]
df_m = df_ads

# #########################################################
df_jobs_paths = get_df_jobs_paths()

# #########################################################
df_jobs = get_df_jobs()

# #########################################################
df_jobs_anal = get_df_jobs_anal()

# #########################################################
from methods import get_df_dft
df_dft = get_df_dft()

# #########################################################
from methods import get_df_job_ids
df_job_ids = get_df_job_ids()
# -

# # Create ΔG_*OOH column from *OH energy

# +
# df_m.g_ooh = 1.16 * df_m.g_oh + 2.8
df_m["g_ooh"] = df_m.g_oh + 2.8

# df_m

# +
smart_format_dict = [
     
    # [{"stoich": "AB2"}, {"color2": "#7FC97F"}],
    # [{"stoich": "AB3"}, {"color2": "#BEAED4"}],

    [{"stoich": "AB2"}, {"color2": "black"}],
    [{"stoich": "AB3"}, {"color2": "grey"}],

    ]

ORR_PLT = ORR_Free_E_Plot(
    free_energy_df=None,
    state_title="ads",
    free_e_title="ads_g",
    # ads_g	ads
    smart_format=smart_format_dict,
    color_list=None,
    rxn_type="OER")


df_m = df_m.set_index(["compenv", "slab_id", ], drop=False)


paths_dict = dict()
for name_i, row_i in df_m.iterrows():

    # #####################################################
    g_o_i = row_i.g_o
    g_oh_i = row_i.g_oh
    g_ooh_i = row_i.g_ooh
    slab_id_i = row_i.slab_id
    active_site_i = row_i.active_site
    job_id_o_i = row_i.job_id_o
    job_id_oh_i = row_i.job_id_oh
    # #####################################################


    # #####################################################
    df_job_ids_i = df_job_ids[df_job_ids.slab_id == slab_id_i]

    bulk_ids = df_job_ids_i.bulk_id.unique()

    mess_i = "SIJFIDSIFJIDSJIf"
    assert len(bulk_ids) == 1, mess_i

    bulk_id_i = bulk_ids[0]

    # #########################################################
    row_dft_i = df_dft.loc[bulk_id_i]
    # #########################################################
    stoich_i = row_dft_i.stoich
    # #########################################################


    data_dict_list =  [
        {"ads_g": g_o_i, "ads": "o", },
        {"ads_g": g_oh_i, "ads": "oh", },
        {"ads_g": g_ooh_i, "ads": "ooh", },
        {"ads_g": 0., "ads": "bulk", },
        ]
    df_i = pd.DataFrame(data_dict_list)

    df_i["stoich"] = stoich_i


    prop_name_list = [
        "stoich",
        ]

    # #########################################################
    # name_i = "IDSJFISDf"
    name_i = slab_id_i + "__" + str(int(active_site_i))
    ORR_PLT.add_series(
        df_i,
        plot_mode="all",
        overpotential_type="OER",
        property_key_list=prop_name_list,
        add_overpot=False,
        name_i=name_i,
        )

# +
print(list(paths_dict.keys()))
print("")

# tmp = [print(i) for i in paths_dict["vuvunira_55__72"]]
# tmp = [print(i) for i in paths_dict["rakawavo_17__25"]]

# +
plot_range = {
    # "y": [2.5, 1.4],
    # "x": [1., 2.6],

    "y": [3.7, 1.4],
    "x": [0.5, 5.],
    }

VP = Volcano_Plot(
    ORR_PLT,
    x_ax_species="o-oh",  # 'o-oh' or 'oh'
    smart_format_dict=smart_format_dict,
    plot_range=plot_range,
    )

VP.create_volcano_relations_plot()

volcano_legs_data = VP.create_volcano_lines(
    gas_molec_dict=gas_molec_dict,
    scaling_dict=scaling_dict_ideal,
    plot_all_legs=False,
    plot_min_max_legs=True,
    trace_priority="bottom",  # 'top' or 'bottom'
    )

data = volcano_legs_data + VP.data_points

layout = VP.get_plotly_layout()

fig = go.Figure(
    data=data,
    layout=layout,
    )

my_plotly_plot(
    figure=fig,
    plot_name="out_plot_02_large")

fig.show()
# -

df_ads.shape

df_ads

# + active=""
#
#
#

# + jupyter={}
# df_i["stoich"] = stoich_i

# + jupyter={}
# # df_m
# df_i["stoich"] = stoich_i

# df_i

# + jupyter={}
# assert False

# + jupyter={}
# layout = 

# # VP.get_plotly_layout?
