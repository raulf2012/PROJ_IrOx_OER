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

# # OER Volcano Plot
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

sys.path.insert(
    0, os.path.join(
        os.environ["PROJ_irox"],
        "data"))

# #########################################################
# Python Modules
import numpy as np

import plotly.graph_objs as go
# #########################################################
# My Modules
from oxr_reaction.oxr_plotting_classes.oxr_plot_volcano import Volcano_Plot
from plotting.my_plotly import my_plotly_plot

# #########################################################
# Project Data
from proj_data_irox import (
    smart_format_dict,
    gas_molec_dict,
    scaling_dict_ideal,
    )

# #########################################################
# Local Imports
from local_methods import get_ORR_PLT
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
    show_plot = True
else:
    from tqdm import tqdm
    verbose = False
    show_plot = False

# ### Script Inputs

save_plot = False
plot_exp_traces = True

# + active=""
#
#

# +
# %%capture

ORR_PLT = get_ORR_PLT()

# +
# assert False

# +
plot_range = {
    "y": [3.7, 1.2],
    "x": [0.2, 3.3],
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

# print("Commented out")
my_plotly_plot(
    figure=fig,
    save_dir=os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/oer_analysis"),
    plot_name="out_plot_02_large")
# -

if show_plot:
    fig.show()

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("oer_analysis.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# print(list(paths_dict.keys()))
# print("")

# tmp = [print(i) for i in paths_dict["vuvunira_55__72"]]
# tmp = [print(i) for i in paths_dict["rakawavo_17__25"]]

# + jupyter={"source_hidden": true}
# ORR_PLT

# + jupyter={"source_hidden": true}
# # #########################################################
# df_ads = get_df_ads()

# df_ads = df_ads[~df_ads.g_oh.isna()]
# df_m = df_ads

# # #########################################################
# df_jobs_paths = get_df_jobs_paths()

# # #########################################################
# df_jobs = get_df_jobs()

# # #########################################################
# df_jobs_anal = get_df_jobs_anal()

# # #########################################################
# from methods import get_df_dft
# df_dft = get_df_dft()

# # #########################################################
# from methods import get_df_job_ids
# df_job_ids = get_df_job_ids()
