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

# # I will collate plots from the project here for ease of access
# ---

# ### Import Modules

# +
import os
import sys

import json

import plotly.graph_objects as go
from plotly.io import from_json

# + active=""
#
#
# -

# ## Linear models (targets: G_O and G_OH) using all features

# + jupyter={}
path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/oer_vs_features",
    "out_plot/oer_lin_model__G_O_plot.json")

with open(path_i, "r") as fle:
    fig_json_str_i = fle.read()

fig = from_json(fig_json_str_i)
fig.show()

# + jupyter={}
path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/oer_vs_features",
    "out_plot/oer_lin_model__G_OH_plot.json")

with open(path_i, "r") as fle:
    fig_json_str_i = fle.read()

fig = from_json(fig_json_str_i)
fig.show()

# + active=""
#
#
# -

# ## Linear scaling plot (*O vs *OH)

# +
path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/oer_analysis/oer_scaling", 
    "out_plot/oer_scaling__O_vs_OH_plot.json")

with open(path_i, "r") as fle:
    fig_json_str_i = fle.read()

fig = from_json(fig_json_str_i)
fig.show()

# + active=""
#
#
# -

# ## G_O and G_OH histograms

# +
path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/oer_analysis/oer_scaling", 
    "out_plot/oer_scaling__O_OH_histogram.json")

with open(path_i, "r") as fle:
    fig_json_str_i = fle.read()

fig = from_json(fig_json_str_i)
fig.show()

# + active=""
#
#
#
#

# + jupyter={}
# fig.write_json(
#     os.path.join(
#         os.environ["PROJ_irox_oer"],
#         "workflow/oer_analysis/oer_scaling", 
#         "out_plot/oer_scaling__O_OH_histogram.json"))
