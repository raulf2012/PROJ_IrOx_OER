# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Plotting scatter plot matrix features
# ---
#

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import numpy as np
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

# # Read Data

# +
from methods import get_df_features_targets
df_features_targets = get_df_features_targets()

from methods import get_df_slab
df_slab = get_df_slab()


# #########################################################
df_i = df_features_targets

# Getting phase > 1 slab ids
df_slab_i = df_slab[df_slab.phase > 1]
phase_2_slab_ids = df_slab_i.slab_id.tolist()
# -

# # Dropping phase 1 slabs

# +
df_index = df_i.index.to_frame()
df_index_i = df_index[
    df_index.slab_id.isin(phase_2_slab_ids)
    ]

print("Dropping phase 1 slabs")
df_i = df_i.loc[
    df_index_i.index
    ]

# +
feat_ads = "o"

# df_i = df_i["features_stan"][feat_ads]
df_i = df_i["features"][feat_ads]
# -

df_i.head()

# +
import plotly.express as px
fig = px.scatter_matrix(df_i)

# print("show_plot:", show_plot)
if show_plot:
    fig.show()

# +
from plotting.my_plotly import my_plotly_plot

my_plotly_plot(
    figure=fig,
    save_dir=os.path.join(
        os.environ["PROJ_irox_oer"],
        # "workflow/oer_vs_features",
        "workflow/feature_engineering/scatter_plot_matrix",
        ),
    plot_name="scatter_matrix_plot",
    write_html=True,
    )
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("scatter_plot_matrix.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_i["features"][feat_ads]
# df_i["features"]
# [feat_ads]

# +
# /mnt/f/Dropbox/01_norskov/00_git_repos/
#     PROJ_IrOx_OER/
#     workflow/feature_engineering/scatter_plot_matrix
