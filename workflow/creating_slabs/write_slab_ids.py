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

# # Write Established Slab IDs to a File
# ---
#
# I'm going to write the slab ids to a json file, the reason is that I'm worried since the slab ids are generated randomly upon creation of the slab that I will lose the id  for a specific bulk + facet combination. So I'm going to write the current slab ids to file and in the slab creation scripts if the bulk_id+facet combination already exists it will be taken from there

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import pandas as pd

# #########################################################
from methods import get_df_slab
# -

df_slab = get_df_slab()

print("df_slab.shape:", df_slab.shape)

# +
# assert False

# +
df_i = df_slab[["bulk_id", "facet", "slab_id", ]]

df_i.to_csv("out_data/slab_id_mapping.csv", index=False)
df_i.to_csv("in_data/slab_id_mapping.csv", index=False)
# -

path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs",
    "in_data/slab_id_mapping.csv")
df_slab_ids = pd.read_csv(path_i, dtype=str)

bulk_id_i = "cq7smr6lvj"
# facet = (1, 0, 0)
facet_i = "100"
facet_i = "109"

# +
from methods import get_df_slab_ids, get_slab_id

df_slab_ids = get_df_slab_ids()
slab_id_i = get_slab_id(bulk_id_i, facet_i, df_slab_ids)

slab_id_i
