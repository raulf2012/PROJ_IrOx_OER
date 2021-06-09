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

# # Get all active sites for slabs
# ---
#
# Analyze slabs for active sites

# # Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

import pickle


import numpy as np
import pandas as pd

from ase import io

# # from tqdm import tqdm
from tqdm.notebook import tqdm

# # #########################################################
from proj_data import metal_atom_symbol

# #########################################################
from methods import (
    get_df_slab,
    get_structure_coord_df,
    get_df_coord,
    get_df_active_sites,
    )

# #########################################################
from local_methods import (
    mean_O_metal_coord,
    get_all_active_sites,
    get_unique_active_sites,
    get_unique_active_sites_temp,
    )
# -

# # Read Data

# +
# #########################################################
df_slab = get_df_slab()
df_slab = df_slab.set_index("slab_id")

# #########################################################
df_active_sites_prev = get_df_active_sites()

if df_active_sites_prev is None:
    df_active_sites_prev = pd.DataFrame()
# -

# # Create Directories

# directory = "out_data"
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/enumerate_adsorption",
    "out_data")
# assert False, "Fix os.makedirs"
if not os.path.exists(directory):
    os.makedirs(directory)

# +
# # df_active_sites_prev.loc[[
# df_active_sites_prev = df_active_sites_prev.drop([
#     "pumusuma_66",
#     "fufalego_15",
#     "tefenipa_47",
#     "silovabu_91",
#     "naronusu_67",
#     "nofabigo_84",
#     "kodefivo_37",
#     ])

# +
# # df_slab_i = 
# df_slab.loc[[
#     "pumusuma_66",
#     "fufalego_15",
#     "tefenipa_47",
#     "silovabu_91",
#     "naronusu_67",
#     "nofabigo_84",
#     "kodefivo_37",
#     ]]

# +
slab_ids_to_proc = []
for slab_id_i, row_i in df_slab.iterrows():
    if slab_id_i not in df_active_sites_prev.index:
        slab_ids_to_proc.append(slab_id_i)

df_slab_i = df_slab.loc[
    slab_ids_to_proc
    ]

df_slab_i = df_slab_i[df_slab_i.phase == 2]
# -

df_slab_i



# +
# df_slab_i = df_slab_i.loc[["pemupehe_18"]]

# df_slab_i

# +
# assert False

# +
# #########################################################
data_dict_list = []
# #########################################################
iterator = tqdm(df_slab_i.index, desc="1st loop")
for i_cnt, slab_id in enumerate(iterator):
    print(i_cnt, slab_id)
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    row_i = df_slab.loc[slab_id]
    # #####################################################
    slab = row_i.slab_final
    slab_id = row_i.name
    bulk_id = row_i.bulk_id
    facet = row_i.facet
    num_atoms = row_i.num_atoms
    # #####################################################

    # #################################################
    df_coord_slab_i = get_df_coord(
        slab_id=slab_id,
        mode="slab",
        slab=slab,
        )

    # #################################################
    active_sites = get_all_active_sites(
        slab=slab,
        slab_id=slab_id,
        bulk_id=bulk_id,
        df_coord_slab_i=df_coord_slab_i,
        )

    # #################################################
    # active_sites_unique = get_unique_active_sites(
    active_sites_unique = get_unique_active_sites_temp(
        slab=slab,
        active_sites=active_sites,
        bulk_id=bulk_id,
        slab_id=slab_id,
        facet=facet,
        metal_atom_symbol=metal_atom_symbol,
        df_coord_slab_i=df_coord_slab_i,
        create_heatmap_plot=True,
        )


    # #################################################
    data_dict_i["active_sites"] = active_sites
    data_dict_i["num_active_sites"] = len(active_sites)
    data_dict_i["active_sites_unique"] = active_sites_unique
    data_dict_i["num_active_sites_unique"] = len(active_sites_unique)
    data_dict_i["slab_id"] = slab_id
    data_dict_i["bulk_id"] = bulk_id
    data_dict_i["facet"] = facet
    data_dict_i["num_atoms"] = num_atoms
    # #####################################################
    data_dict_list.append(data_dict_i)


# #########################################################
df_active_sites = pd.DataFrame(data_dict_list)
df_active_sites = df_active_sites.set_index("slab_id", drop=False)

df_active_sites = df_active_sites = pd.concat([
    df_active_sites,
    df_active_sites_prev,
    ])

# +
# from plotting.my_plotly import my_plotly_plot

# # my_plotly_plot?

# +
# assert False
# -

# # Post-process active site dataframe

# +
from misc_modules.pandas_methods import reorder_df_columns

columns_list = [
    'bulk_id',
    'slab_id',
    'facet',
    'num_atoms',
    'num_active_sites',
    'active_sites',
    ]

df_active_sites = reorder_df_columns(columns_list, df_active_sites)
# -

# # Summary of data objects

print(
    "Number of active sites:",
    df_active_sites.num_active_sites.sum())
print(
    "Number of unique active sites",
    df_active_sites.num_active_sites_unique.sum())

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/enumerate_adsorption",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_active_sites.pickle"), "wb") as fle:
    pickle.dump(df_active_sites, fle)
    # pickle.dump(df_active_sites_prev, fle)
# #########################################################

print(df_active_sites.shape)

assert False

# + active=""
#
#
#
#
#
# -

df_rdf_ij_dict = dict()
for i_cnt, row_i in df_active_sites.iterrows():
    file_name_i = row_i.bulk_id + "__" + row_i.facet + \
        "__" + row_i.slab_id + ".pickle"
    path_i = os.path.join(
        "out_data/df_rdf_ij", file_name_i)

    # #########################################################
    import pickle; import os
    with open(path_i, "rb") as fle:
        df_rdf_ij_i = pickle.load(fle)
    # #########################################################

    df_rdf_ij_dict[row_i.slab_id] = df_rdf_ij_i

rdf_ij_list = [i for i in df_rdf_ij_i.values.flatten() if i != 0.]

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# Combining previous `df_active_sites` and the rows processed during current run

# df_active_sites = df_active_sites = pd.concat([
#     df_active_sites,
#     df_active_sites_prev,
#     ])
