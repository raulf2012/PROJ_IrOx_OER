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

# # Collect feature data into master dataframe
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
from itertools import combinations
from collections import Counter
from functools import reduce

from IPython.display import display

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
# pd.set_option('display.max_rows', None)
pd.options.display.max_colwidth = 100

# #########################################################
from methods import (
    get_df_job_ids,
    get_df_atoms_sorted_ind,
    get_df_jobs_paths,
    get_df_dft,

    get_df_octa_vol,
    get_df_eff_ox,
    get_df_angles,
    get_df_pdos_feat,
    get_df_bader_feat,

    get_df_coord,
    )
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Read feature dataframes

# +
# Base dataframes
df_dft = get_df_dft()

df_job_ids = get_df_job_ids()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_jobs_paths = get_df_jobs_paths()


# Features dataframes
df_octa_vol = get_df_octa_vol()

df_eff_ox = get_df_eff_ox()

df_angles = get_df_angles()

df_pdos_feat = get_df_pdos_feat()

df_bader_feat = get_df_bader_feat()
# -

# ### Filtering down to `oer_adsorbate` jobs

df_ind = df_atoms_sorted_ind.index.to_frame()
df_atoms_sorted_ind = df_atoms_sorted_ind.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_atoms_sorted_ind = df_atoms_sorted_ind.droplevel(level=0)

# +
from local_methods import combine_dfs_with_same_cols

df_dict_i = {
    "df_eff_ox": df_eff_ox,
    "df_octa_vol": df_octa_vol,
    "df_angles": df_angles,
    "df_pdos_feat": df_pdos_feat,
    "df_bader_feat": df_bader_feat,
    }

df_features = combine_dfs_with_same_cols(
    df_dict=df_dict_i,
    verbose=verbose,
    )


# -

# ### Adding in bulk data

# +
def method(row_i):
    new_column_values_dict = {
        "dH_bulk": None,
        "volume_pa": None,
        "bulk_oxid_state": None,
        }


    # #####################################################
    slab_id_i = row_i.name[1]
    # #####################################################
    bulk_ids = df_job_ids[df_job_ids.slab_id == slab_id_i].bulk_id.unique()
    mess_i = "ikjisdjf"
    assert len(bulk_ids) == 1, mess_i
    bulk_id_i = bulk_ids[0]
    # #####################################################

    # #####################################################
    row_dft_i = df_dft.loc[bulk_id_i]
    # #####################################################
    dH_i = row_dft_i.dH
    volume_pa = row_dft_i.volume_pa
    stoich_i = row_dft_i.stoich
    # #####################################################

    if stoich_i == "AB2":
        bulk_oxid_state_i = +4
    elif stoich_i == "AB3":
        bulk_oxid_state_i = +6
    else:
        print("Uh oh, couldn't parse bulk stoich, not good")

    # #####################################################
    new_column_values_dict["dH_bulk"] = dH_i
    new_column_values_dict["volume_pa"] = volume_pa
    new_column_values_dict["bulk_oxid_state"] = bulk_oxid_state_i
    # #####################################################
    for key, value in new_column_values_dict.items():
        row_i[("features", key)] = value
    return(row_i)

df_features = df_features.apply(method, axis=1)
df_features = df_features.reindex(columns = ["data", "features", ], level=0)

# +
if verbose:
    print("df_features.shape:", df_features.shape)

# df_features.head()
# -

# ### Adding magmom data (Spin)

# +
data_dict_list = []
index_list = []
for i_cnt, (name_i, row_i) in enumerate(df_features.iterrows()):
    index_list.append(name_i)
    name_i_2 = name_i[0:-1]

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    from_oh_i = name_i[5]
    # #####################################################
    job_id_max_i = row_i["data"]["job_id_max"]
    # #####################################################

    if ads_i == "o" and not from_oh_i:
        name_new_i = (
            compenv_i, slab_id_i, ads_i, "NaN", att_num_i, )
    else:
        name_new_i = name_i_2


    # #########################################################
    row_paths_i = df_jobs_paths.loc[job_id_max_i]
    # #########################################################
    gdrive_path_i = row_paths_i.gdrive_path
    # #########################################################

    # #####################################################
    row_atoms_i = df_atoms_sorted_ind.loc[name_new_i]
    # #####################################################
    magmoms_i = row_atoms_i.magmoms_sorted_good
    atoms_i = row_atoms_i.atoms_sorted_good
    # #####################################################

    if magmoms_i is None:
        magmoms_i = atoms_i.get_magnetic_moments()

    magmom_active_site_i = magmoms_i[int(active_site_i)]




    init_name_i = (compenv_i, slab_id_i, "o", "NaN", 1)

    df_coord_i = get_df_coord(
        mode='init-slab',
        init_slab_name_tuple=init_name_i,
        )

    row_coord_i = df_coord_i.loc[active_site_i]

    Ir_nn_found = False
    nn_Ir = None
    for nn_i in row_coord_i["nn_info"]:
        symbol_i = nn_i["site"].specie.symbol
        if symbol_i == "Ir":
            nn_Ir = nn_i
            Ir_nn_found = True

    Ir_bader_charge_i = None
    if Ir_nn_found:
        Ir_index = nn_Ir["site_index"]
    else:
        print("Ir not found")

    Ir_magmom_i = magmoms_i[int(Ir_index)]


    # #####################################################
    data_dict_i = dict()
    # #####################################################
    # data_dict_i["magmom_active_site"] = np.abs(magmom_active_site_i)
    data_dict_i["O_magmom"] = np.abs(magmom_active_site_i)
    data_dict_i["Ir_magmom"] = np.abs(Ir_magmom_i)
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################




# #########################################################
df_magmom_i = pd.DataFrame(
    data_dict_list,
    index=pd.MultiIndex.from_tuples(
        index_list,
        names=list(df_features.index.names),
        )
    )

# Add level to column index to match `df_features`
new_cols = []
for col_i in df_magmom_i.columns:
    new_col_i = ("features", col_i)
    new_cols.append(new_col_i)
df_magmom_i.columns = pd.MultiIndex.from_tuples(new_cols)

df_features = pd.concat([
    df_magmom_i,
    df_features,
    ], axis=1)

df_features = df_features.reindex(
    columns=list(df_features.columns.levels[0]),
    level=0)
# #########################################################
# -

# ### Save data to pickle

# +
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering")

# Pickling data ###########################################
directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(root_path_i, "out_data/df_features.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_features, fle)
# #########################################################

# #########################################################
import pickle; import os
with open(path_i, "rb") as fle:
    df_features = pickle.load(fle)
# #########################################################
# -

from methods import get_df_features
get_df_features().head()

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("collect_feature_data.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
