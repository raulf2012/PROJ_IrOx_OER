# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # OER Analysis Notebook
# ---
#
# * Compute overpotential for all systems
# * Save ORR_PLT instance for OXR plotting classes
# * Save df_overpot dataframe to combine with df_features_targets

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

# #########################################################
# Python Modules
import pickle

import numpy as np
import pandas as pd

# #########################################################
# My Modules
from oxr_reaction.oxr_rxn import ORR_Free_E_Plot

from methods import (
    get_df_ads,
    get_df_job_ids,
    get_df_dft,
    get_df_features_targets,
    )
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

# +
# #########################################################
df_dft = get_df_dft()

# #########################################################
df_job_ids = get_df_job_ids()

# #########################################################
df_features_targets = get_df_features_targets()

# +
if verbose:
    print(
        "Change in size of df_features from dropping non-complete rows:"

        "\n",
        df_features_targets.shape[0],
        sep="")

# Only passing through OER sets that are 100% done will all calculations
# if True:
if False:
    df_features_targets = df_features_targets[df_features_targets["data"]["all_done"] == True]


if verbose:
    print(
        df_features_targets.shape[0],
        sep="")

# +
smart_format_dict = [
    [{"stoich": "AB2"}, {"color2": "black"}],
    [{"stoich": "AB3"}, {"color2": "grey"}],
    ]

ORR_PLT = ORR_Free_E_Plot(
    free_energy_df=None,
    state_title="ads",
    free_e_title="ads_g",
    smart_format=smart_format_dict,
    color_list=None,
    rxn_type="OER")


# new_col = (df_features_targets["targets"]["g_oh"] + 2.8)
new_col = (1.16 * df_features_targets["targets"]["g_oh"] + 2.8)

new_col.name = ("targets", "g_ooh", "", )

df_features_targets = pd.concat([
    new_col,
    df_features_targets,
    ], axis=1)



# Loop through data and add to ORR_PLT
data_dict_list_0 = []
for name_i, row_i in df_features_targets.iterrows():


    # #####################################################
    g_o_i = row_i[("targets", "g_o", "", )]
    g_oh_i = row_i[("targets", "g_oh", "", )]
    g_ooh_i = row_i[("targets", "g_ooh", "", )]
    slab_id_i = row_i[("data", "slab_id", "")]
    active_site_i = row_i[("data", "active_site", "")]
    job_id_o_i = row_i[("data", "job_id_o", "")]
    job_id_oh_i = row_i[("data", "job_id_oh", "")]
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
    name_i_2 = slab_id_i + "__" + str(int(active_site_i))
    ORR_PLT.add_series(
        df_i,
        plot_mode="all",
        overpotential_type="OER",
        property_key_list=prop_name_list,
        add_overpot=False,
        name_i= name_i_2,
        )

    # #################################################
    data_dict_i = dict()
    # #################################################
    data_dict_i["name"] = name_i_2
    data_dict_i["compenv"] = name_i[0]
    data_dict_i["slab_id"] = name_i[1]
    data_dict_i["active_site"] = name_i[2]
    # #################################################
    data_dict_list_0.append(data_dict_i)
    # #################################################


df = pd.DataFrame(data_dict_list_0)
df = df.set_index("name", drop=False)

# +
data_dict_list = []
for OXR_Series_i in ORR_PLT.series_list:

    name_i = OXR_Series_i.name_i

    # #####################################################
    overpot_out = OXR_Series_i.calc_overpotential_OER()
    # #####################################################
    overpot_i = overpot_out[0]
    lim_step_i = overpot_out[1]
    # #####################################################


    if lim_step_i == ["bulk", "oh"]:
        lim_step_str_i = "bulk__oh"
        lim_step_num = 1
    elif lim_step_i == ["oh", "o"]:
        lim_step_str_i = "oh__o"
        lim_step_num = 2
    elif lim_step_i == ["o", "ooh"]:
        lim_step_str_i = "o__ooh"
        lim_step_num = 3
    elif lim_step_i == ["ooh", "bulk"]:
        lim_step_str_i = "ooh__bulk"
        lim_step_num = 4

    else:
        print("WOOOOOPS")
        print(lim_step_i)


    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i["name"] = name_i
    data_dict_i["overpot"] = overpot_i
    data_dict_i["lim_step"] = lim_step_i
    data_dict_i["lim_step_str"] = lim_step_str_i
    data_dict_i["lim_step_num"] = lim_step_num
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

df_overpot = pd.DataFrame(data_dict_list)
df_overpot = df_overpot.set_index("name", drop=True)

# +
df_overpot = pd.concat([df, df_overpot], axis=1)

df_overpot = df_overpot.set_index(
    ["compenv", "slab_id", "active_site", ])
# -

# ### Saving data to file

# +
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/oer_analysis",
    "out_data")

# Pickling data ###########################################
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_overpot.pickle"), "wb") as fle:
    pickle.dump(df_overpot, fle)
# #########################################################
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "ORR_PLT.pickle"), "wb") as fle:
    pickle.dump(ORR_PLT, fle)
# #########################################################
# -

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
