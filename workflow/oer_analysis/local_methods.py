"""
"""

#| - Import Modules
import os
import sys

# #########################################################
# Python Modules
import numpy as np
import pandas as pd

# #########################################################
# My Modules
from oxr_reaction.oxr_rxn import ORR_Free_E_Plot

from methods import (
    get_df_ads,
    get_df_job_ids,
    get_df_dft,
    )
#__|


def get_ORR_PLT():
    """
    """
    #| - get_ORR_PLT

    # #########################################################
    df_ads = get_df_ads()

    df_ads = df_ads[~df_ads.g_oh.isna()]
    df_m = df_ads

    # #########################################################
    df_dft = get_df_dft()

    # #########################################################
    df_job_ids = get_df_job_ids()


    # df_m.g_ooh = 1.16 * df_m.g_oh + 2.8
    df_m["g_ooh"] = df_m.g_oh + 2.8


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


    df_m = df_m.set_index(["compenv", "slab_id", ], drop=False)

    paths_dict = dict()
    for name_i, row_i in df_m.iterrows():
        #| - Loop through data and add to ORR_PLT
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
        #__|

    return(ORR_PLT)
    #__|
