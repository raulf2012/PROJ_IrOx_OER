"""Local methods for workflow/octahedra_info."""

#| - Import Modules
import numpy as np
import pandas as pd

import math

# #########################################################
from methods import (
    get_df_coord,
    get_df_coord_wrap,
    get_octahedra_oxy_neigh,
    get_other_job_ids_in_set,
    )
from methods import translate_position_to_image
from methods import angle_between

from methods_features import get_octa_geom, get_octa_vol

from proj_data import metal_atom_symbol
#__|








def read_df_coord(
    init_or_final=None,
    compenv=None,
    slab_id=None,
    ads=None,
    active_site=None,
    att_num=None,
    ):
    """
    """
    # | - read_df_coord
    if init_or_final == "final":
        df_coord = get_df_coord_wrap(
            name=(
                compenv, slab_id,
                ads, active_site, att_num),
            active_site=active_site,
            )
    elif init_or_final == "init":

        init_slab_name_tuple = (
            compenv, slab_id, ads,
            active_site, att_num,
            )

        df_coord = get_df_coord(
            mode="init-slab",  # 'bulk', 'slab', 'post-dft', 'init-slab'
            init_slab_name_tuple=init_slab_name_tuple,
            )

        # df_coord_i.loc[42]["nn_info"]
    else:
        print("HAVE TO SET `init_or_final` variable")


    return(df_coord)
    # __|
