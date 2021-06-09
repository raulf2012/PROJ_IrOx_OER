"""
"""

#| - Import Modules
import numpy as np

# #########################################################
from methods import (
    get_df_coord,
    get_df_coord_wrap,
    )
from methods_features import get_octa_geom, get_octa_vol
from methods import get_octahedra_oxy_neigh
from methods import get_other_job_ids_in_set

from proj_data import metal_atom_symbol
#__|



# name=name_i
# active_site=active_site_i
# active_site_original=active_site_orig_i
# atoms=atoms_i
# octahedra_atoms=octahedra_atoms_i
# metal_active_site=metal_active_site_i
# df_coord=None
# verbose=verbose

def process_row_2(
    name=None,
    active_site=None,
    active_site_original=None,
    atoms=None,
    octahedra_atoms=None,
    metal_active_site=None,
    df_coord=None,
    verbose=False,
    ):
    """
    """
    #| - process_row
    # #####################################################
    name_i = name
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################

    if df_coord is None:
        df_coord_i = get_df_coord_wrap(name=name_i, active_site=active_site)
    else:
        df_coord_i = df_coord

    if df_coord_i is None:
        print(10 * "df_coord_i = None \n")
        print(name_i)

    # #################################################
    from local_methods import get_octa_vol
    vol_i = get_octa_vol(
        df_coord_i=df_coord_i,
        active_site_j=active_site,
        octahedra_atoms=octahedra_atoms,
        metal_active_site=metal_active_site,
        atoms=atoms,
        verbose=verbose,
        )
    # #################################################
    from local_methods import get_octa_geom
    octa_geom_dict = get_octa_geom(
        df_coord_i=df_coord_i,
        active_site_j=active_site,
        atoms=atoms,
        octahedra_atoms=octahedra_atoms,
        verbose=verbose)
    # #################################################


    # #################################################
    data_dict_i = dict()
    # #################################################
    data_dict_i.update(octa_geom_dict)
    # #################################################
    data_dict_i["active_site"] = active_site
    data_dict_i["octa_vol"] = vol_i
    # #################################################
    return(data_dict_i)
    # #################################################

    #__|
