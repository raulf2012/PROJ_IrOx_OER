"""
"""

#| - Import Modules
import numpy as np
# from scipy.spatial import ConvexHull


# #########################################################
from methods import (
    get_df_coord,
    get_df_coord_wrap,
    )


from methods_features import get_octa_geom, get_octa_vol
# from methods import get_df_coord_wrap
from methods import get_octahedra_oxy_neigh
from methods import get_other_job_ids_in_set

from proj_data import metal_atom_symbol
#__|



# df_jobs=df_jobs
# df_init_slabs=df_init_slabs
# atoms_0=atoms_i
# job_id_0=job_id_max_i
# active_site=active_site_i
# compenv=compenv_i
# slab_id=slab_id_i
# ads_0=ads_i
# active_site_0=active_site_orig_i
# att_num_0=att_num_i

def get_octahedra_atoms(
    df_jobs=None,
    df_init_slabs=None,
    atoms_0=None,
    job_id_0=None,
    # #####################################################

    active_site=None,
    compenv=None,
    slab_id=None,
    ads_0=None,
    active_site_0=None,
    att_num_0=None,
    ):
    """
    """
    # | - get_octahedra_atoms

    df_coord_0 = get_df_coord_wrap(
        name=(
            compenv, slab_id,
            ads_0, active_site_0, att_num_0),
        active_site=active_site,
        )

    mean_displacement_octahedra = None
    note_i = None
    error_i = None
    metal_active_site = None
    octahedra_atoms = None

    if df_coord_0 is not None:
        # | - If df_coord can be found (NORMAL)

        # | - If something went wrong or ads is bare then get metal index some other way

        num_neighbors = None
        if ads_0 != "bare":
            row_coord_i = df_coord_0[
                df_coord_0.structure_index == active_site]
            row_coord_i = row_coord_i.iloc[0]
            num_neighbors = row_coord_i.num_neighbors


        active_metal_index = None
        if ads_0 == "bare" or row_coord_i.num_neighbors == 0:
            df_other_jobs = get_other_job_ids_in_set(
                job_id_0, df_jobs=df_jobs,
                oer_set=True, only_last_rev=True,
                )

            df = df_other_jobs
            df = df[
                (df["ads"] == "o") &
                (df["active_site"] == "NaN") &
                [True for i in range(len(df))]
                ]
            row_o_jobs = df.iloc[0]



            name_tmp_i = (
                row_o_jobs["compenv"], row_o_jobs["slab_id"],
                row_o_jobs["ads"], row_o_jobs["active_site"],
                row_o_jobs["att_num"], )
            df_coord_o = get_df_coord_wrap(name=name_tmp_i, active_site=active_site_0)
            row_coord = df_coord_o.loc[active_site_0]

            mess_i = "TMP TMP"
            assert row_coord.neighbor_count[metal_atom_symbol] == 1, mess_i

            # mess_i = "TMP TMP 2"
            # assert len(row_coord.nn_info) == 1, mess_i

            nn_metal_i = None
            num_metal_neigh = 0
            for nn_i in row_coord.nn_info:
                if nn_i["site"].specie.symbol == metal_atom_symbol:
                    nn_metal_i = nn_i
                    num_metal_neigh += 1

            assert num_metal_neigh < 2, "IDJFSD"

            active_metal_index = nn_metal_i["site_index"]

            active_metal_index = row_coord.nn_info[0]["site_index"]
        # __|


        octahedra_data = get_octahedra_oxy_neigh(
            df_coord=df_coord_0,
            ads=ads_0,
            active_site=active_site,
            metal_active_site=active_metal_index,

            compenv=compenv,
            slab_id=slab_id,
            df_init_slabs=df_init_slabs,
            atoms_0=atoms_0,

            )

        metal_active_site = octahedra_data["metal_active_site"]
        octahedral_oxygens = octahedra_data["octahedral_oxygens"]


        if octahedral_oxygens is not None:
            octahedra_atoms = octahedral_oxygens + [metal_active_site, ]

        # __|
    else:
        note_i = "Couldn't get df_coord, came back as None"
        error_i = True

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict.update(octahedra_data)
    # #####################################################
    out_dict["octahedra_atoms"] = octahedra_atoms
    # #####################################################
    return(out_dict)
    # #####################################################
    # __|
