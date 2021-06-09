
# #########################################################
# Local methods for setup_jobs_from_oh
# #########################################################

#| - Import Modules
import os
import sys

import copy

from methods import get_df_coord
#__|



def get_bare_o_from_oh(
    compenv=None,
    slab_id=None,
    active_site=None,
    att_num=None,
    atoms=None,
    ):
    """
    """
    #| - get_bare_o_from_oh

    # #####################################################
    compenv_i = compenv
    slab_id_i = slab_id
    active_site_i = active_site
    att_num_i = att_num
    # #####################################################

    name_i = (compenv_i, slab_id_i, "oh", active_site_i, att_num_i, )
    df_coord_i = get_df_coord(
        slab_id=None,
        bulk_id=None,
        mode="post-dft",  # 'bulk', 'slab', 'post-dft'
        slab=None,
        post_dft_name_tuple=name_i,
        porous_adjustment=True,
        )

    row_coord_i = df_coord_i[df_coord_i.element == "H"]

    mess_i = "isdjfisdif"
    assert row_coord_i.shape[0] == 1, mess_i

    row_coord_i = row_coord_i.iloc[0]

    h_index_i = row_coord_i.structure_index

    nn_info_i = row_coord_i.nn_info

    # mess_i = "Should only be 1 *O atom attached to *H here"
    # assert(len(nn_info_i)) == 1, mess_i

    #| - Reading df_coord with porous_adjustment turned off
    if len(nn_info_i) != 1:
        name_i = (compenv_i, slab_id_i, "oh", active_site_i, att_num_i, )
        df_coord_i = get_df_coord(
            slab_id=None,
            bulk_id=None,
            mode="post-dft",  # 'bulk', 'slab', 'post-dft'
            slab=None,
            post_dft_name_tuple=name_i,
            porous_adjustment=False,
            )

        row_coord_i = df_coord_i[df_coord_i.element == "H"]

        mess_i = "isdjfisdif"
        assert row_coord_i.shape[0] == 1, mess_i

        row_coord_i = row_coord_i.iloc[0]

        h_index_i = row_coord_i.structure_index

        nn_info_i = row_coord_i.nn_info

        mess_i = "Should only be 1 *O atom attached to *H here"
        assert(len(nn_info_i)) == 1, mess_i
    #__|


    nn_info_j = nn_info_i[0]

    site_j = nn_info_j["site"]
    elem_j = site_j.specie.as_dict()["element"]

    mess_i = "Must be an *O atom that *H is attached to"
    assert elem_j == "O", mess_i

    site_index_j = nn_info_j["site_index"]


    # #########################################################
    # #########################################################
    # #########################################################
    # #########################################################
    # #########################################################


    # atoms = atoms_i


    atoms_new = copy.deepcopy(atoms)

    # #########################################################
    indices_to_remove = [site_index_j, h_index_i]

    mask = []
    for atom in atoms_new:
        if atom.index in indices_to_remove:
            mask.append(True)
        else:
            mask.append(False)

    del atoms_new[mask]

    atoms_bare = atoms_new


    # #########################################################
    atoms_new = copy.deepcopy(atoms)

    indices_to_remove = [h_index_i, ]

    mask = []
    for atom in atoms_new:
        if atom.index in indices_to_remove:
            mask.append(True)
        else:
            mask.append(False)

    del atoms_new[mask]

    atoms_O = atoms_new

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["atoms_bare"] = atoms_bare
    out_dict["atoms_O"] = atoms_O
    # #####################################################

    return(out_dict)
    #__|
