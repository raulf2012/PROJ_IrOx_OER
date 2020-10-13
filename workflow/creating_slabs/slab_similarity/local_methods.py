"""
"""

#| - Import Modules
import os

import pickle

from pathlib import Path

from StructurePrototypeAnalysisPackage.ccf import struc2ccf
#__|


def get_ccf(
    slab_id=None,
    slab_final=None,
    verbose=True,
    r_cut_off=None,
    r_vector=None,
    ):
    """
    """
    #| - get_ccf_i

    # #####################################################
    global os
    global pickle

    # #####################################################
    slab_id_i = slab_id
    slab_final_i = slab_final
    # #####################################################

    directory = "out_data/ccf_files"
    name_i = slab_id_i + ".pickle"
    # print("os:", os)
    file_path_i = os.path.join(directory, name_i)

    my_file = Path(file_path_i)
    if my_file.is_file():
        if verbose:
            print("File exists already")

        # #################################################
        import pickle; import os
        path_i = os.path.join(
            os.environ["PROJ_irox_oer"],
            "workflow/creating_slabs/slab_similarity",
            file_path_i)
        with open(path_i, "rb") as fle:
            ccf_i = pickle.load(fle)
        # #################################################
    else:
        ccf_i = struc2ccf(slab_final_i, r_cut_off, r_vector)


        # Pickling data ###################################
        if not os.path.exists(directory): os.makedirs(directory)
        with open(file_path_i, "wb") as fle:
            pickle.dump(ccf_i, fle)
        # #################################################

    return(ccf_i)
    #__|
