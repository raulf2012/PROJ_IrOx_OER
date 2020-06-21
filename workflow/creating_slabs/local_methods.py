"""
"""

#| - Import Modules
import os
import sys

import numpy as np

# #############################################################################
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder

from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    SimplestChemenvStrategy,
    MultiWeightsChemenvStrategy)
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments

from pymatgen.io.ase import AseAtomsAdaptor
#__|


def analyse_local_coord_env(
    atoms=None,

    ):
    """
    """
    #| - analyse_local_coord_env
    # print("RUNNING!")
    # atoms = row_i.atoms
    out_dict = dict()

    Ir_indices = []
    for i, s in enumerate(atoms.get_chemical_symbols()):
        if s == "Ir":
            Ir_indices.append(i)

    struct = AseAtomsAdaptor.get_structure(atoms)

    # TEMP
    # Pickling data ###########################################
    out_dict = dict()
    out_dict["struct"] = struct

    import os; import pickle
    path_i = os.path.join(
        os.environ["HOME"],
        "__temp__",
        "temp.pickle")
    with open(path_i, "wb") as fle:
        pickle.dump(out_dict, fle)
    # #########################################################


    lgf = LocalGeometryFinder()
    lgf.setup_structure(structure=struct)

    se = lgf.compute_structure_environments(
        maximum_distance_factor=1.41,
        only_cations=False)


    strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()

    lse = LightStructureEnvironments.from_structure_environments(
        strategy=strategy, structure_environments=se)

    isite = 0
    cor_env = []
    coor_env_dict = dict()
    for isite in Ir_indices:
        c_env = lse.coordination_environments[isite]
        coor_env_dict[isite] = c_env
        cor_env += [c_env[0]['ce_symbol']]

    out_dict["coor_env_dict"] = coor_env_dict

    # uniques, counts = np.unique(cor_env, return_counts=True)
    # if len(uniques) > 1:
    #     coor = 'mixed'
    #     coor_l = [int(c.split(':')[-1]) for c in cor_env]
    #     mean_coor = np.mean(coor_l)
    # else:
    #     coor = uniques[0]
    #     mean_coor = int(coor.split(':')[-1])

    return(out_dict)
    #__|

def check_if_sys_processed(
    bulk_id_i=None,
    facet_str=None,
    df_slab_old=None,
    ):
    """
    """
    #| - check_if_sys_processed
    sys_processed = False

    num_rows_tot = df_slab_old.shape[0]
    if num_rows_tot > 0:
        df = df_slab_old
        df = df[
            (df["bulk_id"] == bulk_id_i) &
            (df["facet"] == facet_str) &
            [True for i in range(len(df))]
            ]

        num_rows = df.shape[0]
        if num_rows > 0:
            # print("There is a row that already exists")

            sys_processed = True
            if num_rows > 1:
                print("There is more than 1 row for this bulk+facet combination, what to do?")

            row_i = df.iloc[0]

    return(sys_processed)
    #__|
