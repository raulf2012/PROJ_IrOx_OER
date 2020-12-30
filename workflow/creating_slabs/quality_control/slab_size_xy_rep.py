# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Writing slabs that ended up with too many atoms to file (before and afte xy-repitition)
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import numpy as np

from methods import get_df_slab
# -

slab_ids = [
    'lagubapi_05',
    'bipigode_78',
    'paritile_76',
    'gowitepe_67',
    'gekawore_16',
    'vapopihe_87',
    'kuwurupu_88',
    'fodopilu_17',
    'dawakuvo_55',
    'wihuwone_95',
    'taparogo_67',
    'lufinanu_76',
    'helisogi_82',
    'gerokede_95',
    'vugupopo_89',
    'viholaba_21',
    'runopeno_56',
    'sosewedu_22',
    ]

# +
df_slab_0 = get_df_slab(mode="almost-final")
df_slab_1 = get_df_slab(mode="final")
# df_slab_1 =  df_slab

df_slab_0.loc[
    slab_ids
    ]

for slab_id_i in slab_ids:

    row_1 = df_slab_1.loc[slab_id_i]
    row_0 = df_slab_0.loc[slab_id_i]

    slab_final_1 = row_1.slab_final
    slab_final_0 = row_0.slab_final

    cell = slab_final_0.cell.array
    x_mag_0 = np.linalg.norm(cell[0])
    y_mag_0 = np.linalg.norm(cell[1])

    cell = slab_final_1.cell.array
    x_mag_1 = np.linalg.norm(cell[0])
    y_mag_1 = np.linalg.norm(cell[1])

    x_y_mag_0 = "_" + str(np.round(x_mag_0, decimals=1)) + "_" + str(np.round(y_mag_0, decimals=1))
    x_y_mag_1 = "_" + str(np.round(x_mag_1, decimals=1)) + "_" + str(np.round(y_mag_1, decimals=1))

    num_atoms_0 = slab_final_0.get_global_number_of_atoms()
    num_atoms_1 = slab_final_1.get_global_number_of_atoms()

    out_dir = "out_data/comparing_slabs_w_repetition"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_name_pre = slab_id_i + "_"

    slab_final_0.write(
        os.path.join(
            out_dir,
            file_name_pre + "0_" + str(num_atoms_0).zfill(3) + x_y_mag_0 + ".cif"))
            # file_name_pre + "0_" + str(num_atoms_0).zfill(3) + str(x_mag_0) + ".cif"))

    slab_final_1.write(
        os.path.join(
            out_dir,
            file_name_pre + "1_" + str(num_atoms_1).zfill(3) + x_y_mag_1 + ".cif"))

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# import copy
# import shutil
# from pathlib import Path
# from contextlib import contextmanager

# # import pickle; import os

# import pickle
# import  json

# import pandas as pd
# import numpy as np

# from ase import io

# import plotly.graph_objects as go

# from pymatgen.io.ase import AseAtomsAdaptor
# from pymatgen.analysis import local_env

# # #########################################################
# from misc_modules.pandas_methods import drop_columns

# + jupyter={"source_hidden": true}
# from IPython.display import display

# import pandas as pd
# pd.set_option("display.max_columns", None)
# pd.options.display.max_colwidth = 20
# # pd.set_option('display.max_rows', None)


# # #########################################################
# from methods import (
#     get_df_jobs_paths,
#     get_df_dft,
#     get_df_job_ids,
#     get_df_jobs,
#     get_df_jobs_data,
#     get_df_slab_ids,
#     get_df_jobs_data_clusters,
#     get_df_jobs_anal,
#     get_df_slabs_oh,
#     get_df_init_slabs,
#     get_df_magmoms,
#     get_df_ads,
#     get_df_atoms_sorted_ind,
#     get_df_rerun_from_oh,
#     get_df_slab_simil,
#     )
# from methods import (
#     get_other_job_ids_in_set,
#     read_magmom_comp_data,
#     )
