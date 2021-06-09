# TEMP

# | - Import Modules
import os
import sys

import pandas as pd

# #########################################################
from methods import get_other_job_ids_in_set
# __|


def write_other_jobs_in_set(
    job_id,
    dir_path=None,
    df_jobs=None,
    df_atoms=None,
    df_jobs_paths=None,
    df_jobs_data=None,
    ):
    """
    """
    # | - write_other_jobs_in_set
    df_jobs_oer_set_i = get_other_job_ids_in_set(
        job_id,
        df_jobs=df_jobs,
        oer_set=True,
        only_last_rev=True)

    # | - Writing data to csv
    df_data_i = pd.concat([
        df_jobs_oer_set_i,
        df_jobs_data.loc[
            df_jobs_oer_set_i.index
            ],
        df_jobs_paths.loc[
            df_jobs_oer_set_i.index
            ][["gdrive_path", ]]
        ], axis=1)

    columns_to_keep = [
        "bulk_id",
        "slab_id",
        "job_id",
        "facet",
        "compenv",
        "ads",
        "active_site",
        "att_num",
        "rev_num",
        "force_largest",
        "force_sum",
        "force_sum_per_atom",
        "rerun_from_oh",
        "pot_e",
        "gdrive_path",
        ]

    df_data_i = df_data_i[columns_to_keep]

    df_data_i = df_data_i.loc[:,~df_data_i.columns.duplicated()]

    # print("")
    df_data_i.to_csv(dir_path + "/data.csv")
    # __|


    for job_id_i, row_i in df_jobs_oer_set_i.iterrows():

        if job_id_i in df_atoms.index:
            # #############################################
            ads_i = row_i.ads
            att_num_i = row_i.att_num
            # #############################################
            atoms_i = df_atoms.loc[job_id_i].atoms_sorted_good
            # #############################################

            file_name_i = ads_i + "__" + job_id_i + "__" + str(att_num_i).zfill(2)

            file_dir_i = os.path.join(
                dir_path,
                "all_jobs_in_set")

            file_path_i = os.path.join(
                file_dir_i,
                file_name_i)

            if not os.path.exists(file_dir_i):
                os.makedirs(file_dir_i)

            if atoms_i is not None:
                atoms_i.write(file_path_i + ".traj")
                atoms_i.write(file_path_i + ".cif")
            else:
                with open(file_path_i + ".NONE", "w") as file:
                    file.write("Doesn't exist")
    # __|
