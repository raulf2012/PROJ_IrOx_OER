
#| - Import Modules
import os
import sys

# # #########################################################
from methods import (
    get_df_jobs_anal,
    )
#__|



def get_systems_to_run_bare_and_oh(df_jobs_anal):
    """
    Takes df_jobs_anal and filter to:
      * only *O slabs
      * slabs that have 'NaN' in the active site (not *O that are run from  *OH, which have an active site value)
      * Only completed slabs
      * Only the first att_num, so that you don't start new sets of *OH and bare jobs from rerun *O jobs
    """
    #| - get_systems_to_run_bare_and_oh
    # df_jobs_anal = get_df_jobs_anal()


    df_jobs_anal_i = df_jobs_anal

    var = "o"
    df_jobs_anal_i = df_jobs_anal_i.query('ads == @var')

    var = "NaN"
    df_jobs_anal_i = df_jobs_anal_i.query('active_site == @var')

    df_jobs_anal_i = df_jobs_anal_i[df_jobs_anal_i.job_completely_done == True]




    # #########################################################
    indices_to_remove = []
    # #########################################################
    group_cols = ["compenv", "slab_id", "ads", ]
    grouped = df_jobs_anal_i.groupby(group_cols)
    for name, group in grouped:

        num_rows = group.shape[0]
        if num_rows > 1:
            # print(name)
            # print("")
            # # print(num_rows)
            # print("COMBAK CHECK THIS")
            # print("This was made when there was only 1 *O calc, make sure it's not creating new *OH jobs after running more *O calcs")

            group_index = group.index.to_frame()
            group_index_i = group_index[group_index.att_num != 1]

            indices_to_remove.extend(
                group_index_i.index.tolist()
                )

    df_jobs_anal_i = df_jobs_anal_i.drop(index=indices_to_remove)

    indices_out = df_jobs_anal_i.index.tolist()

    return(indices_out)
    #__|
