"""
"""

#| - Import Modules
import os
import sys

from itertools import combinations
from collections import Counter
from functools import reduce

import pandas as pd
#__|



# df_dict = df_dict_i
# verbose=False

def combine_dfs_with_same_cols(
    df_dict=None,
    verbose=False,
    ):
    """
    """
    #| - combine_dfs_with_same_cols


    #| - I
    # df_dict = {
    #     "df_eff_ox": df_eff_ox,
    #     "df_octa_vol": df_octa_vol,
    #     }

    all_data_columns = []
    for df_name_i, df_i in df_dict.items():
        all_data_columns.extend(df_i["data"].columns.tolist())

    repeated_data_cols = []
    count_dict = dict(Counter(all_data_columns))
    for key, val in count_dict.items():
        if val > 1:
            repeated_data_cols.append(key)

    repated_cols_that_are_identical = []
    for col_i in repeated_data_cols:

        if verbose:
            print(20 * "-")
            print("col_i:", col_i)

        dfs_with_col = []
        for df_name_i, df_i in df_dict.items():
            if col_i in df_i["data"].columns:
                temp_col_name = col_i + "__" + df_name_i
                df_tmp = df_i.rename(columns={col_i: temp_col_name, })
                df_tmp = df_tmp.loc[:, [("data", temp_col_name)]]
                dfs_with_col.append(df_tmp)


        df_one_col_comb = pd.concat(dfs_with_col, axis=1)
        df_one_col_comb = df_one_col_comb["data"]

        # #####################################################
        col_pair_equal_check_list = []
        all_col_pairs = list(combinations(df_one_col_comb.columns.tolist(), 2))
        for col_pair_i in all_col_pairs:
            df_one_col_comb_ij = df_one_col_comb[
                list(col_pair_i)
                ]
            df_one_col_comb_ij = df_one_col_comb_ij.dropna()


            col_vals_0 = df_one_col_comb_ij[
                df_one_col_comb_ij.columns[0]
                ]

            col_vals_1 = df_one_col_comb_ij[
                df_one_col_comb_ij.columns[1]
                ]

            col_comparison = (col_vals_0 == col_vals_1)

            all_values_the_same = col_comparison.all()
            col_pair_equal_check_list.append(all_values_the_same)

        # #####################################################
        all_columns_are_the_same = all(col_pair_equal_check_list)
        if all_columns_are_the_same:
            repated_cols_that_are_identical.append(col_i)
            # print(
            #     "all_columns_are_the_same:",
            #     all_columns_are_the_same
            #     )
        else:
            if verbose:
                print(
                    "The duplicated column ",
                    col_i,
                    " isn't identical across all dataframes",
                    sep="")


    if verbose:
        print(
            "\n",
            "repated_cols_that_are_identical:",
            "\n",
            repated_cols_that_are_identical,
            sep="")
    #__|




















    # Renaming non-identical shared columns so that there are no duplicate column names
    non_identical_repeated_cols = []
    for col_i in repeated_data_cols:
        if col_i not in repated_cols_that_are_identical:
            non_identical_repeated_cols.append(col_i)


    # #########################################################
    for df_name_i, df_i in df_dict.items():

        new_df_columns = []
        for col_j in df_i.columns:

            if col_j[1] in non_identical_repeated_cols:
                col_new = col_j[1] + "__" + df_name_i
                col_new_tuple = (col_j[0], col_new)

                new_df_columns.append(col_new_tuple)
            else:
                new_df_columns.append(col_j)

        idx = pd.MultiIndex.from_tuples(new_df_columns)
        df_i.columns = idx

        df_dict[df_name_i] = df_i


















    #| - Collating all data for identical columns
    collated_column_series_dict = dict()
    for col_i in repated_cols_that_are_identical:
        dfs_with_col = []
        for df_name_i, df_i in df_dict.items():
            if col_i in df_i["data"].columns:
                temp_col_name = col_i + "__" + df_name_i
                df_tmp = df_i.rename(columns={col_i: temp_col_name, })
                df_tmp = df_tmp.loc[:, [("data", temp_col_name)]]
                dfs_with_col.append(df_tmp)

        df_one_col_comb = pd.concat(dfs_with_col, axis=1)
        df_one_col_comb = df_one_col_comb["data"]

        dfs = []
        for col_j in df_one_col_comb.columns:
            dfs.append(df_one_col_comb[col_j])

        series_i = reduce(lambda l,r: l.combine_first(r), dfs)

        name_i = series_i.name
        name_orig_i = name_i.split("__")[0]

        series_i.name = name_orig_i

        # print(series_i.shape)
        collated_column_series_dict[col_i] = series_i
    #__|


    #| - Deleting identical columns from all dataframes
    for col_i in repated_cols_that_are_identical:

        found_col_in_df_cnt = 0
        for j_cnt, (df_name_j, df_j) in enumerate(df_dict.items()):

            if col_i in df_j["data"].columns:
                # if found_col_in_df_cnt > 0:
                df_j_new = df_j.drop([("data", col_i)], axis=1)
                df_dict[df_name_j] = df_j_new

                found_col_in_df_cnt += 1
    #__|


    # Combine dataframes
    df_features = pd.concat(
        list(df_dict.values()),
        axis=1)


    # Adding backin the processed identical columns
    for col_i, series_i in collated_column_series_dict.items():
        # series_i = collated_column_series_dict["from_oh"]
        df_series_i = series_i.to_frame()
        df_series_i.columns = pd.MultiIndex.from_tuples([("data", col_i)])

        df_features = pd.concat([df_features, df_series_i], axis=1)

    df_features = df_features.reindex(columns = ["data", "features", ], level=0)


    # Sorting columns
    columns_list_new = []
    identical_cols = []
    for col_i in sorted(df_features.columns):
        if col_i[1] in repated_cols_that_are_identical:
            identical_cols.append(col_i)
        else:
            columns_list_new.append(col_i)

    identical_cols.extend(columns_list_new)
    df_features = df_features.reindex(identical_cols, axis=1)


    # df_features_data =
    # df_features_data.reindex(sorted(df_features_data.columns), axis=1)

    # sorted(df_features_data.columns)



    return(df_features)
    #__|
