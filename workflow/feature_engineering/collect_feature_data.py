# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Collect feature data into master dataframe
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

from itertools import combinations
from collections import Counter
from functools import reduce

from IPython.display import display

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)
pd.options.display.max_colwidth = 100

# #########################################################
from methods import get_df_octa_vol, get_df_eff_ox
from methods import get_df_dft
from methods import get_df_job_ids
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# # Read feature dataframes

# +
df_octa_vol = get_df_octa_vol()

df_eff_ox = get_df_eff_ox()

df_dft = get_df_dft()

df_job_ids = get_df_job_ids()

# +
from local_methods import combine_dfs_with_same_cols

df_dict_i = {
    "df_eff_ox": df_eff_ox,
    "df_octa_vol": df_octa_vol,
    }

df_features = combine_dfs_with_same_cols(
    df_dict=df_dict_i,
    verbose=verbose,
    )


# -

# # Adding in bulk data

# +
def method(row_i):
    new_column_values_dict = {
        "dH_bulk": None,
        "volume_pa": None,
        "bulk_oxid_state": None,
        }


    # #####################################################
    slab_id_i = row_i.name[1]
    # #####################################################
    bulk_ids = df_job_ids[df_job_ids.slab_id == slab_id_i].bulk_id.unique()
    mess_i = "ikjisdjf"
    assert len(bulk_ids) == 1, mess_i
    bulk_id_i = bulk_ids[0]
    # #####################################################

    # #####################################################
    row_dft_i = df_dft.loc[bulk_id_i]
    # #####################################################
    dH_i = row_dft_i.dH
    volume_pa = row_dft_i.volume_pa
    stoich_i = row_dft_i.stoich
    # #####################################################

    if stoich_i == "AB2":
        bulk_oxid_state_i = +4
    elif stoich_i == "AB3":
        bulk_oxid_state_i = +6
    else:
        print("Uh oh, couldn't parse bulk stoich, not good")

    # #####################################################
    new_column_values_dict["dH_bulk"] = dH_i
    new_column_values_dict["volume_pa"] = volume_pa
    new_column_values_dict["bulk_oxid_state"] = bulk_oxid_state_i
    # #####################################################
    for key, value in new_column_values_dict.items():
        row_i[("features", key)] = value
    return(row_i)

df_features = df_features.apply(method, axis=1)
df_features = df_features.reindex(columns = ["data", "features", ], level=0)

# +
if verbose:
    print("df_features.shape:", df_features.shape)

# df_features.head()
# -

# # Save data to pickle

# +
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering")

# Pickling data ###########################################
import os; import pickle
directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(root_path_i, "out_data/df_features.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_features, fle)
# #########################################################

# #########################################################
import pickle; import os
with open(path_i, "rb") as fle:
    df_features = pickle.load(fle)
# #########################################################
# -

from methods import get_df_features
get_df_features().head()

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("collect_feature_data.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_dict = df_dict_i
# verbose = True

# + jupyter={"source_hidden": true}
# # def tmp_combine_dfs_with_same_cols_2(
# #     df_dict=None,
# #     verbose=False,
# #     ):
# """
# """
# #| - tmp_combine_dfs_with_same_cols


# #| - I
# # df_dict = {
# #     "df_eff_ox": df_eff_ox,
# #     "df_octa_vol": df_octa_vol,
# #     }

# all_data_columns = []
# for df_name_i, df_i in df_dict.items():
#     all_data_columns.extend(df_i["data"].columns.tolist())

# repeated_data_cols = []
# count_dict = dict(Counter(all_data_columns))
# for key, val in count_dict.items():
#     if val > 1:
#         repeated_data_cols.append(key)

# repated_cols_that_are_identical = []
# for col_i in repeated_data_cols:

#     if verbose:
#         print(20 * "-")
#         print("col_i:", col_i)

#     dfs_with_col = []
#     for df_name_i, df_i in df_dict.items():
#         if col_i in df_i["data"].columns:
#             temp_col_name = col_i + "__" + df_name_i
#             df_tmp = df_i.rename(columns={col_i: temp_col_name, })
#             df_tmp = df_tmp.loc[:, [("data", temp_col_name)]]
#             dfs_with_col.append(df_tmp)


#     df_one_col_comb = pd.concat(dfs_with_col, axis=1)
#     df_one_col_comb = df_one_col_comb["data"]

#     # #####################################################
#     col_pair_equal_check_list = []
#     all_col_pairs = list(combinations(df_one_col_comb.columns.tolist(), 2))
#     for col_pair_i in all_col_pairs:
#         df_one_col_comb_ij = df_one_col_comb[
#             list(col_pair_i)
#             ]
#         df_one_col_comb_ij = df_one_col_comb_ij.dropna()


#         col_vals_0 = df_one_col_comb_ij[
#             df_one_col_comb_ij.columns[0]
#             ]

#         col_vals_1 = df_one_col_comb_ij[
#             df_one_col_comb_ij.columns[1]
#             ]

#         col_comparison = (col_vals_0 == col_vals_1)

#         all_values_the_same = col_comparison.all()
#         col_pair_equal_check_list.append(all_values_the_same)

#     # #####################################################
#     all_columns_are_the_same = all(col_pair_equal_check_list)
#     if all_columns_are_the_same:
#         repated_cols_that_are_identical.append(col_i)
#         # print(
#         #     "all_columns_are_the_same:",
#         #     all_columns_are_the_same
#         #     )
#     else:
#         if verbose:
#             print(
#                 "The duplicated column ",
#                 col_i,
#                 " isn't identical across all dataframes",
#                 sep="")


# if verbose:
#     print(
#         "\n",
#         "repated_cols_that_are_identical:",
#         "\n",
#         repated_cols_that_are_identical,
#         sep="")
# #__|




















# # Renaming non-identical shared columns so that there are no duplicate column names
# non_identical_repeated_cols = []
# for col_i in repeated_data_cols:
#     if col_i not in repated_cols_that_are_identical:
#         non_identical_repeated_cols.append(col_i)


# # #########################################################
# for df_name_i, df_i in df_dict.items():

#     new_df_columns = []
#     for col_j in df_i.columns:

#         if col_j[1] in non_identical_repeated_cols:
#             col_new = col_j[1] + "__" + df_name_i
#             col_new_tuple = (col_j[0], col_new)

#             new_df_columns.append(col_new_tuple)
#         else:
#             new_df_columns.append(col_j)

#     idx = pd.MultiIndex.from_tuples(new_df_columns)
#     df_i.columns = idx

#     df_dict[df_name_i] = df_i


















# #| - Collating all data for identical columns
# collated_column_series_dict = dict()
# for col_i in repated_cols_that_are_identical:
#     dfs_with_col = []
#     for df_name_i, df_i in df_dict.items():
#         if col_i in df_i["data"].columns:
#             temp_col_name = col_i + "__" + df_name_i
#             df_tmp = df_i.rename(columns={col_i: temp_col_name, })
#             df_tmp = df_tmp.loc[:, [("data", temp_col_name)]]
#             dfs_with_col.append(df_tmp)

#     df_one_col_comb = pd.concat(dfs_with_col, axis=1)
#     df_one_col_comb = df_one_col_comb["data"]

#     dfs = []
#     for col_j in df_one_col_comb.columns:
#         dfs.append(df_one_col_comb[col_j])

#     series_i = reduce(lambda l,r: l.combine_first(r), dfs)

#     name_i = series_i.name
#     name_orig_i = name_i.split("__")[0]

#     series_i.name = name_orig_i

#     # print(series_i.shape)
#     collated_column_series_dict[col_i] = series_i
# #__|


# #| - Deleting identical columns from all dataframes
# for col_i in repated_cols_that_are_identical:

#     found_col_in_df_cnt = 0
#     for j_cnt, (df_name_j, df_j) in enumerate(df_dict.items()):

#         if col_i in df_j["data"].columns:
#             # if found_col_in_df_cnt > 0:
#             df_j_new = df_j.drop([("data", col_i)], axis=1)
#             df_dict[df_name_j] = df_j_new

#             found_col_in_df_cnt += 1
# #__|


# # Combine dataframes
# df_features = pd.concat(
#     list(df_dict.values()),
#     axis=1)


# # Adding backin the processed identical columns
# for col_i, series_i in collated_column_series_dict.items():
#     # series_i = collated_column_series_dict["from_oh"]
#     df_series_i = series_i.to_frame()
#     df_series_i.columns = pd.MultiIndex.from_tuples([("data", col_i)])

#     df_features = pd.concat([df_features, df_series_i], axis=1)

# df_features = df_features.reindex(columns = ["data", "features", ], level=0)


# # Sorting columns
# columns_list_new = []
# identical_cols = []
# for col_i in sorted(df_features.columns):
#     if col_i[1] in repated_cols_that_are_identical:
#         identical_cols.append(col_i)
#     else:
#         columns_list_new.append(col_i)

# identical_cols.extend(columns_list_new)
# df_features = df_features.reindex(identical_cols, axis=1)


# # df_features_data =
# # df_features_data.reindex(sorted(df_features_data.columns), axis=1)

# # sorted(df_features_data.columns)



# # return(df_features)
# #__|

# + jupyter={"source_hidden": true}
# dfs_with_col[1].shape

# dfs_with_col[0].shape

# df_tmp = dfs_with_col[0]

# + jupyter={"source_hidden": true}
# df_tmp.index.is_unique

# + jupyter={"source_hidden": true}
# # df_tmp[df_tmp.index.duplicated()]

# df_tmp[
#     df_tmp.index.duplicated(keep=False)
#     ]

# + jupyter={"source_hidden": true}
# df_eff_ox
# df_octa_vol

# + jupyter={"source_hidden": true}
# from local_methods import tmp_combine_dfs_with_same_cols
# from local_methods import tmp_combine_dfs_with_same_cols_1
# from local_methods import tmp_combine_dfs_with_same_cols_2

# df_features_comb = tmp_combine_dfs_with_same_cols(
# df_features_comb = tmp_combine_dfs_with_same_cols_1(
# df_features_comb = tmp_combine_dfs_with_same_cols_2(

# + jupyter={"source_hidden": true}
# print(222 * "TEMP | ")
# assert False
