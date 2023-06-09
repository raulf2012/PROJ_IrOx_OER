"""
"""

# | - Import Modules

import pandas as pd

# SciKitLearn
from sklearn.decomposition import PCA

# __|


from methods_models import process_pca_analysis


# df_features_targets=df_comb
# target_ads="oh"
# feature_ads="o"

def simplify_df_features_targets(
    df_features_targets,
    target_ads=None,
    feature_ads=None,
    ):
    """Manipulates master features/targets dataframe into more managable form."""
    #| - simplify_df_features_targets
    # #####################################################
    df_i = df_features_targets
    target_ads_i = target_ads
    feature_ads_i = feature_ads
    # #####################################################


    # Parsing down dataframe into just feature columns and the target column
    cols_to_include = []

    # Getting all `features` columns for the appropriate adsorbate
    for col_i in df_i.columns.tolist():
        if col_i[0] == "features":
            if col_i[1] in ["o", "oh", "ooh", ]:
                if col_i[1] == feature_ads_i:
                    cols_to_include.append(col_i)
                else:
                    tmp = 42
            else:
                cols_to_include.append(col_i)

    # Adding single target column to list of columns
    cols_to_include.extend([
        ("targets", "g_" + target_ads_i, ""),
        ])

    col_i = ("index_str", "", "", )
    if col_i in df_i.columns:
        cols_to_include.extend([col_i])


    df_i_2 = df_i[cols_to_include]


    # Manipulating levels of column multiindex

    # Resetting target column level names, just repeating g_oh/g_o into levels 2 & 3
    columns_new = []
    for col_i in df_i_2.columns:
        if col_i == ("targets", "g_" + target_ads_i, "", ):
            columns_new.append(
                ("targets", "g_" + target_ads_i, "g_" + target_ads_i, ))
        elif col_i[0] == "features" and col_i[-1] == "":
            columns_new.append(
                ("features", col_i[1], col_i[1], ))
        else:
            columns_new.append(col_i)

    df_i_2.columns = pd.MultiIndex.from_tuples(columns_new)

    # Just dropping middle level
    df_i_2.columns = df_i_2.columns.droplevel(level=1)

    # Dropping any NaN rows, looks in either target or feature columns
    df_i_3 = df_i_2.dropna()

    return(df_i_3)
    # __|


# df_features_targets=df_j
# cols_to_use=cols_to_use
# df_format=df_format
# run_pca=True
# num_pca_comp=3
# k_fold_partition_size=100
# model_workflow=run_gp_workflow
# model_settings=model_settings

def run_kfold_cv_wf(
    df_features_targets=None,
    cols_to_use=None,
    df_format=None,

    run_pca=True,
    num_pca_comp=3,
    k_fold_partition_size=20,
    model_workflow=None,
    model_settings=None,
    # kdict=None,
    ):
    """

    df_features_targets

        Column levels are as follows:

        features                         targets
        feat_0    feat_1    feat_2...    g_oh

    """
    #| -  run_kfold_cv_wf
    # df_j = df_features_targets

    df_j = process_feature_targets_df(
        df_features_targets=df_features_targets,
        cols_to_use=cols_to_use,
        )

    df_j = df_j.dropna()

    # COMBAK New line
    # print(17 * "TEMP | ")
    df_j = df_j.dropna(axis=1)

    # print(df_j)

    if run_pca:
        # #####################################################
        # df_pca = process_pca_analysis(
        out_dict = process_pca_analysis(
            df_features_targets=df_j,
            num_pca_comp=num_pca_comp,
            )
        # #####################################################
        df_pca = out_dict["df_pca_train"]
        pca = out_dict["PCA"]
        # #####################################################



        df_data = df_pca
    else:
        df_data = df_j
        pca = None


    #| - Creating k-fold partitions
    x = df_j.index.tolist()

    partitions = []
    for i in range(0, len(x), k_fold_partition_size):
        slice_item = slice(i, i + k_fold_partition_size, 1)
        partitions.append(x[slice_item])
    #__|

    #| - Run k-fold cross-validation
    df_target_pred_parts = []
    df_target_pred_parts_2 = []
    regression_model_list = []
    for k_cnt, part_k in enumerate(range(len(partitions))):

        test_partition = partitions[part_k]

        train_partition = partitions[0:part_k] + partitions[part_k + 1:]
        train_partition = [item for sublist in train_partition for item in sublist]


        # #####################################################
        # df_test = df_pca.loc[test_partition]
        # df_train = df_pca.loc[train_partition]
        df_test = df_data.loc[test_partition]
        df_train = df_data.loc[train_partition]
        # #####################################################
        df_train_features = df_train["features"]
        df_train_targets = df_train["targets"]
        df_test_features = df_test["features"]
        df_test_targets = df_test["targets"]
        # #####################################################

        # #####################################################
        # Using the model on the test set (Normal)
        # #####################################################
        out_dict = model_workflow(
            df_train_features=df_train_features,
            df_train_targets=df_train_targets,
            df_test_features=df_test_features,
            df_test_targets=df_test_targets,
            model_settings=model_settings,
            )
        # #####################################################
        df_target_pred = out_dict["df_target_pred"]
        min_val = out_dict["min_val"]
        max_val = out_dict["max_val"]
        RM_1 = out_dict["RegressionModel"]
        # #####################################################

        regression_model_list.append(RM_1)

        # #####################################################
        # Using the model on the training set (check for bad model)
        # #####################################################
        out_dict_2 = model_workflow(
            df_train_features=df_train_features,
            df_train_targets=df_train_targets,
            df_test_features=df_train_features,
            df_test_targets=df_train_targets,
            model_settings=model_settings,
            # kdict=kdict,
            )
        # #####################################################
        df_target_pred_2 = out_dict_2["df_target_pred"]
        min_val_2 = out_dict_2["min_val"]
        max_val_2 = out_dict_2["max_val"]
        RM_2 = out_dict_2["RegressionModel"]
        # #####################################################





        df_target_pred_parts.append(df_target_pred)
        df_target_pred_parts_2.append(df_target_pred_2)

    # #########################################################
    df_target_pred_concat = pd.concat(df_target_pred_parts)
    df_target_pred = df_target_pred_concat

    # Get format column from main `df_features_targets` dataframe

    # df_features_targets["format"]["color"]["stoich"]
    # df_format = df_features_targets[("format", "color", "stoich", )]

    # df_format.name = "color"
    #
    # # Combining model output and target values
    # df_target_pred = pd.concat([
    #     df_format,
    #     df_target_pred,
    #     ], axis=1)

    df_target_pred = df_target_pred.dropna()
    # #########################################################


    # #########################################################
    df_target_pred_concat_2 = pd.concat(df_target_pred_parts_2)
    df_target_pred_2 = df_target_pred_concat_2

    # Get format column from main `df_features_targets` dataframe

    # df_features_targets["format"]["color"]["stoich"]
    # df_format = df_features_targets[("format", "color", "stoich", )]

    # df_format.name = "color"

    # # Combining model output and target values
    # df_target_pred_2 = pd.concat([
    #     df_format,
    #     df_target_pred_2,
    #     ], axis=1)

    # new_col_list = []
    # for name_i, row_i in df_target_pred_2.iterrows():
    #     color_i = df_format.loc[row_i.name]
    #     new_col_list.append(color_i)
    #
    # df_target_pred_2["color"] = new_col_list


    df_target_pred_2 = df_target_pred_2.dropna()
    df_target_pred_2 = df_target_pred_2.sort_index()
    # #########################################################

    #__|

    # Calc MAE
    MAE = df_target_pred["diff_abs"].sum() / df_target_pred["diff"].shape[0]

    MAE_2 = df_target_pred_2["diff_abs"].sum() / df_target_pred_2["diff"].shape[0]


    # Calc R2
    from sklearn.metrics import r2_score
    coefficient_of_dermination = r2_score(
        df_target_pred["y"],
        df_target_pred["y_pred"],
        )


    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["df_target_pred"] = df_target_pred
    out_dict["MAE"] = MAE
    out_dict["R2"] = coefficient_of_dermination
    out_dict["pca"] = pca
    out_dict["regression_model_list"] = regression_model_list

    out_dict["df_target_pred_on_train"] = df_target_pred_2
    out_dict["MAE_pred_on_train"] = MAE_2
    out_dict["RM_2"] = RM_2
    # #####################################################
    return(out_dict)
    # #####################################################
    # __|


def run_regression_wf(
    df_features_targets=None,
    cols_to_use=None,
    df_format=None,
    run_pca=True,
    num_pca_comp=3,
    model_workflow=None,
    model_settings=None,
    ):
    """

    df_features_targets

        Column levels are as follows:

        features                         targets
        feat_0    feat_1    feat_2...    g_oh

    """
    #| -  run_regression_wf
    df_j = process_feature_targets_df(
        df_features_targets=df_features_targets,
        cols_to_use=cols_to_use,
        )

    # # COMBAK New line
    # # print(17 * "TEMP | ")
    # df_j = df_j.dropna(axis=1)

    if run_pca:
        # #####################################################
        # df_pca = process_pca_analysis(
        out_dict = process_pca_analysis(
            df_features_targets=df_j,
            num_pca_comp=num_pca_comp,
            )

        # print("IDJFIDSIFD")
        # print(out_dict.keys())

        # #####################################################
        # df_pca = out_dict["df_pca"]
        df_pca = out_dict["df_pca_train"]
        pca = out_dict["PCA"]
        # #####################################################
        df_data = df_pca
    else:
        df_data = df_j
        pca = None


    # #####################################################
    # df_test = df_pca
    # df_train = df_pca
    df_test = df_data
    df_train = df_data
    # #####################################################
    df_train_features = df_train["features"]
    df_train_targets = df_train["targets"]
    df_test_features = df_test["features"]
    df_test_targets = df_test["targets"]
    # #####################################################


    # #####################################################
    out_dict = model_workflow(
        df_train_features=df_train_features,
        df_train_targets=df_train_targets,
        df_test_features=df_test_features,
        df_test_targets=df_test_targets,
        model_settings=model_settings,
        )
    # #####################################################
    df_target_pred = out_dict["df_target_pred"]
    min_val = out_dict["min_val"]
    max_val = out_dict["max_val"]
    # RM = out_dict["RegressionModel"]
    # #####################################################



    # df_format.name = "color"
    #
    # # Combining model output and target values
    # df_target_pred = pd.concat([
    #     df_format,
    #     df_target_pred,
    #     ], axis=1)

    df_target_pred = df_target_pred.dropna()
    # #########################################################

    # Calc MAE
    MAE = df_target_pred["diff_abs"].sum() / df_target_pred["diff"].shape[0]

    # Calc R2
    from sklearn.metrics import r2_score
    coefficient_of_dermination = r2_score(
        df_target_pred["y"],
        df_target_pred["y_pred"],
        )


    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["df_target_pred"] = df_target_pred
    out_dict["MAE"] = MAE
    out_dict["R2"] = coefficient_of_dermination
    out_dict["pca"] = pca
    # #####################################################
    return(out_dict)
    # #####################################################
    # __|

def process_feature_targets_df(
    df_features_targets=None,
    cols_to_use=None,
    ):
    """
    """
    #| - process_feature_targets_df
    df_j = df_features_targets

    #| - Controlling feature columns to use
    cols_to_keep = []
    cols_to_drop = []
    for col_i in df_j["features"].columns:
        if col_i in cols_to_use:
            cols_to_keep.append(("features", col_i))
        else:
            cols_to_drop.append(("features", col_i))

    df_j = df_j.drop(columns=cols_to_drop)
    #__|

    # | - Changing target column name to `y`
    new_cols = []
    for col_i in df_j.columns:
        tmp = 42

        if col_i[0] == "targets":
            col_new_i = ("targets", "y")
            new_cols.append(col_new_i)
        else:
            new_cols.append(col_i)

    df_j.columns = pd.MultiIndex.from_tuples(new_cols)
    #__|

    # Splitting dataframe into features and targets dataframe
    df_feat = df_j["features"]


    # Standardizing features
    df_feat = (df_feat - df_feat.mean()) / df_feat.std()
    df_j["features"] = df_feat


    # # #####################################################
    # df_feat_pca = pca_analysis(
    #     df_j["features"],
    #     pca_mode="num_comp",  # 'num_comp' or 'perc'
    #     pca_comp=num_pca_comp,
    #     verbose=False,
    #     )
    #
    # cols_new = []
    # for col_i in df_feat_pca.columns:
    #     col_new_i = ("features", col_i)
    #     cols_new.append(col_new_i)
    # df_feat_pca.columns = pd.MultiIndex.from_tuples(cols_new)
    #
    # df_pca = pd.concat([
    #     df_feat_pca,
    #     df_j[["targets"]],
    #     ], axis=1)

    return(df_j)
    #__|































# df_features_targets=df_j
# cols_to_use=df_j["features"].columns.tolist()
# # df_format=df_format
# run_pca=True
# num_pca_comp=3
# k_fold_partition_size=100
# # model_workflow=run_SVR_workflow
# model_workflow=run_gp_workflow
# # model_settings=None
# model_settings=dict(
#     gp_settings=gp_settings,
#     kdict=kdict,
#     )



def run_kfold_cv_wf__TEMP(
    df_features_targets=None,
    cols_to_use=None,
    run_pca=True,
    num_pca_comp=3,
    k_fold_partition_size=20,
    model_workflow=None,
    model_settings=None,
    ):
    """

    df_features_targets

        Column levels are as follows:

        features                         targets
        feat_0    feat_1    feat_2...    g_oh

    """
    #| -  run_kfold_cv_wf
    # df_j = df_features_targets

    df_j = process_feature_targets_df(
        df_features_targets=df_features_targets,
        cols_to_use=cols_to_use,
        )

    df_j = df_j.dropna()

    # COMBAK New line
    # print(17 * "TEMP | ")
    df_j = df_j.dropna(axis=1)

    # print(df_j)

    if run_pca:
        # #####################################################
        # df_pca = process_pca_analysis(
        out_dict = process_pca_analysis(
            df_features_targets=df_j,
            num_pca_comp=num_pca_comp,
            )
        # #####################################################

        # print("kisjdfijsdf8934yt89eruoidsff8s")
        # print(out_dict.keys())

        df_pca = out_dict["df_pca_train"]
        pca = out_dict["PCA"]
        # #####################################################

        df_data = df_pca
    else:
        df_data = df_j
        pca = None


    #| - Creating k-fold partitions
    x = df_j.index.tolist()

    partitions = []
    for i in range(0, len(x), k_fold_partition_size):
        slice_item = slice(i, i + k_fold_partition_size, 1)
        partitions.append(x[slice_item])
    #__|

    #| - Run k-fold cross-validation
    df_target_pred_parts = []
    df_target_pred_parts_2 = []
    regression_model_list = []
    for k_cnt, part_k in enumerate(range(len(partitions))):

        test_partition = partitions[part_k]

        train_partition = partitions[0:part_k] + partitions[part_k + 1:]
        train_partition = [item for sublist in train_partition for item in sublist]


        # #####################################################
        # df_test = df_pca.loc[test_partition]
        # df_train = df_pca.loc[train_partition]
        df_test = df_data.loc[test_partition]
        df_train = df_data.loc[train_partition]
        # #####################################################
        df_train_features = df_train["features"]
        df_train_targets = df_train["targets"]
        df_test_features = df_test["features"]
        df_test_targets = df_test["targets"]
        # #####################################################

        # #####################################################
        # Using the model on the test set (Normal)
        # #####################################################
        out_dict = model_workflow(
            df_train_features=df_train_features,
            df_train_targets=df_train_targets,
            df_test_features=df_test_features,
            df_test_targets=df_test_targets,
            model_settings=model_settings,
            )
        # #####################################################
        df_target_pred = out_dict["df_target_pred"]
        min_val = out_dict["min_val"]
        max_val = out_dict["max_val"]
        # RM_1 = out_dict["RegressionModel"]
        # #####################################################

        # regression_model_list.append(RM_1)

        # #####################################################
        # Using the model on the training set (check for bad model)
        # #####################################################
        out_dict_2 = model_workflow(
            df_train_features=df_train_features,
            df_train_targets=df_train_targets,
            df_test_features=df_train_features,
            df_test_targets=df_train_targets,
            model_settings=model_settings,
            # kdict=kdict,
            )
        # #####################################################
        df_target_pred_2 = out_dict_2["df_target_pred"]
        min_val_2 = out_dict_2["min_val"]
        max_val_2 = out_dict_2["max_val"]
        # RM_2 = out_dict_2["RegressionModel"]
        # #####################################################





        df_target_pred_parts.append(df_target_pred)
        df_target_pred_parts_2.append(df_target_pred_2)

    # #########################################################
    df_target_pred_concat = pd.concat(df_target_pred_parts)
    df_target_pred = df_target_pred_concat

    # Get format column from main `df_features_targets` dataframe

    # df_features_targets["format"]["color"]["stoich"]
    # df_format = df_features_targets[("format", "color", "stoich", )]

    # df_format.name = "color"

    # # Combining model output and target values
    # df_target_pred = pd.concat([
    #     df_format,
    #     df_target_pred,
    #     ], axis=1)

    df_target_pred = df_target_pred.dropna()
    # #########################################################


    # #########################################################
    df_target_pred_concat_2 = pd.concat(df_target_pred_parts_2)
    df_target_pred_2 = df_target_pred_concat_2

    # Get format column from main `df_features_targets` dataframe

    # df_features_targets["format"]["color"]["stoich"]
    # df_format = df_features_targets[("format", "color", "stoich", )]

    # df_format.name = "color"

    # # # Combining model output and target values
    # # df_target_pred_2 = pd.concat([
    # #     df_format,
    # #     df_target_pred_2,
    # #     ], axis=1)

    # new_col_list = []
    # for name_i, row_i in df_target_pred_2.iterrows():
    #     color_i = df_format.loc[row_i.name]
    #     new_col_list.append(color_i)

    # df_target_pred_2["color"] = new_col_list


    df_target_pred_2 = df_target_pred_2.dropna()
    df_target_pred_2 = df_target_pred_2.sort_index()
    # #########################################################

    #__|

    # Calc MAE
    MAE = df_target_pred["diff_abs"].sum() / df_target_pred["diff"].shape[0]

    MAE_2 = df_target_pred_2["diff_abs"].sum() / df_target_pred_2["diff"].shape[0]


    # Calc R2
    from sklearn.metrics import r2_score
    coefficient_of_dermination = r2_score(
        df_target_pred["y"],
        df_target_pred["y_pred"],
        )


    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["df_target_pred"] = df_target_pred
    out_dict["MAE"] = MAE
    out_dict["R2"] = coefficient_of_dermination
    out_dict["pca"] = pca
    out_dict["regression_model_list"] = regression_model_list

    out_dict["df_target_pred_on_train"] = df_target_pred_2
    out_dict["MAE_pred_on_train"] = MAE_2
    # out_dict["RM_2"] = RM_2
    # #####################################################
    return(out_dict)
    # #####################################################
    # __|
