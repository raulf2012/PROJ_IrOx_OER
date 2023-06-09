"""
"""

#| - Import Modules
import os
import sys

import numpy as np
import pandas as pd

# #########################################################
from methods_models import ModelAgent
#__|

def model_workflow_wrap(
    df_data=None,
    RegressionInstance=None,
    Regression_class=None,

    init_pca_comp=None,
    do_every_nth_pca_comp=None,
    k_fold_partition_size=None,

    adsorbates=None,
    verbose=None,
    ):
    """
    """
    #| - model_workflow_wrap
    data_dict_list = []
    num_feat_cols = df_data.features.shape[1]
    for num_pca_i in range(init_pca_comp, num_feat_cols + 1, do_every_nth_pca_comp):

        # if verbose:
        #     print("")
        #     print(40 * "*")
        #     print(num_pca_i)

        MA = ModelAgent(
            df_features_targets=df_data,
            Regression=RegressionInstance,
            Regression_class=Regression_class,
            use_pca=True,
            num_pca=num_pca_i,
            adsorbates=adsorbates,
            stand_targets=False,  # True was giving much worse errors, keep False
            )

        MA.run_kfold_cv_workflow(
            k_fold_partition_size=k_fold_partition_size,
            )

        # if MA.can_run:
        #     if verbose:
        #         print("MAE:", np.round(MA.mae, 4))
        #         print("MA.r2:", np.round(MA.r2, 4))
        #         print("MAE (in_fold):", np.round(MA.mae_infold, 4))

        data_dict_i = dict()
        data_dict_i["num_pca"] = num_pca_i
        data_dict_i["MAE"] = MA.mae
        data_dict_i["ModelAgent"] = MA
        data_dict_list.append(data_dict_i)

    df_models = pd.DataFrame(data_dict_list)
    df_models = df_models.set_index("num_pca")




    # #########################################################
    # Finding best performing model
    row_models_i = df_models.sort_values("MAE").iloc[0]

    opt_pca_comp = row_models_i.name

    MA_best = row_models_i.ModelAgent

    if verbose:
        print(4 * "\n")

        print(
            row_models_i.name,
            " PCA components are ideal with an MAE of ",
            np.round(
            row_models_i.MAE,
                4),
            sep="")


        print(
            "Num features: ",
            len(df_data.features.columns.tolist()),
            "\n",
            sep="")
























    # #########################################################
    adsorbates_features = []
    other_features = []
    for i in df_data.features.columns.tolist():
        if i[0] in adsorbates:
            adsorbates_features.append(i[1])
        else:
            other_features.append(i[0])

    if verbose:

        print(
            "Adsorbates features: ",
            ", ".join(
                sorted(adsorbates_features, key = lambda s: s.casefold())
                ),
            sep="")

        print(
            "O features: ",
            ", ".join(list(np.sort(other_features))),
            sep="")



    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["opt_pca_comp"] = opt_pca_comp
    out_dict["MAE"] = row_models_i.MAE
    out_dict["features"] = adsorbates_features + other_features
    out_dict["adsorbates_features"] = adsorbates_features
    out_dict["other_features"] = other_features
    # #####################################################
    return(out_dict)
    # #####################################################
    #__|
