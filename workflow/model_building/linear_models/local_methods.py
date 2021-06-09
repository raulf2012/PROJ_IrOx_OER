"""
"""

# | - Import Modules
import numpy as np

from sklearn.linear_model import LinearRegression
# __|


def run_linear_workflow(
    df_train_features=None,
    df_train_targets=None,
    df_test_features=None,
    df_test_targets=None,

    model_settings=None,  # Not needed for this model, just adding for consistency
    ):
    """
    """
    #| - run_linear_workflow

    X_pred = df_test_features.to_numpy()
    X_pred = X_pred.reshape(-1, X_pred.shape[1])

    y_pred = df_test_targets["y"].tolist()


    X = df_train_features.to_numpy()
    X = X.reshape(-1, X.shape[1])

    y = df_train_targets["y"].tolist()

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X_pred)

    df_test_targets["y_pred"] = y_pred

    df_target_pred = df_test_targets



    # Drop NaN rows again
    df_target_pred = df_target_pred.dropna()


    # Create columns for y_pred - y and |y_pred - y|
    df_target_pred["diff"] = df_target_pred["y_pred"] - df_target_pred["y"]
    df_target_pred["diff_abs"] = np.abs(df_target_pred["diff"])

    # Get global min/max values (min/max over targets and predictions)
    df_target_pred_i = df_target_pred[["y", "y_pred"]]

    max_val = df_target_pred_i.max().max()
    min_val = df_target_pred_i.min().min()

    max_min_diff = max_val - min_val


    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["df_target_pred"] = df_target_pred
    out_dict["min_val"] = min_val
    out_dict["max_val"] = max_val
    out_dict["RegressionModel"] = model
    # #####################################################
    return(out_dict)
    # #####################################################
    #__|
