"""Methods and classes for model workflow purposes.
"""

# | - Import  Modules
import os
import copy

import pickle

import numpy as np
import pandas as pd

# SciKitLearn
from sklearn.decomposition import PCA

# Catlearn
from catlearn.regression.gaussian_process import GaussianProcess
from catlearn.preprocess.clean_data import (
    clean_infinite,
    clean_variance,
    clean_skewness)
from catlearn.preprocess.scaling import standardize

from IPython.display import display
# __|

from abc import ABC, abstractmethod


class ModelAgent:
    """
    I'm going to construct this class to encapsulate all model building, running, etc.

    """

    # | - ModelAgent

class ModelWorkflow:
    """Encapsulates everything to do with running a regression workflow, including training
    """

    # | - ModelWorkflow


    # df_data=df_features_targets

    def __init__(self,
        df_data=None,

        adsorbates=None,
        ):
        """
        Parameters
        ----------
        df_data : pandas.core.frame.DataFrame
            Main dataframe continaing all training data, features, outputs, etc.
        adsorbates : list of strings
            List of possible adsorbates
            ["o", "oh", "bare", ]

        """
        # | - __init__
        # #################################################
        self.df_data__init = df_data
        self.adsorbates = adsorbates
        # #################################################
        self.percent_reduction = None
        # self.PCA_instance = None
        # self.__TEMP = None
        # #################################################

        # Process dataframe
        df_data = self._process_df_data()
        self.df_data = df_data
        #__|



    def _process_df_data(self):
        """Tue Jun  8 15:16:47 PDT 2021
        """
        # | - _process_df_data
        # #################################################
        df_data__init = self.df_data__init
        adsorbates = self.adsorbates
        # #################################################

        df_data = copy.deepcopy(df_data__init)

        # | - Create df_cols
        data_dict_list = []
        for col_i in df_data.columns:
            len_col_i = len(col_i)

            data_dict_i = dict()
            data_dict_i["col"] = col_i
            data_dict_i["len_col"] = len_col_i
            data_dict_list.append(data_dict_i)
        df_cols = pd.DataFrame(data_dict_list)

        idx = pd.MultiIndex.from_tuples(df_cols["col"].tolist())
        df_cols.index = idx
        # __|

        # #########################################################
        all_cols_are_len_3 = np.all(df_cols.len_col == 3)

        assert all_cols_are_len_3, "Come back to generalize if this isn't true"

        # TEMP
        if all_cols_are_len_3:
            tmp = 42



        #| - Generate new columns from old
        new_cols = []
        for col_i, row_i in df_cols.iterrows():

            # Parse features columns
            if col_i[0] == "features":
                col_lev_1 = col_i[0]
                if col_i[-1] == "":
                    col_lev_2 = col_i[1]
                elif col_i[1] in adsorbates:
                    col_lev_2 = col_i[2]
                else:
                    print("Not good")

                new_col_i = (col_lev_1, col_lev_2, )

            # Parse target columns
            elif col_i[0] == "targets":
                col_lev_1 = col_i[0]
                if col_i[-1] == "":
                    col_lev_2 = col_i[1]
                else:
                    print("Not good")

                new_col_i = (col_lev_1, col_lev_2, )

            # Parse other columns
            else:
                if "" in col_i:
                    col_levs = []
                    for col_lev_j in col_i:
                        if not col_lev_j == "":
                            col_levs.append(col_lev_j)
                else:
                    Print("Woops")

                new_col_i = tuple(col_levs)

                assert len(new_col_i) == 2, "Come back to this"

            new_cols.append(new_col_i)
        # __|

        idx = pd.MultiIndex.from_tuples(new_cols)


        # #########################################################
        df_data.columns = idx



        df_data_2 = df_data.dropna()

        # Percent reduction in data size from removing Nan
        percent_reduction = 100 * (df_data_2.shape[0] - df_data.shape[0]) / df_data.shape[0]
        assert percent_reduction <= 0, "percent reduction should be negative here"
        percent_reduction = np.abs(percent_reduction)




        # #################################################
        self.percent_reduction = percent_reduction
        # #################################################
        return(df_data_2)
        # #################################################
        # __|

    def _run_pca(self, num_pca=None):
        """Tue Jun  8 22:16:54 PDT 2021
        """
        # | - run_pca
        # #################################################
        df_data = self.df_data
        # #################################################

        pca_out_dict = process_pca_analysis(
            df_features_targets=df_data,
            num_pca_comp=num_pca,
            )
        pca = pca_out_dict["pca"]
        df_pca = pca_out_dict["df_pca"]

        # self.PCA_instance = pca


        return(df_pca)
        # __|


    def temp_method(self):
        """Tue Jun  8 15:16:47 PDT 2021
        """
        # | - temp_method
        tmp = 2 + 2
        # __|

    # __|



class RegressionModel_2(ABC):
    """
    """

    # | - RegressionModel_2
    print("RegressionModel_2 will eventually replace  RegressionModel_1")

    def __init__(self):
        """
        """
        # | - __init__
        # #################################################
        # self.model = None
        # #################################################

        #__|

    @property
    @abstractmethod
    def predict_wrap(self):
        """Regression method specific method to carry out prediction"""

    # @property
    # @abstractmethod
    def predict(self,
        test_features,
        test_targets=None,
        ):
        """
        """
        # | - predict
        #################################################
        predict_wrap = self.predict_wrap
        #################################################


        df_predict = predict_wrap(
            test_features,
            test_targets=test_targets,
            )

        # | - Attach actual target values if test_targets is given
        if test_targets is not None:
            test_targets.columns = ["actual"]

            df_predict = pd.concat([df_predict, test_targets], axis=1)
            df_predict["error"] = df_predict.prediction - df_predict.actual

            df_predict_cols = df_predict.columns.tolist()
            cols_to_keep_together = ["prediction", "actual", "error", ]
            for col_i in cols_to_keep_together:
                df_predict_cols.remove(col_i)

            new_cols = cols_to_keep_together + df_predict_cols
            df_predict = df_predict[new_cols]
        # __|

        return(df_predict)
        # __|


    # __|

class GP_Regression(RegressionModel_2):
    """
    """

    # | - GP_Regression

    def __init__(self,

        kernel_list=None,
        regularization=None,
        optimize_hyperparameters=None,
        scale_data=None,
        ):
        """

        """
        # | - __init__

        # #################################################
        self.kernel_list = kernel_list
        self.regularization = regularization
        self.optimize_hyperparameters = optimize_hyperparameters
        self.scale_data = scale_data
        # #################################################
        self.model = None
        # #################################################


        # Inherit all methods and properties from parent RegressionModel class
        super().__init__()
        #__|


    def run_regression(self, train_features, train_targets):
        """
        """
        # | - run_regression
        # #################################################
        kernel_list = self.kernel_list
        regularization = self.regularization
        optimize_hyperparameters = self.optimize_hyperparameters
        scale_data = self.scale_data
        # #################################################


        GP = GaussianProcess(
            kernel_list=kernel_list,
            regularization=regularization,
            train_fp=train_features,
            train_target=train_targets,
            scale_data=False,
            )

        if optimize_hyperparameters:
            GP.optimize_hyperparameters(
                global_opt=False,
                algomin='L-BFGS-B',
                eval_jac=False,
                loss_function='lml',
                # loss_function='rmse',
                )

        model = GP
        self.model = model
        # __|


    # test_features=df_data.features
    # test_targets=df_data.targets

    # def predict(self,
    def predict_wrap(self,
        test_features,
        test_targets=None,
        ):
        """
        """
        # | - predict
        # #################################################
        model = self.model
        # #################################################

        prediction = model.predict(
            test_fp=test_features,
            uncertainty=True,
            )

        # Construct dataframe of predictions
        df_predict = pd.DataFrame()
        df_predict["prediction"] = prediction["prediction"].flatten()
        df_predict["uncertainty"] = prediction["uncertainty"]
        df_predict["uncertainty_with_reg"] = prediction["uncertainty"]

        df_predict.index = test_features.index

        return(df_predict)
        # __|


    # __|

class SVR_Regression(RegressionModel_2):
    """
    """

    # | - SVR_Regression

    # def __init__(self):
    #     """
    #
    #     """
    #     # | - __init__
    #     # #################################################
    #     # self.kernel_list = kernel_list
    #     # self.regularization = regularization
    #     # self.optimize_hyperparameters = optimize_hyperparameters
    #     # self.scale_data = scale_data
    #     # #################################################
    #     # self.model = None
    #     # #################################################
    #
    #
    #     # Inherit all methods and properties from parent RegressionModel class
    #     super().__init__()
    #     #__|


    # def run_regression(self, train_features, train_targets):
    #     """
    #     """
    #     # | - run_regression
    #     # #################################################
    #
    #     # #################################################
    #
    #
    #
    #     # self.model = model
    #     # __|


    # test_features=df_data.features
    # test_targets=df_data.targets

    # def predict_wrap(self,
    #     test_features,
    #     test_targets=None,
    #     ):
    #     """
    #     """
    #     # | - predict
    #     # #################################################
    #     model = self.model
    #     # #################################################
    #
    #     prediction = model.predict(
    #         test_fp=test_features,
    #         uncertainty=True,
    #         )
    #
    #     # Construct dataframe of predictions
    #     df_predict = pd.DataFrame()
    #     df_predict["prediction"] = prediction["prediction"].flatten()
    #     df_predict["uncertainty"] = prediction["uncertainty"]
    #     df_predict["uncertainty_with_reg"] = prediction["uncertainty"]
    #
    #     df_predict.index = test_features.index
    #
    #     return(df_predict)
    #     # __|
    #

    # __|




















class RegressionModel:
    """
    Took this from PROJ_irox files
    """

    # | - RegressionModel ******************************************************
    _TEMP = "TEMP"


    def __init__(self,
        df_train=None,
        train_targets=None,
        df_test=None,

        opt_hyperparameters=True,
        gp_settings_dict=None,
        model_settings=None,
        uncertainty_type="regular",
        verbose=True,
        ):
        """

        Args:
            uncertainty_type: 'regular' or 'with_reg'
                Whether to use the regular uncertainty from the GP or the
                "regulization" corrected one
        """
        # | - __init__

        # | - Setting Argument Instance Attributes
        self.df_train = df_train
        self.train_targets = train_targets
        self.df_test = df_test
        self.opt_hyperparameters = opt_hyperparameters
        self.gp_settings_dict = gp_settings_dict
        self.model_settings = model_settings
        self.uncertainty_type = uncertainty_type
        self.verbose = verbose
        #__|

        # | - Initializing Internal Instance Attributes
        self.model = None
        #__|

        #__|

    def set_df_train(self, df_train):
        """
        """
        # | - set_df_train
        self.df_train = df_train
        #__|

    def set_train_targets(self, train_targets):
        """
        """
        # | - set_train_targets
        self.train_targets = train_targets
        #__|

    def set_df_test(self, df_test):
        """
        """
        # | - set_df_test
        self.df_test = df_test
        #__|


    def run_regression(self):
        """
        """
        # | - run_regression
        # #####################################################################
        df_train = self.df_train
        train_targets = self.train_targets
        df_test = self.df_test
        opt_hyperparameters = self.opt_hyperparameters
        gp_settings_dict = self.gp_settings_dict
        model_settings = self.model_settings
        uncertainty_type = self.uncertainty_type
        verbose = self.verbose
        # #####################################################################

        train_x = df_train
        train_y = train_targets
        # TEMP
        # print("train_y.describe():", train_y.describe())
        train_y_standard = (train_y - train_y.mean()) / train_y.std()

        # TEST PRINT TEMP
        # print("Ijfefh69w6y7")
        # print("train_y_standard.describe():", train_y_standard.describe())

        gp_model = self.gp_model_catlearn

        gp_model_out, m = gp_model(
            train_x,
            train_y_standard,
            df_predict=df_test,
            opt_hyperparameters=opt_hyperparameters,
            gp_settings_dict=gp_settings_dict,
            model_settings=model_settings,
            )


        # TEMP
        self.gp_model_out = gp_model_out
        self.gp_model = m

        if uncertainty_type == "regular":
            gp_model_out = {
                "y": gp_model_out["prediction"],
                "err": gp_model_out["uncertainty"]}
        elif uncertainty_type == "with_reg":
            gp_model_out = {
                "y": gp_model_out["prediction"],
                "err": gp_model_out["uncertainty_with_reg"]}

        model_0 = pd.DataFrame(
            gp_model_out,
            index=df_test.index)


        # | - Add column to model df that indicates the acquired points
        df_acquired = pd.DataFrame(index=df_train.index.unique())
        df_acquired["acquired"] = True

        model_i = pd.concat(
            [model_0, df_acquired],
            axis=1,
            sort=False)

        model_i = model_i.fillna(value={'acquired': False})
        #__|

        #  ####################################################################
        # Unstandardizing the output ##########################################

        y_std = train_y.std()

        if type(y_std) != float and not isinstance(y_std, np.float64):
            # print("This if is True")
            y_std = y_std.values[0]

        y_mean = train_y.mean()
        if type(y_mean) != float and not isinstance(y_mean, np.float64):
            y_mean = y_mean.values[0]

        model_i["y"] = (model_i["y"] * y_std) + y_mean
        model_i["err"] = (model_i["err"] * y_std)

        self.model = model_i
        #__|

    def gp_model_catlearn(self,
        train_features,
        train_target,
        df_predict=None,
        gp_settings_dict={},
        model_settings=None,
        opt_hyperparameters=False,
        ):
        """test_features
        """
        # | - gp_model_catlearn
        test_features = df_predict

        noise_default = 0.01  # Regularisation parameter.
        sigma_l_default = 0.8  # Length scale parameter.
        sigma_f_default = 0.2337970892240513  # Scaling parameter.
        alpha_default = 2.04987167  # Alpha parameter.

        if model_settings is None:
            model_settings = dict()

        kdict = model_settings.get("kdict", None)

        noise = gp_settings_dict.get("noise", noise_default)
        # sigma_l = gp_settings_dict.get("sigma_l", sigma_l_default)
        # sigma_f = gp_settings_dict.get("sigma_f", sigma_f_default)
        alpha = gp_settings_dict.get("alpha", alpha_default)

        # | - Jose Optimized GP
        # Define initial prediction parameters.
        #
        # noise = 0.0042  # Regularisation parameter.
        # sigma_l = 6.3917  # Length scale parameter.
        # sigma_f = 0.5120  # Scaling parameter.
        # alpha = 0.3907  # Alpha parameter.
        #
        # kdict = [
        #     {
        #         'type': 'quadratic',
        #         'dimension': 'single',
        #         # 'dimension': 'features',
        #         'slope': sigma_l,
        #         'scaling': sigma_f,
        #         'degree': alpha,
        #         }
        #     ]
        #
        # GP = GaussianProcess(
        #     kernel_list=kdict, regularization=noise, train_fp=train_features,
        #     train_target=train_target, optimize_hyperparameters=True,
        #     scale_data=False,
        #     )
        #__|


        # | - HIDE
        # noise = 0.0042  # Regularisation parameter.
        # sigma_l = 6.3917  # Length scale parameter.
        # sigma_f = 0.5120  # Scaling parameter.
        # alpha = 0.3907  # Alpha parameter.

        # noise = 0.00042  # Regularisation parameter.
        # sigma_l = 3.3917  # Length scale parameter.
        # sigma_f = 1.5120  # Scaling parameter.
        # alpha = 0.1907  # Alpha parameter.

        # noise = 0.01  # Regularisation parameter.
        # sigma_l = 0.8  # Length scale parameter.
        # sigma_f = 0.2337970892240513  # Scaling parameter.
        # alpha = 2.04987167  # Alpha parameter.
        #__|

        if kdict is None:

            kdict = [

                # | - Rational Quadratic Kernel
                # {
                #     'type': 'quadratic',
                #     'dimension': 'single',
                #     # 'dimension': 'features',
                #     'slope': sigma_l,
                #     'scaling': sigma_f,
                #     'degree': alpha,
                #     },
                #__|

                # | - Guassian Kernel (RBF)
                {
                    'type': 'gaussian',
                    'dimension': 'single',
                    # 'dimension': 'features',

                    'width': sigma_l_default,
                    'scaling': sigma_f_default,

                    # 'bounds': (
                    #     (0.0001, 10.),
                    #     (0.0001, 10.),
                    #     (0.0001, 10.),
                    #     ),

                    'scaling_bounds': ((0.0001, 10.),),

                    # 'scaling_bounds': (0.0001, 100.),
                    },

                # ORIGINAL DICT HERE
                # {
                #     'type': 'gaussian',
                #     # 'dimension': 'single',
                #     'dimension': 'features',
                #
                #     'width': sigma_l,
                #     'scaling': sigma_f,
                #
                #     'bounds': ((0.0001, 10.),),
                #     'scaling_bounds': ((0.0001, 10.),),
                #
                #     # 'scaling_bounds': (0.0001, 100.),
                #     },

                # | - __old__
                {
                    'type': 'gaussian',
                    'dimension': 'single',
                    # 'dimension': 'features',
                    'width': sigma_l_default / 10,
                    'scaling': sigma_f_default / 10,
                    'bounds': ((0.0001, 10.),),
                    'scaling_bounds': ((0.0001, 10.),),
                    },
                # __|

                #__|

                # | - Constant Kernel
                # {
                #     "type": "constant",
                #     # "operation": 0.2,
                #     # "features": ,
                #     "dimension": "single",
                #     # "dimension": "features",
                #     "const": 0.1,
                #     # "bound": ,
                #     },
                #__|

                ]


        # print("train_features:", train_features.describe())
        # print("train_target:", train_target.describe())

        #| - READ WRITE TEMP OBJ
        # import os
        # import sys
        # import pickle
        #
        # # Pickling data ###########################################
        # # out_dict = dict()
        # # out_dict["TEMP"] = None
        #
        # out_dict = dict(
        #     kdict=kdict,
        #     noise=noise,
        #     train_features=train_features,
        #     train_target=train_target,
        #     )
        #
        # import os; import pickle
        # path_i = os.path.join(
        #     os.environ["HOME"],
        #     "__temp__",
        #     "temp.pickle")
        # with open(path_i, "wb") as fle:
        #     pickle.dump(out_dict, fle)
        # # #########################################################
        #
        # # # #########################################################
        # # import pickle; import os
        # # path_i = os.path.join(
        # #     os.environ["HOME"],
        # #     "__temp__",
        # #     "temp.pickle")
        # # with open(path_i, "rb") as fle:
        # #     out_dict = pickle.load(fle)
        # # # #########################################################
        #__|

        GP = GaussianProcess(
            kernel_list=kdict, regularization=noise, train_fp=train_features,
            train_target=train_target, optimize_hyperparameters=False,
            scale_data=False,
            )


        if opt_hyperparameters:
            GP.optimize_hyperparameters(
                global_opt=False,
                # global_opt=True,

                algomin='L-BFGS-B',  # The standard one ***********************

                # | - algomin
                # algomin='Nelder-Mead',  # Seems to work well **********************
                # algomin='Newton-CG',  # Doesn't work
                # algomin='BFGS',  # Didn't work
                # algomin='CG',  # Didn't work
                # algomin='dogleg',  # Didn't work
                # algomin='Powell',  # Didn't work
                # algomin='TNC',  # Does work ***************************************
                # algomin='COBYLA',  # Didn't work
                # algomin='SLSQP  # Does work ***************************************
                # algomin='trust-constr',
                # algomin='trust-ncg',  # Didn't work
                # algomin='trust-krylov',  # Didn't work
                # algomin='trust-exact',  # Didn't work
                # algomin='',
                #__|

                eval_jac=False,
                loss_function='lml',
                # loss_function='rmse',
                )


        # TEMP
        # print("test_features.describe():", test_features.describe())

        pred = GP.predict(test_fp=test_features, uncertainty=True)

        pred["prediction"] = pred["prediction"].flatten()

        return(pred, GP)

        #__|

    #__| **********************************************************************




# df_train_features = df_train_features
# df_train_targets = df_train_targets
# df_test_features = df_test_features
# df_test_targets = df_test_targets

def run_gp_workflow(
    df_train_features=None,
    df_train_targets=None,
    df_test_features=None,
    df_test_targets=None,
    model_settings=None,
    kdict=None,
    ):
    """
    """
    # | - run_gp_workflow

    # | - __old__
    # df_j = df_features_targets_simple
    #
    #
    # # Splitting dataframe into features and targets dataframe
    # df_feat = df_j["features"]
    # df_targets = df_j["targets"]
    #
    #
    # # Standardizing features
    # df_feat = (df_feat - df_feat.mean()) / df_feat.std()
    #
    # # Renaming target column to simply `y`
    # df_targets.columns = ["y"]
    #
    #
    # if df_test is None:
    #     df_test = df_feat
    # __|


    if model_settings is None:
        # GP kernel parameters
        gp_settings = {
            "noise": 0.02542,
            "sigma_l": 2.5,
            "sigma_f": 0.8,
            "alpha": 0.2,
            }

        # gp_settings = {
        #     # "noise": 0.02542,
        #     "noise": 0.12542,
        #     "sigma_l": 3.5,
        #     "sigma_f": 1.8,
        #     "alpha": 0.3,
        #     }
    else:
        gp_settings = model_settings["gp_settings"]

    #| - Setting up Gaussian Regression model

    # Instantiate GP regression model
    RM = RegressionModel(
        df_train=df_train_features,
        train_targets=df_train_targets,

        df_test=df_test_features,

        opt_hyperparameters=True,
        gp_settings_dict=gp_settings,
        model_settings=model_settings,
        uncertainty_type='regular',
        verbose=True,
        )

    RM.run_regression()


    # Clean up model dataframe a bit
    model = RM.model
    model = model.drop(columns=["acquired"])
    model.columns = ["y_pred", "err_pred"]
    # __|

    # Combining model output and target values
    df_target_pred = pd.concat([
        df_test_targets,
        model,
        ], axis=1)

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
    out_dict["RegressionModel"] = RM
    # #####################################################
    return(out_dict)
    # #####################################################
    # __|


# df_train_features=df_train_features
# df_train_targets=df_train_targets
# df_test_features=df_test_features
# df_test_targets=df_test_targets
# model_settings=model_settings

def run_SVR_workflow(
    df_train_features=None,
    df_train_targets=None,
    df_test_features=None,
    df_test_targets=None,
    model_settings=None,
    ):
    """
    """
    # | - run_gp_workflow

    # if model_settings is None:
    #     # GP kernel parameters
    #     gp_settings = {
    #         "noise": 0.02542,
    #         "sigma_l": 2.5,
    #         "sigma_f": 0.8,
    #         "alpha": 0.2,
    #         }
    # else:
    #     gp_settings = model_settings["gp_settings"]

    #| - Setting up Gaussian Regression model

    # Instantiate GP regression model

    # RM = RegressionModel(
    #     df_train=df_train_features,
    #     train_targets=df_train_targets,

    #     df_test=df_test_features,

    #     opt_hyperparameters=True,
    #     gp_settings_dict=gp_settings,
    #     model_settings=model_settings,
    #     uncertainty_type='regular',
    #     verbose=True,
    #     )

    # RM.run_regression()


    # Clean up model dataframe a bit
    # model = RM.model
    # model = model.drop(columns=["acquired"])
    # model.columns = ["y_pred", "err_pred"]

    train_x = df_train_features.to_numpy()
    # train_y = train_targets
    train_y = df_train_targets
    # TEMP
    # print("train_y.describe():", train_y.describe())
    train_y_standard = (train_y - train_y.mean()) / train_y.std()

    from sklearn import svm

    # X = [[0, 0], [2, 2]]
    # y = [0.5, 2.5]

    # regr = svm.SVR(
    #
    #     )

    regr = svm.SVR(
        kernel='rbf',
        degree=3,
        # gamma='scale',
        gamma='auto',
        coef0=0.0,
        tol=0.001,
        C=1.0,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
        )

    regr.fit(train_x, train_y_standard["y"].to_numpy())

    predicted_y = regr.predict(
        df_test_features.to_numpy()
        )


    model_i = pd.DataFrame(
        predicted_y,
        columns=["y_pred"],
        index=df_test_features.index,
        )


    train_y = df_test_targets["y"]

    y_std = train_y.std()

    if type(y_std) != float and not isinstance(y_std, np.float64):
        # print("This if is True")
        y_std = y_std.values[0]

    y_mean = train_y.mean()
    if type(y_mean) != float and not isinstance(y_mean, np.float64):
        y_mean = y_mean.values[0]

    model_i["y_pred"] = (model_i["y_pred"] * y_std) + y_mean
    # model_i["err"] = (model_i["err"] * y_std)

    # __|

    # Combining model output and target values
    df_target_pred = pd.concat([
        # df_targets,
        df_test_targets,
        model_i,
        ], axis=1)

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
    # out_dict["RegressionModel"] = RM
    # #####################################################
    return(out_dict)
    # #####################################################
    # __|
