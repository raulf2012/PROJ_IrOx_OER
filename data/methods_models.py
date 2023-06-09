"""Methods and classes for model workflow purposes.
"""

# | - Import  Modules
import os
import copy

import pickle
import random

import numpy as np
# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd

# SciKitLearn
from sklearn.decomposition import PCA

from abc import ABC, abstractmethod

# Catlearn
from catlearn.regression.gaussian_process import GaussianProcess
from catlearn.preprocess.clean_data import (
    clean_infinite,
    clean_variance,
    clean_skewness)
from catlearn.preprocess.scaling import standardize

from IPython.display import display
# __|

from sklearn.metrics import r2_score
from sklearn import svm

from sklearn.linear_model import LinearRegression

import plotly.graph_objs as go



class ModelAgent_Plotter:
    """Plotting class for ModelAgent regression workflows.
    """

    # | - ModelAgent_Plotter

    def __init__(self,
        ModelAgent=None,
        layout_shared=None,
        ):
        """
        Parameters
        ----------
        ModelAgent : ModelAgent instance
            Instance of ModelAgent class
        """
        # | - __init__
        # #################################################
        self.ModelAgent = ModelAgent
        self.layout_shared = layout_shared
        # #################################################

        # #################################################

        # __|


    def plot_residuals(self):
        """
        """
        #| - plot_residuals
        # #################################################
        MA = self.ModelAgent
        layout_shared = self.layout_shared
        # #################################################
        df_predict = MA.df_predict
        # #################################################


        # | - layout
        layout = go.Layout(

            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    font=None,
                    text="Systems",
                    ),
                ),

            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    font=None,
                    text="Residuals (eV)",
                    ),
                ),

            )
        # __|


        layout.update(dict1=layout_shared)

        df_predict["error_abs"] = df_predict.error.abs()


        trace = go.Scatter(
            y=df_predict.sort_values("error_abs", ascending=False).error_abs
            )
        data = [trace]

        fig = go.Figure(data=data, layout=layout)
        # fig.show()


        # #################################################
        self.plot_residuals__PLT = fig
        # #################################################
        # __|

    def _plot_parity(self, df_predict):
        """
        """
        # | - plot_parity_infold
        # #################################################
        MA = self.ModelAgent
        layout_shared = self.layout_shared
        # #################################################
        # df_predict = MA.RW_infold.df_predict
        target_ads = MA.target_ads
        # #################################################


        max_val = df_predict[["prediction", "actual"]].max().max()
        min_val = df_predict[["prediction", "actual"]].min().min()

        dd = 0.1

        trace_parity = go.Scatter(
            y=[min_val - 2 * dd, max_val + 2 * dd],
            x=[min_val - 2 * dd, max_val + 2 * dd],
            mode="lines",
            name="Parity line",
            line_color="black",
            )

        trace_i = go.Scatter(
            y=df_predict["actual"],
            x=df_predict["prediction"],
            mode="markers",
            name="CV Regression",
            # opacity=0.8,
            opacity=1.,
            marker=dict(
                # color=df_predict["color"],
                # **scatter_marker_props.to_plotly_json(),
                ),
            )


        # #################################################
        layout_mine = go.Layout(

            showlegend=True,

            yaxis=go.layout.YAxis(
                range=[min_val - dd, max_val + dd],
                title=dict(
                    text="Simulated ΔG<sub>*{}</sub>".format(target_ads.upper()),
                    ),
                ),

            xaxis=go.layout.XAxis(
                range=[min_val - dd, max_val + dd],
                title=dict(
                    text="Predicted ΔG<sub>*{}</sub>".format(target_ads.upper()),
                    ),
                ),

            )


        # #########################################################
        layout_shared_i = copy.deepcopy(layout_shared)
        layout_shared_i = layout_shared_i.update(layout_mine)

        # data = [trace_parity, trace_i, trace_j]
        data = [trace_parity, trace_i, ]

        fig = go.Figure(data=data, layout=layout_shared_i)

        # fig.show()

        return(fig)

        # self.plot_parity_infold__PLT = fig
        # __|

    def plot_parity(self):
        """
        """
        # | - plot_parity_infold
        # #################################################
        MA = self.ModelAgent
        # #################################################
        df_predict = MA.df_predict
        # #################################################
        _plot_parity = self._plot_parity
        # #################################################


        fig = _plot_parity(df_predict)

        self.plot_parity__PLT = fig
        # __|

    def plot_parity_infold(self):
        """
        """
        # | - plot_parity_infold
        # #################################################
        MA = self.ModelAgent
        # layout_shared = self.layout_shared
        # #################################################
        df_predict = MA.RW_infold.df_predict
        # target_ads = MA.target_ads
        # #################################################
        _plot_parity = self._plot_parity
        # #################################################


        fig = _plot_parity(df_predict)

        self.plot_parity_infold__PLT = fig
        # __|

    #__|

class ModelAgent:
    """I'm going to construct this class to encapsulate all model building, running, etc.
    """

    # | - ModelAgent **************************************

    def __init__(self,
        df_features_targets=None,
        Regression=None,
        Regression_class=None,

        use_pca=False,
        num_pca=None,
        adsorbates=None,
        stand_targets=False,
        ):
        """
        Parameters
        ----------
        df_features_targets : pandas.core.frame.DataFrame
            Main dataframe continaing all training data, features, outputs, etc.
        Regression : RegressionModel (GP_Regression, SVR_Regression, etc.)
            Regression model class, will take care of training and prediction
        Regression_class : Uninstantiated RegressionModel (GP_Regression, SVR_Regression, etc.)
            Regression model class object, not instantiated, will be used to instantiate different instances of the Regression class for various purposes
        use_pca : boolean
            Whether to use PCA decomposition on features
        num_pca : integer
            Number of PCA components
        adsorbates : list of strings
            List of possible adsorbates
            ["o", "oh", "bare", ]
        stand_targets : boolean
            Whether to standardize targets for regression (subtract mean, normalize by std. dev.)
        """
        #| - __init__
        # #################################################
        self._df_features_targets__init = df_features_targets
        self.Regression = Regression
        self.Regression_class = Regression_class
        self._use_pca = use_pca
        self.num_pca = num_pca
        self._adsorbates = adsorbates
        self._stand_targets = stand_targets
        # #################################################
        self.percent_reduction = None
        self.cv_data = None
        self.RW_infold = None
        self.mae_infold = None
        self.mae = None
        self.r2 = None
        self.target_ads = None
        self.PCA_infold = None
        self.df_pca_comp = None
        self.can_run = None
        # #################################################


        # print(111 * "TEMP | ")

        # Process dataframe
        df_features_targets = self._process_df_data()
        self.df_features_targets = df_features_targets

        self._get_target_col_ads()

        self._check_if_can_run()
        # __|


    def _check_if_can_run(self):
        """Checking whether there are enough feature columns for PCA analysis
        """
        # | - _check_if_can_run
        # #################################################
        df_features_targets = self.df_features_targets
        num_pca = self.num_pca
        use_pca = self._use_pca
        # #################################################

        can_run = True
        if df_features_targets.features.shape[1] < num_pca:
            if use_pca:
                can_run = False


        # #################################################
        self.can_run = can_run
        # #################################################
        # __|

    def _process_df_data(self):
        """Tue Jun  8 15:16:47 PDT 2021
        """
        # | - _process_df_data
        # #################################################
        df_features_targets__init = self._df_features_targets__init
        adsorbates = self._adsorbates
        # #################################################

        df_data = copy.deepcopy(df_features_targets__init)

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
                    print("Woops")

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
        # Remove columns that have 0 variance
        df_data_3 = copy.deepcopy(df_data_2)

        for col_i in df_data_2.features.columns:
            std_i = df_data_3.features[col_i].std()

            if std_i == 0:
                print('Removing column "{0}" because it has a standard deviation of 0'.format(col_i))
                df_data_3 = df_data_3.drop(columns=[("features", col_i, )])


        # #################################################
        self.percent_reduction = percent_reduction
        # #################################################
        return(df_data_3)
        # #################################################
        # __|


    def _run_pca(self,
        df_data,
        df_test=None,
        num_pca=None,
        ):
        """Tue Jun  8 22:16:54 PDT 2021
        """
        # | - run_pca
        pca_out_dict = process_pca_analysis(
            df_features_targets=df_data,
            df_test=df_test,
            num_pca_comp=num_pca,
            )
        PCA = pca_out_dict["PCA"]
        df_pca_train = pca_out_dict["df_pca_train"]
        df_pca_test = pca_out_dict["df_pca_test"]


        return(df_pca_train, df_pca_test, PCA)
        # __|


    # self = MA
    # k_fold_partition_size=100
    # from methods_models import RegressionWorkflow

    def _run_kfold_cv_workflow__run_infold(self):
        """Thu Jun 10 16:03:57 PDT 2021
        """
        # | - _run_kfold_cv_workflow__run_infold
        # #################################################
        df_features_targets = self.df_features_targets
        Regression = self.Regression
        Regression_class = self.Regression_class
        use_pca = self._use_pca
        num_pca = self.num_pca
        stand_targets = self._stand_targets
        # #################################################
        _standardize_train_test = self._standardize_train_test
        # #################################################
        init_params = Regression.init_params
        # #################################################


        df_data = df_features_targets

        df_train = df_data
        df_test = df_data

        df_train_std, df_test_std = \
            _standardize_train_test(
                df_train,
                df_test=df_test,
                stand_targets=stand_targets,
                )
        df_train_final = df_train_std
        df_test_final = df_test_std


        if use_pca:
            df_pca_train, df_pca_test, PCA = \
                self._run_pca(df_train, df_test=df_test, num_pca=num_pca)
            df_train_final = df_pca_train
            df_test_final = df_pca_test

        # #############################################
        # Running regression workflow
        RC = Regression_class(**init_params)

        RW_infold = RegressionWorkflow(
            df_data=df_train_final,
            Regression=RC,
            )
        RW_infold.run_Regression()

        RW_infold.predict(df_test_final.features, df_test_final.targets)

        df_predict = RW_infold.df_predict
        mae_infold = df_predict.error.abs().mean()


        # #################################################
        self.RW_infold = RW_infold
        self.mae_infold = mae_infold
        self.PCA_infold = PCA
        # #################################################
        # __|


    # import random
    #
    # self = MA
    # k_fold_partition_size=100
    # from methods_models import RegressionWorkflow

    def _run_kfold_cv_workflow__get_cv_data(self,
        k_fold_partition_size=None,
        ):
        """Wed Jun  9 20:48:08 PDT 2021
        """
        # | - _run_kfold_cv_workflow__get_cv_data
        # #################################################
        df_features_targets = self.df_features_targets
        Regression = self.Regression
        Regression_class = self.Regression_class
        use_pca = self._use_pca
        num_pca = self.num_pca
        stand_targets = self._stand_targets
        # #################################################
        _standardize_train_test = self._standardize_train_test
        # #################################################
        init_params = Regression.init_params
        # #################################################


        df_data = df_features_targets

        df_train = df_data
        df_test = df_data

        df_train_std, df_test_std = \
            _standardize_train_test(
                df_train,
                df_test=df_test,
                stand_targets=stand_targets,
                )
        df_train_final = df_train_std
        df_test_final = df_test_std


        if use_pca:
            df_pca_train, df_pca_test, PCA = \
                self._run_pca(df_train, df_test=df_test, num_pca=num_pca)
            df_train_final = df_pca_train
            df_test_final = df_pca_test

        # #############################################
        # Running regression workflow
        RC = Regression_class(**init_params)

        RW_infold = RegressionWorkflow(
            df_data=df_train_final,
            Regression=RC,
            )
        RW_infold.run_Regression()

        RW_infold.predict(df_test_final.features, df_test_final.targets)

        df_predict = RW_infold.df_predict
        mae_infold = df_predict.error.abs().mean()


        #| - Creating k-fold partitions
        indices = df_data.index.tolist()
        random.shuffle(indices)

        partitions = []
        for i in range(0, len(indices), k_fold_partition_size):
            slice_item = slice(i, i + k_fold_partition_size, 1)
            partitions.append(indices[slice_item])
        #__|

        #| - Run k-fold cross-validation
        cv_data = dict()
        for part_i, test_partition in enumerate(partitions):
            train_partition = partitions[0:part_i] + partitions[part_i + 1:]
            train_partition = [item for sublist in train_partition for item in sublist]

            df_test = df_data.loc[test_partition]
            df_train = df_data.loc[train_partition]

            df_train_std, df_test_std = \
                _standardize_train_test(
                    df_train,
                    df_test=df_test,
                    stand_targets=stand_targets,
                    )
            df_train_final = df_train_std
            df_test_final = df_test_std


            if use_pca:
                df_pca_train, df_pca_test, PCA = \
                    self._run_pca(df_train, df_test=df_test, num_pca=num_pca)
                df_train_final = df_pca_train
                df_test_final = df_pca_test


            # #############################################
            # Running regression workflow
            RC = Regression_class(**init_params)

            RW = RegressionWorkflow(
                df_data=df_train_final,
                Regression=RC,
                )
            RW.run_Regression()

            RW.predict(df_test_final.features, df_test_final.targets)



            # #############################################
            data_dict_i = dict()
            # #############################################
            data_dict_i["RegressionWorkflow"] = RW
            data_dict_i["df_train"] = df_train_final
            data_dict_i["df_test"] = df_test_final
            # #############################################
            cv_data[part_i] = data_dict_i
            # #############################################
            # __|


        # #################################################
        self.cv_data = cv_data
        # #################################################
        # __|


    def _run_kfold_cv_workflow__process_df_predict(self,
        ):
        """Wed Jun  9 20:56:08 PDT 2021
        """
        # | - _run_kfold_cv_workflow__get_cv_data
        # #################################################
        cv_data = self.cv_data
        # #################################################


        df_predict_list = []
        for ind_i, cv_data_i in cv_data.items():
            RW = cv_data_i["RegressionWorkflow"]

            df_predict_i = RW.df_predict
            df_predict_list.append(df_predict_i)

        df_predict_comb = pd.concat(df_predict_list)

        mae = np.abs(df_predict_comb.error).mean()



        # Calculate R2 metric
        r2 = r2_score(
            df_predict_comb.actual,
            df_predict_comb.prediction,
            )

        # #################################################
        self.df_predict = df_predict_comb
        self.mae = mae
        self.r2 = r2
        # #################################################
        # __|


    # self = MA
    # k_fold_partition_size=30

    def run_kfold_cv_workflow(self,
        k_fold_partition_size=None,
        ):
        """Wed Jun  9 20:48:08 PDT 2021
        """
        # | - run_kfold_cv_workflow
        # #################################################
        can_run = self.can_run
        # #################################################
        _run_kfold_cv_workflow__get_cv_data = \
            self._run_kfold_cv_workflow__get_cv_data
        _run_kfold_cv_workflow__process_df_predict = \
            self._run_kfold_cv_workflow__process_df_predict
        _run_kfold_cv_workflow__run_infold = \
            self._run_kfold_cv_workflow__run_infold
        # #################################################

        if can_run:
            _run_kfold_cv_workflow__run_infold()

            _run_kfold_cv_workflow__get_cv_data(
                k_fold_partition_size=k_fold_partition_size,
                )

            _run_kfold_cv_workflow__process_df_predict()
        else:
            print("Can't run workflow, check 'can_run'")
        # __|


    # self=MA
    # df_train=df_train
    # df_test=df_test

    def _standardize_train_test(self,
        df_train,
        df_test=None,
        stand_targets=False,
        ):
        """Wed Jun  9 20:34:05 PDT 2021
        """
        # | - _standardize_train_test

        # Standardize training data first
        df_train_feat = df_train["features"]
        df_train_targets = df_train["targets"]


        df_train_feat_std = (df_train_feat - df_train_feat.mean()) / df_train_feat.std()
        df_train["features"] = df_train_feat_std

        if stand_targets:
            df_train_targets_std = (df_train_targets - df_train_targets.mean()) / df_train_targets.std()
            df_train["targets"] = df_train_targets_std

        if df_test is not None:
            # Standardize testing data using mean and std from training set
            df_test_feat = df_test["features"]
            df_test_targets = df_test["targets"]


            df_test_feat_std = (df_test_feat - df_train_feat.mean()) / df_train_feat.std()
            df_test["features"] = df_test_feat_std

            if stand_targets:
                df_test_targets_std = \
                    (df_test_targets - df_train_targets.mean()) / df_train_targets.std()
                df_test["targets"] = df_test_targets_std


            out_tuple = (df_train, df_test)
        else:
            out_tuple = (df_train, )

        return(out_tuple)
        # __|


    def _get_target_col_ads(self):
        """Sun Jun 13 18:00:19 PDT 2021
        """
        # | - _get_target_col_ads
        # #################################################
        df_features_targets = self.df_features_targets
        # #################################################


        target_col = df_features_targets.targets.columns[0]

        ads = None
        if target_col == "g_oh":
            ads = "oh"
        elif target_col == "g_o":
            ads = "o"

        # #################################################
        self.target_ads = ads
        # #################################################
        # __|


    def run_pca_analysis(self):
        """
        """
        # | - run_pca_analysis
        # #################################################
        PCA = self.PCA_infold
        df_features_targets = self.df_features_targets
        # #################################################
        # #################################################


        verbose = True

        if verbose:
            print("Explained variance percentage")
            print(40 * "-")
            tmp = [print(100 * i) for i in PCA.explained_variance_ratio_]
            print("")

        df_pca_comp = pd.DataFrame(
            abs(PCA.components_),
            columns=list(df_features_targets.features.columns),
            )

        # if verbose:
        #     display(df_pca_comp)

        if verbose:
            for i in range(df_pca_comp.shape[0]):
                print(40 * "-")
                print(i)
                print(40 * "-")

                df_pca_comp_i = df_pca_comp.loc[i].sort_values(ascending=False)

                print(df_pca_comp_i.iloc[0:4].to_string())
                print("")


        # #################################################
        self.df_pca_comp = df_pca_comp
        # #################################################
        # __|


    def cleanup_for_pickle(self):
        """Tue Jun 15 19:01:55 PDT 2021
        """
        # | - cleanup_for_pickle
        # #################################################
        cv_data = self.cv_data
        # #################################################

        if cv_data is not None:
            for cv_ind_i, cv_data_i in cv_data.items():
                RW = cv_data_i["RegressionWorkflow"]

                Regression = RW.Regression
                Regression.cleanup_for_pickle()
        else:
            print("cv_data is None, can't clean")
        # __|

    # __| *************************************************

class RegressionWorkflow:
    """Encapsulates everything to do with running a regression workflow, including training and prediction and all analysis
    """

    # | - RegressionWorkflow ******************************


    # df_data=df_features_targets

    def __init__(self,
        df_data=None,
        Regression=None,
        ):
        """
        Parameters
        ----------
        df_data : pandas.core.frame.DataFrame
            Main dataframe continaing all training data, features, outputs, etc.
        Regression : RegressionModel (GP_Regression, SVR_Regression, etc.)
            Regression model class, will take care of training and prediction
        """
        # | - __init__
        # #################################################
        self.df_data = df_data
        self.Regression = Regression
        # #################################################
        self.df_predict = None
        # #################################################

        #__|

    def run_Regression(self):
        """Wed Jun  9 00:30:03 PDT 2021
        """
        # | - run_Regression
        # #################################################
        df_data = self.df_data
        Regression = self.Regression
        # #################################################


        # Run regression (train model)
        Regression.run_regression(
            train_features=df_data.features,
            train_targets=df_data.targets,
            )
        # __|

    def predict(self, df_features, df_targets=None):
        """Wed Jun  9 10:11:15 PDT 2021
        """
        # | - run_Regression
        # #################################################
        Regression = self.Regression
        # #################################################


        df_predict = Regression.predict(df_features, df_targets=df_targets)

        self.df_predict = df_predict
        # __|

    # __| *************************************************


# #########################################################
# Regression model classes
# #########################################################

class RegressionModel_2(ABC):
    """Generic regression model parent class"""

    # | - RegressionModel_2 *******************************
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


    def predict(self,
        df_features,
        df_targets=None,
        ):
        """
        """
        # | - predict
        #################################################
        predict_wrap = self.predict_wrap
        #################################################


        df_predict = predict_wrap(df_features)

        # | - Attach actual target values if test_targets is given
        if df_targets is not None:
            df_targets.columns = ["actual"]

            df_predict = pd.concat([df_predict, df_targets], axis=1)
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


    # __| *************************************************

class GP_Regression(RegressionModel_2):
    """Gaussian Process Regression class"""

    # | - GP_Regression ***********************************

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
        self.init_params = None
        # #################################################


        # Inherit all methods and properties from parent RegressionModel class
        super().__init__()

        init_params = dict(
            kernel_list=kernel_list, regularization=regularization,
            optimize_hyperparameters=optimize_hyperparameters, scale_data=scale_data,
            )
        self.init_params = init_params
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

        # GP = GaussianProcess(
        #     kernel_list=kdict, regularization=noise, train_fp=train_features,
        #     train_target=train_target, optimize_hyperparameters=False,
        #     scale_data=False,
        #     )

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

    def predict_wrap(self,
        df_features,
        # df_targets=None,
        ):
        """
        """
        # | - predict
        # #################################################
        model = self.model
        # #################################################

        prediction = model.predict(
            test_fp=df_features,
            uncertainty=True,
            )

        # Construct dataframe of predictions
        df_predict = pd.DataFrame()
        df_predict["prediction"] = prediction["prediction"].flatten()
        df_predict["uncertainty"] = prediction["uncertainty"]
        df_predict["uncertainty_with_reg"] = prediction["uncertainty"]

        df_predict.index = df_features.index

        return(df_predict)
        # __|

    def cleanup_for_pickle(self):
        """
        """
        # | - cleanup_for_pickle
        model = self.model

        model.cinv = "I replaced this attribute because it causes the pickle to balloon in storage"

        self.model = model
        # __|

    # __| *************************************************

class SVR_Regression(RegressionModel_2):
    """Support Vector Regression class"""

    # | - SVR_Regression **********************************

    def __init__(self,
        # kernel_list=None,
        # regularization=None,
        # optimize_hyperparameters=None,
        # scale_data=None,
        ):
        """

        """
        # | - __init__
        # #################################################
        # self.kernel_list = kernel_list
        # self.regularization = regularization
        # self.optimize_hyperparameters = optimize_hyperparameters
        # self.scale_data = scale_data
        # #################################################
        self.model = None
        self.init_params = None
        # #################################################


        # Inherit all methods and properties from parent RegressionModel class
        super().__init__()

        init_params = dict(
            # kernel_list=kernel_list, regularization=regularization,
            # optimize_hyperparameters=optimize_hyperparameters, scale_data=scale_data,
            )
        self.init_params = init_params
        #__|


    # train_features=df_data.features
    # train_targets=df_data.targets

    def run_regression(self, train_features, train_targets):
        """
        """
        # | - run_regression
        # #################################################
        # #################################################

        model_SVR = svm.SVR(
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

        model_SVR.fit(train_features, train_targets.values.ravel())

        # #################################################
        self.model = model_SVR
        # #################################################
        # __|

    # test_features=df_data.features
    # test_targets=df_data.targets

    def predict_wrap(self,
        df_features,
        # df_targets=None,
        ):
        """
        """
        # | - predict_wrap
        # #################################################
        model = self.model
        # #################################################

        # prediction = model.predict(
        #     test_fp=df_features,
        #     uncertainty=True,
        #     )

        prediction = model.predict(
            # test_fp=df_features,
            df_features,
            # uncertainty=True,
            )

        # Construct dataframe of predictions
        df_predict = pd.DataFrame()
        df_predict["prediction"] = prediction
        # df_predict["prediction"] = prediction["prediction"].flatten()
        # df_predict["uncertainty"] = prediction["uncertainty"]
        # df_predict["uncertainty_with_reg"] = prediction["uncertainty"]

        df_predict.index = df_features.index

        return(df_predict)
        # __|


    def cleanup_for_pickle(self):
        """
        """
        # | - cleanup_for_pickle
        print("COMBAK Not implemented")

        # model = self.model
        #
        # model.cinv = "I replaced this attribute because it causes the pickle to balloon in storage"
        #
        # self.model = model
        # __|

    # __| *************************************************

class Linear_Regression(RegressionModel_2):
    """Linear Regression class"""

    # | - Linear_Regression *******************************

    def __init__(self,
        # kernel_list=None,
        # regularization=None,
        # optimize_hyperparameters=None,
        # scale_data=None,
        ):
        """

        """
        # | - __init__
        # #################################################
        # #################################################
        self.model = None
        self.init_params = None
        # #################################################


        # Inherit all methods and properties from parent RegressionModel class
        super().__init__()

        init_params = dict(
            # kernel_list=kernel_list, regularization=regularization,
            # optimize_hyperparameters=optimize_hyperparameters, scale_data=scale_data,
            )
        self.init_params = init_params
        #__|


    # from sklearn.linear_model import LinearRegression
    #
    # train_features=df_data.features
    # train_targets=df_data.targets

    def run_regression(self, train_features, train_targets):
        """
        """
        # | - run_regression
        # #################################################
        # #################################################


        X = train_features.to_numpy()
        X = X.reshape(-1, X.shape[1])


        model = LinearRegression()
        model.fit(X, train_targets)

        self.model = model
        # __|


    # test_features=df_data.features
    # test_targets=df_data.targets

    def predict_wrap(self,
        df_features,
        ):
        """
        """
        # | - predict
        # #################################################
        model = self.model
        # #################################################

        # prediction = model.predict(
        #     test_fp=df_features,
        #     uncertainty=True,
        #     )

        X_pred = df_features.to_numpy()
        X_pred = X_pred.reshape(-1, X_pred.shape[1])

        # y_pred = df_test_targets["y"].tolist()


        prediction = model.predict(df_features)

        # df_test_targets["y_pred"] = y_pred

        # df_target_pred = df_test_targets



        # Construct dataframe of predictions
        df_predict = pd.DataFrame()
        # df_predict["prediction"] = prediction["prediction"].flatten()
        df_predict["prediction"] = prediction.flatten()
        # df_predict["uncertainty"] = prediction["uncertainty"]
        # df_predict["uncertainty_with_reg"] = prediction["uncertainty"]

        df_predict.index = df_features.index

        return(df_predict)
        # __|


    def cleanup_for_pickle(self):
        """
        """
        # | - cleanup_for_pickle
        print("COMBAK")

        # model = self.model
        #
        # model.cinv = "I replaced this attribute because it causes the pickle to balloon in storage"
        #
        # self.model = model
        # __|

    # __| *************************************************

class Decision_Tree_Regression(RegressionModel_2):
    """Decision Tree Regression class"""

    # | - Decision_Tree_Regression ************************

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
        self.init_params = None
        # #################################################


        # Inherit all methods and properties from parent RegressionModel class
        super().__init__()

        init_params = dict(
            kernel_list=kernel_list, regularization=regularization,
            optimize_hyperparameters=optimize_hyperparameters, scale_data=scale_data,
            )
        self.init_params = init_params
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

        from sklearn import tree
        clf = tree.DecisionTreeRegressor()
        clf = clf.fit(train_features, train_targets)

        model = clf
        self.model = model
        # __|


    # test_features=df_data.features
    # test_targets=df_data.targets

    def predict_wrap(self,
        df_features,
        # df_targets=None,
        ):
        """
        """
        # | - predict
        # #################################################
        model = self.model
        # #################################################

        prediction = model.predict(df_features)

            # test_fp=df_features,
            # uncertainty=True,
            # )

        # Construct dataframe of predictions
        df_predict = pd.DataFrame()
        df_predict["prediction"] = prediction

        # df_predict["prediction"] = prediction["prediction"].flatten()
        # df_predict["uncertainty"] = prediction["uncertainty"]
        # df_predict["uncertainty_with_reg"] = prediction["uncertainty"]

        df_predict.index = df_features.index

        return(df_predict)
        # __|

    def cleanup_for_pickle(self):
        """
        """
        # | - cleanup_for_pickle
        model = self.model

        model.cinv = "I replaced this attribute because it causes the pickle to balloon in storage"

        self.model = model
        # __|

    # __| *************************************************






















# df_features_targets=df_j
# df_test=None
# num_pca_comp=num_pca_comp

def process_pca_analysis(
    df_features_targets=None,
    df_test=None,
    num_pca_comp=None,
    ):
    """
    """
    #| - process_pca_analysis
    df_j = df_features_targets

    # #####################################################
    out_dict = pca_analysis(
        df_j["features"],
        pca_mode="num_comp",  # 'num_comp' or 'perc'
        pca_comp=num_pca_comp,
        verbose=False,
        )
    # #####################################################
    PCA = out_dict["pca"]
    df_feat_pca = out_dict["df_pca"]
    # #####################################################



    # #####################################################
    # PCA on train data
    cols_new = []
    for col_i in df_feat_pca.columns:
        col_new_i = ("features", col_i)
        cols_new.append(col_new_i)
    df_feat_pca.columns = pd.MultiIndex.from_tuples(cols_new)

    df_pca_train = pd.concat([
        df_feat_pca,
        df_j[["targets"]],
        ], axis=1)


    # #####################################################
    # PCA on test data

    df_pca_test = None
    if df_test is not None:
        pca_test_features = PCA.transform(df_test.features)

        num_pca_comp = pca_test_features.shape[-1]

        df_pca_test = pd.DataFrame(
            pca_test_features,
            columns=['PCA%i' % i for i in range(num_pca_comp)],
            index=df_test.index)



        cols_new = []
        for col_i in df_pca_test.columns:
            col_new_i = ("features", col_i)
            cols_new.append(col_new_i)
        df_pca_test.columns = pd.MultiIndex.from_tuples(cols_new)

        df_pca_test = pd.concat([
            df_pca_test,
            df_test[["targets"]],
            ], axis=1)



    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["PCA"] = PCA
    out_dict["df_pca_train"] = df_pca_train
    out_dict["df_pca_test"] = df_pca_test
    # #####################################################
    return(out_dict)
    # #####################################################
    # __|


# pca_mode = "num_comp"  # 'num_comp' or 'perc'
# pca_comp = 5

def pca_analysis(
    df_features,
    pca_mode="num_comp",  # 'num_comp' or 'perc'
    pca_comp=5,
    verbose=True,
    ):
    """
    """
    # | - pca_analysis
    df = df_features

    shared_pca_attributes = dict(
        svd_solver="auto",
        whiten=True,
        )

    if pca_mode == "num_comp":
        num_data_points = df.shape[0]
        if num_data_points < pca_comp:
            pca_comp = num_data_points

        pca = PCA(
            n_components=pca_comp,
            **shared_pca_attributes)

    elif pca_mode == "perc":
        pca = PCA(
            n_components=pca_perc,
            **shared_pca_attributes)

    else:
        print("ISDJFIESIFJ NO GOODD")

    pca.fit(df)


    # | - Transforming the training data set
    pca_features_cleaned = pca.transform(df)

    num_pca_comp = pca_features_cleaned.shape[-1]

    if verbose:
        print("num_pca_comp: ", num_pca_comp)
        print(df.shape)

    df_pca = pd.DataFrame(
        pca_features_cleaned,
        columns=['PCA%i' % i for i in range(num_pca_comp)],
        index=df.index)

    if verbose:
        print(df_pca.shape)
    #__|

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["pca"] = pca
    out_dict["df_pca"] = df_pca
    # #####################################################
    return(out_dict)
    # #####################################################
    #__|


def plot_mae_vs_pca(
    df_models=None,
    layout_shared=None,
    scatter_marker_props=None,
    ):
    """
    """
    # | - plot_mae_vs_pca

    layout_mine = go.Layout(

        showlegend=False,

        yaxis=go.layout.YAxis(
            title=dict(
                text="K-Fold Cross Validated MAE",
                ),
            ),

        xaxis=go.layout.XAxis(
            title=dict(
                text="Num PCA Components",
                ),
            ),

        )


    # #########################################################
    layout_shared_i = layout_shared.update(layout_mine)

    trace_i = go.Scatter(
        x=df_models.index,
        y=df_models.MAE,

        mode="markers",
        marker=dict(
            **scatter_marker_props.to_plotly_json(),
            ),
        )

    data = [trace_i, ]

    fig = go.Figure(
        data=data,
        layout=layout_shared_i,
        )

    return(fig)

    # if show_plot:
    #     fig.show()

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
    # | - run_SVR_workflow

    #| - Setting up Gaussian Regression model
    train_x = df_train_features.to_numpy()
    train_y = df_train_targets
    train_y_standard = (train_y - train_y.mean()) / train_y.std()

    from sklearn import svm


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
