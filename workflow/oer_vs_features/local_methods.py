"""
"""

#| - Import Modules
import os
import sys

import copy

import numpy as np
from sklearn.linear_model import LinearRegression

import plotly.graph_objs as go

# #########################################################
from plotting.my_plotly import my_plotly_plot
#__|



# df = df_j
# feature_columns = [feature_col_i, ]
# ads = ads_i
# feature_ads = feature_ads_i
# layout = layout
# verbose = True

def create_linear_model_plot(
    df=None,
    feature_columns=None,
    ads=None,
    feature_ads=None,
    format_dict=None,
    layout=None,
    verbose=True,
    save_plot_to_file=False,
    ):
    """
    """
    #| - create_linear_model_plot
    # #####################################################
    df_i = df
    features_cols_to_include = feature_columns
    # #####################################################

    #| - Dropping feature columns
    if features_cols_to_include is None or features_cols_to_include == "all":
        features_cols_to_include = df_i["features_stan"][feature_ads].columns

    cols_to_drop = []
    for col_i in df_i["features_stan"][feature_ads].columns:
        if col_i not in features_cols_to_include:
            cols_to_drop.append(col_i)
    df_tmp = copy.deepcopy(df_i)

    for col_i in cols_to_drop:
        df_i = df_i.drop(columns=[("features_stan", feature_ads, col_i)])

    # feature_cols = list(df_i.features_stan.columns)
    feature_cols = list(df_i["features_stan"][feature_ads].columns)
    # print(feature_cols)


    plot_title = " | ".join(feature_cols)
    plot_title = "Features: " + plot_title
    #__|

    #| - Creating linear model
    X = df_i["features_stan"][feature_ads].to_numpy()
    X = X.reshape(-1, len(df_i["features_stan"][feature_ads].columns))

    y = df_i.targets[
        df_i.targets.columns[0]
        ]

    model = LinearRegression()
    model.fit(X, y)

    y_predict = model.predict(X)

    if verbose:
        print(20 * "-")
        print("model.score(X, y):", model.score(X, y))
        print("")

        # print(feature_cols)
        # print(model.coef_)

        for i, j in zip(list(df_i["features_stan"][ads].columns), model.coef_):
            print(i, ": ", j, sep="")
        print(20 * "-")

    #__|

    #| - Plotting
    data = []


    from methods import get_df_slab
    df_slab = get_df_slab()


    #| - DEPRECATED | Getting colors ready
    # df_slab_tmp = df_slab[["slab_id", "bulk_id"]]
    #
    # bulk_id_slab_id_lists = np.reshape(
    #     df_slab_tmp.to_numpy(),
    #     (
    #         2,
    #         df_slab_tmp.shape[0],
    #         )
    #     )
    #
    # slab_bulk_mapp_dict = dict(zip(
    #     list(bulk_id_slab_id_lists[0]),
    #     list(bulk_id_slab_id_lists[1]),
    #     ))
    #
    #
    # slab_bulk_id_map_dict = dict()
    # for i in df_slab_tmp.to_numpy():
    #     slab_bulk_id_map_dict[i[0]] = i[1]
    #
    # # print("list(bulk_id_slab_id_lists[0]):", list(bulk_id_slab_id_lists[0]))
    # # print("")
    # # print("list(bulk_id_slab_id_lists[1]):", list(bulk_id_slab_id_lists[1]))
    # # print("")
    # # print("slab_bulk_mapp_dict:", slab_bulk_mapp_dict)
    #
    # import random
    # get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    #
    # slab_id_unique_list = df_i.index.to_frame()["slab_id"].unique().tolist()
    #
    # bulk_id_list = []
    # for slab_id_i in slab_id_unique_list:
    #     # bulk_id_i = slab_bulk_mapp_dict[slab_id_i]
    #     bulk_id_i = slab_bulk_id_map_dict[slab_id_i]
    #     bulk_id_list.append(bulk_id_i)
    #
    # color_map_dict = dict(zip(
    #     bulk_id_list,
    #     get_colors(len(slab_id_unique_list)),
    #     ))
    #
    # # Formatting processing
    # color_list = []
    # for name_i, row_i in df_i.iterrows():
    #     # #################################################
    #     slab_id_i = name_i[1]
    #     # #################################################
    #     phase_i = row_i["data"]["phase"][""]
    #     stoich_i = row_i["data"]["stoich"][""]
    #     sum_norm_abs_magmom_diff_i = row_i["data"]["sum_norm_abs_magmom_diff"][""]
    #     norm_sum_norm_abs_magmom_diff_i = row_i["data"]["norm_sum_norm_abs_magmom_diff"][""]
    #     # #################################################
    #
    #     # #################################################
    #     row_slab_i = df_slab.loc[slab_id_i]
    #     # #################################################
    #     bulk_id_i = row_slab_i.bulk_id
    #     # #################################################
    #
    #     bulk_color_i = color_map_dict[bulk_id_i]
    #
    #     if stoich_i == "AB2":
    #         color_list.append("#46cf44")
    #     elif stoich_i == "AB3":
    #         color_list.append("#42e3e3")
    #
    #     # color_list.append(norm_sum_norm_abs_magmom_diff_i)
    #     # color_list.append(bulk_color_i)
    #__|


    #| - Creating parity line
    # x_parity = y_parity = np.linspace(0., 8., num=100, )
    x_parity = y_parity = np.linspace(-2., 8., num=100, )

    trace_i = go.Scatter(
        x=x_parity,
        y=y_parity,
        line=go.scatter.Line(color="black", width=2.),
        mode="lines")
    data.append(trace_i)
    #__|

    #| - Main Data Trace

    color_list_i = df_i["format"]["color"][format_dict["color"]]

    trace_i = go.Scatter(
        y=y,
        x=y_predict,
        mode="markers",
        marker=go.scatter.Marker(
            size=12,
            color=color_list_i,

            colorscale='Viridis',
            colorbar=dict(thickness=20),

            opacity=0.8,

            ),
        # text=df_i.name_str,
        text=df_i.data.name_str,
        textposition="bottom center",
        )
    data.append(trace_i)
    #__|

    #| - Layout

    # y_axis_target_col = df_i.target_cols.columns[0]
    y_axis_target_col = df_i.targets.columns[0]
    y_axis_target_col = y_axis_target_col[0]

    # print("y_axis_target_col:", y_axis_target_col)

    # print("y_axis_target_col:", y_axis_target_col)
    if y_axis_target_col == "g_o":
        # print("11111")
        layout.xaxis.title.text = "Predicted ΔG<sub>*O</sub>"
        layout.yaxis.title.text = "Simulated ΔG<sub>*O</sub>"
    elif y_axis_target_col == "g_oh":
        # print("22222")
        layout.xaxis.title.text = "Predicted ΔG<sub>*OH</sub>"
        layout.yaxis.title.text = "Simulated ΔG<sub>*OH</sub>"
    else:
        print("Woops isdfsdf8osdfio")


    layout.xaxis.title.font.size = 25
    layout.yaxis.title.font.size = 25

    layout.yaxis.tickfont.size = 20
    layout.xaxis.tickfont.size = 20

    layout.xaxis.range = [2.5, 5.5]

    layout.showlegend = False

    dd = 0.2
    layout.xaxis.range = [
        np.min(y_predict) - dd,
        np.max(y_predict) + dd,
        ]


    layout.yaxis.range = [
        np.min(y) - dd,
        np.max(y) + dd,
        ]

    # layout.title = "TEMP isdjfijsd"
    layout.title = plot_title
    #__|

    fig = go.Figure(data=data, layout=layout)

    if save_plot_to_file:
        my_plotly_plot(
            figure=fig,
            save_dir=os.path.join(
                os.environ["PROJ_irox_oer"],
                "workflow/oer_vs_features",
                ),
            plot_name="parity_plot",
            write_html=True)

    #__|

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["fig"] = fig
    # #####################################################
    return(out_dict)
    #__|

def isolate_target_col(df, target_col=None):
    """
    """
    #| - isolate_target_col
    df_i = df
    target_col_to_plot = target_col

    cols_tuples = []
    for col_i in list(df_i.columns):
        if "features_stan" in col_i[0]:
            cols_tuples.append(col_i)
        # elif col_i == ("target_cols", target_col_to_plot):
        elif col_i == ("targets", target_col_to_plot, ""):
            cols_tuples.append(col_i)
        elif col_i[0] == "data":
            cols_tuples.append(col_i)
        elif col_i[0] == "format":
            cols_tuples.append(col_i)
        else:
            tmp = 42

    df_j = df_i.loc[:, cols_tuples]

    df_j = df_j.dropna()

    return(df_j)
    #__|
