# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

from pathlib import Path

import numpy as np
import pandas as pd

import plotly.graph_objs as go
import chart_studio.plotly as py

# #########################################################
from vasp.vasp_methods import parse_incar


# +
compenv = os.environ["COMPENV"]

vasp_dir = "."

# For testing purposes
if compenv == "wsl":
    vasp_dir = os.path.join(
        os.environ["PROJ_irox_oer"],
        "__test__/anal_job_out")

# +
from vasp.parse_oszicar import parse_oszicar

out_dict = parse_oszicar(vasp_dir=vasp_dir)

ion_step_conv_dict = out_dict["ion_step_conv_dict"]
N_tot = out_dict["N_tot"]

# +
path_i = os.path.join(
    vasp_dir,
    "INCAR")
my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "r") as f:
        incar_lines = f.read().splitlines()
    incar_dict = parse_incar(incar_lines)

nelm_i = incar_dict["NELM"]

# +
# y_plot_quant = "E"
# y_plot_quant = "dE"
y_plot_quant = "dE_abs"

# spacing = N_tot / 15
# spacing = N_tot / 10
spacing = int(N_tot / 5)
# -

data = []
x_axis_cum = 0
for i_cnt, ion_step_i in enumerate(list(ion_step_conv_dict.keys())):

    # if i_cnt == 2:
    #     break

    df_i = ion_step_conv_dict[ion_step_i]

    

    num_N_i = df_i.N.max() - df_i.N.min()
    print("")
    print("num_N_i:", num_N_i)

    extra_spacing = nelm_i - num_N_i - spacing
    print("extra_spacing:", extra_spacing)


    df_i["below_0"] = df_i.dE < 0. 
    df_i["dE_abs"] = np.abs(df_i["dE"])

    # x_array = df_i.N + x_axis_cum + 50
    x_array = df_i.N + x_axis_cum

    # y_array = df_i.E
    y_array = df_i[y_plot_quant]

    color_array = df_i["below_0"]

    color_array_2 = []
    for i in color_array:
        if i:
            color_array_2.append("red")
        else:
            color_array_2.append("black")

    num_N = df_i.N.max()
    # x_axis_cum += num_N + 100
    # x_axis_cum += num_N + spacing + extra_spacing
    x_axis_cum += num_N + (nelm_i - num_N_i) + spacing

    # #####################################################
    trace_i = go.Scatter(
        x=x_array,
        y=y_array,
        mode="markers",
        opacity=0.8,
        marker_color=color_array_2,
        )
    data.append(trace_i)

    # #####################################################
    trace_i = go.Scatter(
        x=2 * [x_array[0] + nelm_i],
        y=[1e-10, 1e10],
        mode="lines",
        line_color="grey",
        # opacity=0.8,
        # marker_color=color_array_2,
        )
    data.append(trace_i)

# +
max_list = []
min_list = []
for i_cnt, ion_step_i in enumerate(list(ion_step_conv_dict.keys())):
    df_i = ion_step_conv_dict[ion_step_i]

    max_dE = df_i.dE_abs.max()
    min_dE = df_i.dE_abs.min()

    # print("")
    # print("max_dE:", max_dE)
    # print("min_dE:", min_dE)

    max_list.append(max_dE)
    min_list.append(min_dE)

max_y = np.max(max_list)
min_y = np.min(min_list)

# +
num_N_i = df_i.N.max() - df_i.N.min()

print("num_N_i:", num_N_i)

extra_spacing = nelm_i - num_N_i - spacing
print("extra_spacing:", extra_spacing)

# +
# assert False

# +
fig = go.Figure(data=data)

fig.update_layout(
    title=os.getcwd(),

    xaxis=go.layout.XAxis(
        title="dE",
        ),
    yaxis=go.layout.YAxis(
        title="N",
        type="log",
        # range=[min_y, max_y],
        range=[-7, 6],
        ),

    # xaxis_type="log",
    # yaxis_type="log",
    )


if compenv == "wsl":
    fig.show()

# +
from plotting.my_plotly import my_plotly_plot

if compenv != "wsl":
    write_png = True
else:
    write_png = False

my_plotly_plot(
    figure=fig,
    plot_name="scf_convergence",
    write_html=True,
    write_png=write_png,
    # png_scale=6.0,
    # write_pdf=False,
    # write_svg=False,
    try_orca_write=True,
    )
# -

# # Copy figure html file to Dropbox with rclone

# +
rclone_comm = "rclone copy out_plot/scf_convergence.html " + os.environ["rclone_dropbox"] + ":__temp__/"

import subprocess
result = subprocess.run(
    rclone_comm.split(" "),
    stdout=subprocess.PIPE)
# -

print(40 * "*")
print("*** Script finished running " + 12 * "*")
print(40 * "*")

# + active=""
#
#
#

# + jupyter={}
# root_dir = "."

# # path_i = "./job.out"
# path_i = os.path.join(root_dir, "OSZICAR")

# compenv = os.environ["COMPENV"]

# if compenv == "wsl":
#     root_dir = os.path.join(
#         os.environ["PROJ_irox_oer"],
#         "__test__/anal_job_out")

#     # path_i = "./OSZICAR"
#     # path_i = "./OSZICAR.new"
#     path_i = os.path.join(
#         root_dir,
#         # os.environ["PROJ_irox_oer"],
#         # "__test__/anal_job_out/OSZICAR.new",
#         "OSZICAR",
#         )
# with open(path_i, "r") as f:
#     oszicar_lines = f.read().splitlines()

# from vasp.vasp_methods import parse_incar

# from pathlib import Path

# path_i = os.path.join(
#     root_dir,
#     "INCAR")
# my_file = Path(path_i)
# if my_file.is_file():
#     with open(path_i, "r") as f:
#         incar_lines = f.read().splitlines()

#     incar_dict = parse_incar(incar_lines)

#     nsw_i = incar_dict["NSW"]
#     nelm_i = incar_dict["NELM"]
#     incar_parsed = True
# else:
#     incar_parsed = False

# line_beginnings = ["DAV:", "RMM:", ]

# lines_groups = []

# group_lines_i = []
# for line_i in oszicar_lines:

#     if line_i[0:4] in line_beginnings:
#         group_lines_i.append(line_i)

#     if "F= " in line_i:
#         # print("IDJIFSD")
#         lines_groups.append(group_lines_i)
#         group_lines_i = []

# # This should add the final group_lines in the case that it hasn't finished yet
# if "F= " not in oszicar_lines[-1]:
#     lines_groups.append(group_lines_i)

# N_tot = 0.

# ion_step_conv_dict = dict()
# for ion_step_i, lines_group_i in enumerate(lines_groups):

#     data_dict_list = []
#     for line_i in lines_group_i:
#         data_dict_i = dict()

#         line_list_i = [i for i in line_i.split(" ") if i != ""]

#         N_i = line_list_i[1]
#         data_dict_i["N"] = int(N_i)

#         E_i = line_list_i[2]
#         data_dict_i["E"] = float(E_i)

#         dE_i = line_list_i[3]
#         data_dict_i["dE"] = float(dE_i)

#         d_eps_i = line_list_i[4]
#         data_dict_i["d_eps"] = float(d_eps_i)

#         ncg_i = line_list_i[5]
#         data_dict_i["ncg"] = int(ncg_i)

#         rms_i = line_list_i[6]
#         data_dict_i["rms"] = float(rms_i)

#         if len(line_list_i) > 7:
#             rms_c_i = line_list_i[7]
#             data_dict_i["rms_c"] = float(rms_c_i)

#         # #################################################
#         data_dict_list.append(data_dict_i)

#     df_i = pd.DataFrame(data_dict_list)
#     # print(N_tot)
#     N_tot += df_i.N.max()

#     ion_step_conv_dict[ion_step_i] = df_i

# print("N_tot", N_tot)
