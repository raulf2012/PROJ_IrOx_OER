# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# # Create SOAP features
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle

import numpy as np
import pandas as pd

from quippy.descriptors import Descriptor

# #########################################################
from methods import (
    get_df_features_targets,
    get_df_jobs_data,
    get_df_atoms_sorted_ind,
    get_df_coord,
    get_df_jobs,
    )
# -

# ### Read Data

# +
df_features_targets = get_df_features_targets()

df_jobs = get_df_jobs()

df_jobs_data = get_df_jobs_data()

df_atoms = get_df_atoms_sorted_ind()

# +
# # TEMP
# print(222 * "TEMP")

# df_features_targets = df_features_targets.sample(n=100)

# # df_features_targets = df_features_targets.loc[
# #     [
# #         ('sherlock', 'tanewani_59', 53.0),
# #         ('slac', 'diwarise_06', 33.0),
# #         ('sherlock', 'bidoripi_03', 37.0),
# #         ('nersc', 'winomuvi_99', 83.0),
# #         ('nersc', 'legofufi_61', 90.0),
# #         ('sherlock', 'werabosi_10', 42.0),
# #         ('slac', 'sunuheka_77', 51.0),
# #         ('nersc', 'winomuvi_99', 96.0),
# #         ('slac', 'kuwurupu_88', 26.0),
# #         ('sherlock', 'sodakiva_90', 52.0),
# #         ]
# #     ]

# +
# df_features_targets = df_features_targets.loc[[
#     ("sherlock", "momaposi_60", 50., )
#     ]]

# +
# ('oer_adsorbate', 'sherlock', 'momaposi_60', 'o', 50.0, 1)
# -

# ### Filtering down to systems that won't crash script

# +
# #########################################################
rows_to_process = []
# #########################################################
for name_i, row_i in df_features_targets.iterrows():
    # #####################################################
    active_site_i = name_i[2]
    # #####################################################
    job_id_o_i = row_i[("data", "job_id_o", "")]
    # #####################################################

    # #####################################################
    row_jobs_o_i = df_jobs.loc['guhenihe_85']
    # #####################################################
    active_site_o_i = row_jobs_o_i.active_site
    # #####################################################

    # #####################################################
    row_data_i = df_jobs_data.loc[job_id_o_i]
    # #####################################################
    att_num_i = row_data_i.att_num
    # #####################################################

    atoms_index_i = (
        "oer_adsorbate",
        name_i[0], name_i[1],
        "o", active_site_o_i,
        att_num_i,
        )

    if atoms_index_i in df_atoms.index:
        rows_to_process.append(name_i)

# #########################################################
df_features_targets = df_features_targets.loc[rows_to_process]
# #########################################################
# -

# ### Main loop, running SOAP descriptors

# #########################################################
active_site_SOAP_list = []
metal_site_SOAP_list = []
ave_SOAP_list = []
# #########################################################
for name_i, row_i in df_features_targets.iterrows():
    # #####################################################
    active_site_i = name_i[2]
    # #####################################################
    job_id_o_i = row_i[("data", "job_id_o", "")]
    job_id_oh_i = row_i[("data", "job_id_oh", "")]
    job_id_bare_i = row_i[("data", "job_id_bare", "")]
    # #####################################################

    # #####################################################
    row_jobs_o_i = df_jobs.loc['guhenihe_85']
    # #####################################################
    active_site_o_i = row_jobs_o_i.active_site
    # #####################################################

    # #####################################################
    row_data_i = df_jobs_data.loc[job_id_o_i]
    # #####################################################
    # atoms_i = row_data_i.final_atoms
    att_num_i = row_data_i.att_num
    # #####################################################

    atoms_index_i = (
        # "dos_bader",
        "oer_adsorbate",
        name_i[0],
        name_i[1],
        "o",
        # name_i[2],
        active_site_o_i,
        att_num_i,
        )

    try:
        # #####################################################
        row_atoms_i = df_atoms.loc[atoms_index_i]
        # #####################################################
        atoms_i = row_atoms_i.atoms_sorted_good
        # #####################################################

    except:
        print(name_i)

    # print(
    #     "N_atoms: ",
    #     atoms_i.get_global_number_of_atoms(),
    #     sep="")

    # Original
    # desc = Descriptor("soap cutoff=4 l_max=3 n_max=4 normalize=T atom_sigma=0.5 n_Z=1 Z={14} ")

    # This one works
    # desc = Descriptor("soap cutoff=4 l_max=10 n_max=10 normalize=T atom_sigma=0.5 n_Z=2 Z={8 77} ")

    # THIS ONE IS GOOD ******************************
    # desc = Descriptor("soap cutoff=4 l_max=3 n_max=4 normalize=F atom_sigma=0.2 n_Z=2 Z={8 77} ")

    # Didn't work great
    # desc = Descriptor("soap cutoff=8 l_max=6 n_max=6 normalize=F atom_sigma=0.1 n_Z=2 Z={8 77} ")

    # desc = Descriptor("soap cutoff=4 l_max=6 n_max=6 normalize=F atom_sigma=0.1 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=6 n_max=6 normalize=F atom_sigma=0.5 n_Z=2 Z={8 77} ")

    # desc = Descriptor("soap cutoff=4 l_max=6 n_max=6 normalize=T atom_sigma=0.5 n_Z=2 Z={8 77} ")

    # desc = Descriptor("soap cutoff=4 l_max=3 n_max=4 normalize=T atom_sigma=0.2 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=3 n_max=4 normalize=T atom_sigma=0.4 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=3 n_max=4 normalize=T atom_sigma=0.6 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=3 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=3 n_max=4 normalize=T atom_sigma=0.2 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=3 n_max=4 normalize=T atom_sigma=0.25 n_Z=2 Z={8 77} ")

    # desc = Descriptor("soap cutoff=4 l_max=3 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=2 l_max=3 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=6 l_max=3 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=5 l_max=3 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=3 l_max=3 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=6 l_max=3 n_max=4 normalize=T atom_sigma=0.1 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=6 l_max=3 n_max=4 normalize=T atom_sigma=0.05 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=6 l_max=3 n_max=4 normalize=T atom_sigma=0.2 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=6 l_max=3 n_max=4 normalize=T atom_sigma=0.4 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=6 l_max=3 n_max=4 normalize=T atom_sigma=0.5 n_Z=2 Z={8 77} ")

    # desc = Descriptor("soap cutoff=4 l_max=3 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=3 l_max=3 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=3 l_max=3 n_max=4 normalize=T atom_sigma=0.2 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=3 l_max=3 n_max=4 normalize=T atom_sigma=0.5 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=3 l_max=3 n_max=4 normalize=T atom_sigma=0.1 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=3 l_max=3 n_max=4 normalize=T atom_sigma=0.6 n_Z=2 Z={8 77} ")

    # desc = Descriptor("soap cutoff=4 l_max=4 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=5 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=6 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=7 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")

    # desc = Descriptor("soap cutoff=4 l_max=6 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=6 n_max=6 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=6 n_max=3 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")

    #  Optimizing the new SOAP_ave model
    # desc = Descriptor("soap cutoff=4 l_max=6 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=6 n_max=4 normalize=T atom_sigma=0.2 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=4 l_max=6 n_max=4 normalize=T atom_sigma=0.4 n_Z=2 Z={8 77} ")

    # desc = Descriptor("soap cutoff=5 l_max=6 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")
    # desc = Descriptor("soap cutoff=3 l_max=6 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")

    desc = Descriptor("soap cutoff=4 l_max=6 n_max=4 normalize=T atom_sigma=0.3 n_Z=2 Z={8 77} ")


    # desc.sizes(atoms_i)

    d = desc.calc(atoms_i)
    SOAP_m_i = d["data"]

    active_site_SOAP_vector_i = SOAP_m_i[int(active_site_i)]

    active_site_SOAP_list.append(
        (name_i, active_site_SOAP_vector_i)
        )

    # print(
    #     "data shape: ",
    #     d['data'].shape,
    #     sep="")

    # #####################################################
    # Get df_coord to find nearest neighbors
    init_slab_name_tuple_i = (
        name_i[0],
        name_i[1],
        "o",
        # name_i[2],
        active_site_o_i,
        att_num_i
        )

    df_coord_i = get_df_coord(
        mode="init-slab",  # 'bulk', 'slab', 'post-dft', 'init-slab'
        init_slab_name_tuple=init_slab_name_tuple_i,
        verbose=False,
        )

    # #####################################################
    row_coord_i = df_coord_i.loc[active_site_i]
    # #####################################################
    nn_info_i = row_coord_i.nn_info
    # #####################################################


    # assert len(nn_info_i) == 1, "Only one bound Ir"

    ir_nn_present = False
    for j_cnt, nn_j in enumerate(nn_info_i):
        if nn_j["site"].specie.symbol == "Ir":
            ir_nn_present = True
    assert ir_nn_present, "Ir has to be in nn list"

    # assert nn_info_i[j_cnt]["site"].specie.symbol == "Ir", "Has to be"

    metal_index_i = nn_info_i[0]["site_index"]

    metal_site_SOAP_vector_i = SOAP_m_i[int(metal_index_i)]

    metal_site_SOAP_list.append(
        (name_i, metal_site_SOAP_vector_i)
        )

    # #####################################################
    # Averaging SOAP vectors for Ir and 6 oxygens
    row_coord_Ir_i = df_coord_i.loc[metal_index_i]

    vectors_to_average = []
    for nn_j in row_coord_Ir_i["nn_info"]:
        if nn_j["site"].specie.symbol == "O":
            O_SOAP_vect_i = SOAP_m_i[int(nn_j["site_index"])]
            vectors_to_average.append(O_SOAP_vect_i)

    vectors_to_average.append(metal_site_SOAP_vector_i)

    SOAP_vector_ave_i = np.mean(
        vectors_to_average,
        axis=0
        )
    ave_SOAP_list.append(
        (name_i, SOAP_vector_ave_i)
        )

# +
# vectors_to_average = []
# for nn_j in row_coord_Ir_i["nn_info"]:
#     if nn_j["site"].specie.symbol == "O":
#         O_SOAP_vect_i = SOAP_m_i[int(nn_j["site_index"])]
#         vectors_to_average.append(O_SOAP_vect_i)

# vectors_to_average.append(metal_site_SOAP_vector_i)

# SOAP_vector_ave_i = np.mean(
#     vectors_to_average,
#     axis=0
#     )

# +
# data = []
# for i in vectors_to_average:
#     trace = go.Scatter(
#         y=i,
#         )
#     data.append(trace)

# +
# tmp = np.mean(
#     vectors_to_average,
#     axis=0
#     )

# # import plotly.graph_objs as go

# trace = go.Scatter(
#     # x=x_array,
#     y=tmp,
#     )
# # data = [trace]
# data.append(trace)

# fig = go.Figure(data=data)
# fig.show()
# -

# ### Forming the SOAP vector dataframe about the active site atom

# +
data_dict_list = []
tmp_SOAP_vector_list = []
tmp_name_list = []
for name_i, SOAP_vect_i in active_site_SOAP_list:
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    name_dict_i = dict(zip(
        ["compenv", "slab_id", "active_site", ],
        name_i, ))
    # #####################################################

    tmp_SOAP_vector_list.append(SOAP_vect_i)
    tmp_name_list.append(name_i)

    # #####################################################
    data_dict_i.update(name_dict_i)
    # #####################################################
    # data_dict_i[""] = 
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
SOAP_vector_matrix_AS = np.array(tmp_SOAP_vector_list)
df_SOAP_AS = pd.DataFrame(SOAP_vector_matrix_AS)
df_SOAP_AS.index = pd.MultiIndex.from_tuples(tmp_name_list, names=["compenv", "slab_id", "active_site"])
# #########################################################
# -

df_SOAP_AS.head()

# ### Forming the SOAP vector dataframe about the active Ir atom

# +
data_dict_list = []
tmp_SOAP_vector_list = []
tmp_name_list = []
for name_i, SOAP_vect_i in metal_site_SOAP_list:
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    name_dict_i = dict(zip(
        ["compenv", "slab_id", "active_site", ],
        name_i, ))
    # #####################################################

    tmp_SOAP_vector_list.append(SOAP_vect_i)
    tmp_name_list.append(name_i)

    # #####################################################
    data_dict_i.update(name_dict_i)
    # #####################################################
    # data_dict_i[""] = 
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
SOAP_vector_matrix_MS = np.array(tmp_SOAP_vector_list)
df_SOAP_MS = pd.DataFrame(SOAP_vector_matrix_MS)
df_SOAP_MS.index = pd.MultiIndex.from_tuples(tmp_name_list, names=["compenv", "slab_id", "active_site"])
# #########################################################
# -

df_SOAP_MS.head()

# ### Forming the SOAP vector dataframe averaged from Ir + 6 O

# +
data_dict_list = []
tmp_SOAP_vector_list = []
tmp_name_list = []
for name_i, SOAP_vect_i in ave_SOAP_list:
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    name_dict_i = dict(zip(
        ["compenv", "slab_id", "active_site", ],
        name_i, ))
    # #####################################################

    tmp_SOAP_vector_list.append(SOAP_vect_i)
    tmp_name_list.append(name_i)

    # #####################################################
    data_dict_i.update(name_dict_i)
    # #####################################################
    # data_dict_i[""] = 
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
SOAP_vector_matrix_ave = np.array(tmp_SOAP_vector_list)
df_SOAP_ave = pd.DataFrame(SOAP_vector_matrix_ave)
df_SOAP_ave.index = pd.MultiIndex.from_tuples(tmp_name_list, names=["compenv", "slab_id", "active_site"])
# #########################################################
# -

# ### TEMP Plotting

# +
import plotly.graph_objs as go

from plotting.my_plotly import my_plotly_plot

import plotly.express as px

# +
y_array = SOAP_m_i[int(metal_index_i)]

trace = go.Scatter(
    y=y_array,
    )
data = [trace]

fig = go.Figure(data=data)
# fig.show()

# +
fig = px.imshow(
    df_SOAP_AS.to_numpy(),
    aspect='auto',  # 'equal', 'auto', or None
    )

my_plotly_plot(
    figure=fig,
    plot_name="df_SOAP_AS",
    write_html=True,
    )

fig.show()

# +
fig = px.imshow(
    df_SOAP_MS.to_numpy(),
    aspect='auto',  # 'equal', 'auto', or None
    )

my_plotly_plot(
    figure=fig,
    plot_name="df_SOAP_MS",
    write_html=True,
    )

fig.show()

# +
fig = px.imshow(
    df_SOAP_ave.to_numpy(),
    aspect='auto',  # 'equal', 'auto', or None
    )

my_plotly_plot(
    figure=fig,
    plot_name="df_SOAP_MS",
    write_html=True,
    )

fig.show()
# -

# ### Save data to file

# +
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering/SOAP_QUIP")

directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory): os.makedirs(directory)

# Pickling data ###########################################
path_i = os.path.join(root_path_i, "out_data/df_SOAP_AS.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_SOAP_AS, fle)
# #########################################################

# Pickling data ###########################################
path_i = os.path.join(root_path_i, "out_data/df_SOAP_MS.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_SOAP_MS, fle)
# #########################################################

# Pickling data ###########################################
path_i = os.path.join(root_path_i, "out_data/df_SOAP_ave.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_SOAP_ave, fle)
# #########################################################



# # #########################################################
# import pickle; import os
# with open(path_i, "rb") as fle:
#     df_SOAP_AS = pickle.load(fle)
# # #########################################################

# +
from methods import get_df_SOAP_AS, get_df_SOAP_MS, get_df_SOAP_ave

df_SOAP_AS_tmp = get_df_SOAP_AS()
df_SOAP_AS_tmp

df_SOAP_MS_tmp = get_df_SOAP_MS()
df_SOAP_MS_tmp

df_SOAP_ave_tmp = get_df_SOAP_ave()
df_SOAP_ave_tmp
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("SOAP_features.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# SOAP_m_i.shape

# + jupyter={"source_hidden": true}
# # desc?

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# atoms_i

# + jupyter={"source_hidden": true}
# atoms_i

# + jupyter={"source_hidden": true}
# import

# import quippy


# quippy.descriptors.

# + jupyter={"source_hidden": true}
# SOAP_m_i.shape

# + jupyter={"source_hidden": true}
# dir(plt.matshow(d['data']))

# plt.matshow(d['data'])

# + jupyter={"source_hidden": true}
# import os

# import numpy as np
# import matplotlib.pylab as plt

# from quippy.potential import Potential


# from ase import Atoms, units
# from ase.build import add_vacuum
# from ase.lattice.cubic import Diamond
# from ase.io import write

# from ase.constraints import FixAtoms

# from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
# from ase.md.verlet import VelocityVerlet
# from ase.md.langevin import Langevin

# from ase.optimize.precon import PreconLBFGS, Exp

# # from gap_si_surface import ViewStructure

# + jupyter={"source_hidden": true}

# # pd.MultiIndex.from_tuples?
# -



# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# job_id_o_i

# + jupyter={"source_hidden": true}
# dir(nn_info_i[0]["site"])

# 'properties',
# 'specie',
# 'species',
# 'species_and_occu',
# 'species_string',

# + jupyter={"source_hidden": true}
# metal_index_i

# + jupyter={"source_hidden": true}
# assert False

# +
# from plotting.my_plotly import my_plotly_plot


