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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Processing IrOx Data from Seoin
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys


import pickle

import numpy as np
import pandas as pd

from ase import Atoms
# -

# ### Script Inputs

# +
# From Seoin
oxy_ref = -7.4484
hyd_ref = -3.3851

# # Mine
# oxy_ref = -7.45942759
# hyd_ref = -3.38574595

# MISC
# # oxy_ref = -7.459
# oxy_ref = -7.463
# hyd_ref = -3.38574595

# oxy_ref = -7.469
# hyd_ref = -3.38574595

# oxy_ref = -7.489
# hyd_ref = -3.38574595
# -

# From Seoin
oxy_ref = -7.4484
# hyd_ref = -3.3851
# hyd_ref = -3.395
# hyd_ref = -3.405
# hyd_ref = -3.415
# hyd_ref = -3.43
hyd_ref = -3.44

-3.3851 - -3.44

# +
# # From Seoin
# G_corr_o = 0.05
# G_corr_oh = 0.34
# G_corr_ooh = 0.37

# G_corr_h2 = -0.04
# G_corr_h2o = 0.0


# Mine
G_corr_o = 0.081
G_corr_oh = 0.307
G_corr_ooh = 0.426

G_corr_h2 = -0.049
G_corr_h2o = -0.012

# +
G_corr_o_tot = G_corr_o - (G_corr_h2o - G_corr_h2)
G_corr_oh_tot = G_corr_oh - (G_corr_h2o - 0.5 * G_corr_h2)
G_corr_ooh_tot = G_corr_ooh - (2 * G_corr_h2o - 1.5 * G_corr_h2)

print(
    "G_corr_o_tot: ",
    G_corr_o_tot,
    "\n",

    "G_corr_oh_tot: ",
    G_corr_oh_tot,
    "\n",

    "G_corr_ooh_tot: ",
    G_corr_ooh_tot,
    "\n",

    sep="")

# +
# G_corr_o_tot: 0.010000000000000002
# G_corr_oh_tot: 0.32
# G_corr_ooh_tot: 0.31

# +
# G_corr_o_tot: 0.044
# G_corr_oh_tot: 0.2945
# G_corr_ooh_tot: 0.3765
# -

# ### Read Data

# +
# #########################################################
path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data",
    "in_data/oer.pkl")
with open(path_i, 'rb') as f:
     oer_data = pickle.load(f) 

# #########################################################
path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data",
    "in_data/all_info.csv")
df_oer = pd.read_csv(path_i, dtype={"facet": object})

# +
# #########################################################
path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data",
    "manually_id_active_site.csv")
df_active_sites = pd.read_csv(path_i,
    dtype={"facet": object},
    )

df_active_sites = df_active_sites.set_index(
    ["crystal", "facet", "coverage", "termination", "active_site", ])

# +
# # TEMP
# print(222 * "TEMP | ")

# df_oer = df_oer.loc[

#     # [
#     #     # 3,
#     #     32,
#     #   ]

#     # #####################################################

#     # [
#     #     3, 150, 178, 186, 211, 212,
#     #     351, 368, 382, 443, 461, 506,
#     #     524, 533, 534, 618, 624, 664,
#     #     702, 710, 722, 789, 859, 901,
#     #     913, 939, 971, 975, 991, 992,
#     #     993, 1001, 1016, 1021, 1037, 1055,
#     #     1061, 1065, 1083,
#     #     ]

#     # #####################################################

#     # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

#     # [
#     #     887, 889, 898, 899, 900,
#     #     914, 925, 926, 932, 933,
#     #     937, 947, 948, 949, 957,
#     #     958, 959, 961, 973, 974,
#     #     975, 983, 984,
#     #     ]

#     [961, 973, 974, 975, 983, 984, 963],

#     ]
# -

# ### Process `df_oer`

df_oer = df_oer.rename(
    columns={
        "termination": "termination_str",
        }
    )


# +
def method(row_i):

    # #####################################################
    new_column_values_dict = {
        "ads": None,
        }
    # #####################################################
    name_i = row_i["name"]
    energy_i = row_i["energy"]
    location_i = row_i["location"]
    termination_str_i = row_i["termination_str"]
    # #####################################################




    # #####################################################
    # Parse termination text
    if "O_covered" in termination_str_i:
        coverage_type_i = "O_covered"
    if "OH_covered" in termination_str_i:
        coverage_type_i = "OH_covered"

    # row_i = df_oer.iloc[0]
    # termination_str_i = row_i.termination_str

    termination_int_i = None
    for i in termination_str_i.split("_"):
        try:
            int_i = int(i)
            termination_int_i = int_i
            break
        except:
            tmp = 42


    # #####################################################
    # Short location
    loc_short_i = location_i[56:]



    # #####################################################
    # Parsing adsorbate species
    is_bare = any([i == "Bare" for i in name_i.split("_")])
    is_O = any([i == "O" for i in name_i.split("_")])
    is_OH = any([i == "OH" for i in name_i.split("_")])
    is_OOH = any([i == "OOH" for i in name_i.split("_")])

    if is_bare:
        ads_i = "bare"
    elif is_O:
        ads_i = "o"
    elif is_OH:
        ads_i = "oh"
    elif is_OOH:
        ads_i = "ooh"
    else:
        ads_i = None


    # #####################################################
    # Getting OER data from sorted dict object
    num_matches = 0
    oer_ind_i = None
    
    # print(20 * "-")
    # print(energy_i)

    for oer_ind_j, oer_data_j in enumerate(oer_data):
        energy_j = oer_data_j["results"]["energy"]

        is_close = np.isclose(
            energy_i, energy_j,
            # rtol=1e-05,
            # atol=1e-08,

            rtol=1e-09,
            atol=1e-09,
            )
        if is_close:
            # print(energy_j)

            num_matches += 1
            oer_ind_i = oer_ind_j

    if num_matches > 1:
        # print(row_i.name)
        assert False

    # #####################################################
    oer_data_i = oer_data[oer_ind_i]
    # #####################################################
    atoms_i = oer_data_i["atoms"]
    calc = oer_data_i["calc"]
    results = oer_data_i["results"]
    # #####################################################
    initial_configuration = oer_data_i["initial_configuration"]
    # #####################################################
    atoms_init_i = initial_configuration["atoms"]
    # results_init_i = initial_configuration["results"]
    # #####################################################


    symbol_counts_i = atoms_i["symbol_counts"]

    O_Ir_frac_i = symbol_counts_i["O"] / symbol_counts_i["Ir"]
    # print(O_Ir_frac_i, ",", sep="")

        
    # #####################################################
    symbols_list = []
    positions_list = []
    for atom_i in atoms_i["atoms"]:
        symbols_list.append(atom_i["symbol"])
        positions_list.append(atom_i["position"])


    atoms_i_2 = Atoms(
        symbols=symbols_list,
        positions=positions_list,

        cell=atoms_i["cell"],
        pbc=atoms_i["pbc"],
        constraint=atoms_i["constraints"],

        # numbers=None,
        # tags=None,
        # momenta=None,
        # masses=None,
        # magmoms=None,
        # charges=None,
        # scaled_positions=None,

        # celldisp=None,

        # calculator=None,
        # info=None,
        # velocities=None,
        )

    # #####################################################
    symbols_list = []
    positions_list = []
    for atom_i in atoms_init_i["atoms"]:
        symbols_list.append(atom_i["symbol"])
        positions_list.append(atom_i["position"])


    atoms_init_i_2 = Atoms(
        symbols=symbols_list,
        positions=positions_list,

        cell=atoms_init_i["cell"],
        pbc=atoms_init_i["pbc"],
        constraint=atoms_init_i["constraints"],
        )



    # #####################################################
    # Getting active site and attempt number
    name_split_i = name_i.split("_")

    # if name_split_i[0] == "OH":
    if "OH" in name_split_i:
        # doubled_OH_covered_0_0
        active_site_i = name_split_i[-2]
        attempt_i = name_split_i[-1]

    elif name_split_i[0] == "OOH":
        active_site_i = name_split_i[1]
        attempt_i = name_split_i[2]

    elif name_split_i[0] == "Bare":
        active_site_i = name_split_i[1]
        if len(name_split_i) == 2:
            attempt_i = 0
        else:
            print("Check this out")

    # elif name_split_i[0] == "O":
    # name_split_i[0] == "O":
    elif "O" in name_split_i:
        active_site_i = name_split_i[-1]
        attempt_i = 0

    else:
        print(name_i)
        print("Wooooooooops")







    # #####################################################
    new_column_values_dict["ads"] = ads_i
    new_column_values_dict["oer_ind"] = oer_ind_i
    new_column_values_dict["atoms"] = atoms_i_2
    new_column_values_dict["atoms_init"] = atoms_init_i_2
    new_column_values_dict["loc_short"] = loc_short_i
    new_column_values_dict["active_site"] = int(active_site_i)
    new_column_values_dict["attempt"] = attempt_i
    new_column_values_dict["coverage_type"] = coverage_type_i
    new_column_values_dict["termination"] = termination_int_i
    new_column_values_dict["O_Ir_frac"] = O_Ir_frac_i
    # #####################################################
    for key, value in new_column_values_dict.items():
        row_i[key] = value
    # #####################################################
    return(row_i)
    # #####################################################


# #########################################################
df_oer = df_oer.apply(
    method,
    axis=1,
    )
# #########################################################
# -

# ### Looping over OER sets, preparing adsorption energies

# +
### Removing *OH covered slabs
# It was messing up the code below, and I don't really need these calculations

# df_oer = df_oer[df_oer.coverage_type != "OH_covered"]



df_oer_wo_O = df_oer[df_oer.ads != "o"]

# #########################################################
data_dict_list = []
# #########################################################
group_cols = ["crystal", "facet", "termination_str", "termination", "coverage_type", "active_site", ]
grouped = df_oer_wo_O.groupby(group_cols)
# #########################################################
for name, group in grouped:

# for i in range(1):
#     name = ('columbite', '120', 'O_covered_1_OER', 1, 'O_covered', 3)
#     group = grouped.get_group(name)

    # #####################################################
    crystal_i = name[0]
    facet_i = name[1]
    termination_str_i = name[2]
    termination_i = name[3]
    coverage_type_i = name[4]
    active_site_i = name[5]
    # #####################################################

    O_Ir_frac_ave_i = group["O_Ir_frac"].mean()

    if O_Ir_frac_ave_i < 3:
        bulk_oxid_state_i = 4
        stoich_i = "AB2"
    else:
        bulk_oxid_state_i = 6
        stoich_i = "AB3"


    # #####################################################
    oh_present_i = "oh" in group.ads.unique().tolist()
    ooh_present_i = "ooh" in group.ads.unique().tolist()
    bare_present_i = "bare" in group.ads.unique().tolist()
    # #####################################################
    all_ads_present = np.all([oh_present_i, ooh_present_i, bare_present_i])
    # #####################################################

    if all_ads_present:

        # #####################################################
        df = group
        df = df[
            (df["ads"] == "oh") &
            [True for i in range(len(df))]
            ]
        group_oh = df

        row_oh_i = group_oh.sort_values("energy").iloc[[0]]

        # #####################################################
        df = group
        df = df[
            (df["ads"] == "ooh") &
            [True for i in range(len(df))]
            ]
        group_ooh = df

        row_ooh_i = group_ooh.sort_values("energy").iloc[[0]]

        # #####################################################
        df = group
        df = df[
            (df["ads"] == "bare") &
            [True for i in range(len(df))]
            ]
        group_bare = df

        row_bare_i = group_bare.sort_values("energy").iloc[[0]]


        # #####################################################
        df = df_oer
        df = df[
            (df["termination"] == termination_i) &
            (df["ads"] == "o") &
            (df["crystal"] == crystal_i) &
            (df["facet"] == facet_i) &
            [True for i in range(len(df))]
            ]
        group_o = df

        if group_o.shape[0] > 1:
            print("There are various *O slabs, doesn't seem right")
            print(
                '"crystal", "facet", "termination_str", "termination", "coverage_type", "active_site",'
                )
            print(name)

        row_o_i = group_o.sort_values("energy").iloc[[0]]

        assert row_o_i.shape[0] == 1, "ISDFIJDSIJfi"


        # Adsorbate indices for df_eor
        bare_index_i = row_bare_i.index.tolist()[0]
        oh_index_i = row_oh_i.index.tolist()[0]
        ooh_index_i = row_ooh_i.index.tolist()[0]
        o_index_i = row_o_i.index.tolist()[0]

        # #####################################################
        df_oer_set_i = pd.concat([
            row_bare_i,
            row_oh_i,
            row_ooh_i,
            row_o_i,
            ], axis=0)

        energy_o_i = df_oer_set_i[df_oer_set_i.ads == "o"].iloc[0].energy
        energy_oh_i = df_oer_set_i[df_oer_set_i.ads == "oh"].iloc[0].energy
        energy_ooh_i = df_oer_set_i[df_oer_set_i.ads == "ooh"].iloc[0].energy
        energy_bare_i = df_oer_set_i[df_oer_set_i.ads == "bare"].iloc[0].energy

        dE_O = energy_o_i - energy_bare_i - oxy_ref
        dE_OH = energy_oh_i - energy_bare_i - oxy_ref - hyd_ref
        dE_OOH = energy_ooh_i - energy_bare_i - 2 * oxy_ref - hyd_ref

        dG_O = dE_O + G_corr_o_tot
        dG_OH = dE_OH + G_corr_oh_tot
        dG_OOH = dE_OOH + G_corr_ooh_tot








        # #################################################
        data_dict_i = dict()
        # #################################################
        data_dict_i["stoich"] = stoich_i
        data_dict_i["crystal"] = crystal_i
        data_dict_i["facet"] = facet_i
        data_dict_i["termination"] = termination_i
        data_dict_i["active_site"] = active_site_i
        data_dict_i["coverage"] = coverage_type_i
        data_dict_i["active_site"] = active_site_i
        data_dict_i["e_o"] = dE_O
        data_dict_i["e_oh"] = dE_OH
        data_dict_i["e_ooh"] = dE_OOH
        data_dict_i["g_o"] = dG_O
        data_dict_i["g_oh"] = dG_OH
        data_dict_i["g_ooh"] = dG_OOH
        data_dict_i["index_bare"] = bare_index_i
        data_dict_i["index_o"] = o_index_i
        data_dict_i["index_oh"] = oh_index_i
        data_dict_i["index_ooh"] = ooh_index_i
        data_dict_i["O_Ir_frac_ave"] = O_Ir_frac_ave_i
        data_dict_i["bulk_oxid_state"] = bulk_oxid_state_i
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

# #########################################################
df_ads_e = pd.DataFrame(data_dict_list)

df_ads_e = df_ads_e.set_index(
    ["crystal", "facet", "coverage", "termination", "active_site", ]
    )
# #########################################################

# +
# oer_data[8].keys()
# # oer_data[8]["calc"]
# oer_data[8]["atoms"]
# -

# ### Write atoms to file

if False:
    for name_i, row_i in df_ads_e.iterrows():

        atoms_bare_i = df_oer.loc[row_i.index_bare].atoms
        atoms_o_i = df_oer.loc[row_i.index_o].atoms
        atoms_oh_i = df_oer.loc[row_i.index_oh].atoms
        atoms_ooh_i = df_oer.loc[row_i.index_ooh].atoms


        dir_name = "_".join([str(i) for i in list(name_i)])
        # print("'" + name_i[1], sep="")
        # print(name_i[1], sep="")

        # print(name_i[4], sep="")

        dir_i = os.path.join(
            os.environ["PROJ_irox_oer"],
            "workflow/seoin_irox_data",
            "out_data/oer_sets",
            dir_name)
        if not os.path.exists(dir_i):
            os.makedirs(dir_i)

        atoms_bare_i.write(os.path.join(dir_i, "atoms_bare.traj"))
        atoms_o_i.write(os.path.join(dir_i, "atoms_o.traj"))
        atoms_oh_i.write(os.path.join(dir_i, "atoms_oh.traj"))
        atoms_ooh_i.write(os.path.join(dir_i, "atoms_ooh.traj"))

        # atoms_init_j.write(os.path.join(dir_i, "atoms_init.traj"))

df_ads_e_2 = pd.concat([
    df_ads_e,
    df_active_sites,
    ], axis=1)

# ### Writting data to file

# +
# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)

with open(os.path.join(directory, "df_ads_e.pickle"), "wb") as fle:
    pickle.dump(df_ads_e_2, fle)

with open(os.path.join(directory, "df_oer.pickle"), "wb") as fle:
    pickle.dump(df_oer, fle)
# #########################################################

# +
# df_ads_e_2


# -

assert False

# ### Write atoms to file

for j_ind, row_j in group.iterrows():
    # #################################################
    loc_short_j = row_j.loc_short
    atoms_j = row_j.atoms
    atoms_init_j = row_j.atoms_init
    # #################################################

    dir_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/seoin_irox_data",
        "out_data",
        loc_short_j,
        )

    if not os.path.exists(dir_i):
        os.makedirs(dir_i)

    atoms_j.write(os.path.join(dir_i, "atoms.traj"))
    atoms_init_j.write(os.path.join(dir_i, "atoms_init.traj"))

assert False

# + active=""
#
#
#

# +
# import plotly.express as px
# df = px.data.tips()
# fig = px.histogram(
#     df_ads_e_2,
#     x="O_Ir_frac_ave",
#     nbins=100,
#     )

# fig.show()
