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

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

from IPython.display import display

import pandas as pd
pd.set_option("display.max_columns", None)
pd.options.display.max_colwidth = 20
# pd.set_option('display.max_rows', None)

# #########################################################
# from methods import get_df_jobs_paths
from methods import (
    get_df_jobs_paths,
    get_df_dft,
    get_df_job_ids,
    get_df_jobs,
    get_df_jobs_data,
    get_df_slab,
    get_df_slab_ids,
    get_df_jobs_data_clusters,
    get_df_jobs_anal,
    get_df_slabs_oh,
    get_df_init_slabs,
    get_df_magmoms,
    )

# +
import os
import sys

import copy
import shutil
from pathlib import Path
from contextlib import contextmanager

# import pickle; import os

import pickle
import  json

import pandas as pd
import numpy as np

from ase import io

import plotly.graph_objects as go

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis import local_env

# #########################################################
from misc_modules.pandas_methods import drop_columns
# -

from methods import get_other_job_ids_in_set

# # Read data objects with methods

df_dft = get_df_dft()
df_job_ids = get_df_job_ids()
df_jobs = get_df_jobs(exclude_wsl_paths=True)
df_jobs_data = get_df_jobs_data(exclude_wsl_paths=True)
df_jobs_data_clusters = get_df_jobs_data_clusters()
df_slab = get_df_slab()
df_slab_ids = get_df_slab_ids()
df_jobs_anal = get_df_jobs_anal()
df_jobs_paths = get_df_jobs_paths()
df_slabs_oh = get_df_slabs_oh()
df_init_slabs = get_df_init_slabs()
df_magmoms = get_df_magmoms()


def display_df(df, df_name, display_head=True, num_spaces=3):
    print(40 * "*")
    print(df_name)
    print("df_i.shape:", df_i.shape)
    print(40 * "*")

    if display_head:
        display(df.head())

    print(num_spaces * "\n")


df_list = [
    ("df_dft", df_dft),
    ("df_job_ids", df_job_ids),
    ("df_jobs", df_jobs),
    ("df_jobs_data", df_jobs_data),
    ("df_jobs_data_clusters", df_jobs_data_clusters),
    ("df_slab", df_slab),
    ("df_slab_ids", df_slab_ids),
    ("df_jobs_anal", df_jobs_anal),
    ("df_jobs_paths", df_jobs_paths),
    ("df_slabs_oh", df_slabs_oh),
    ("df_magmoms", df_magmoms),
    ]

for name_i, df_i in df_list:
    display_df(df_i, name_i)

# +
print("")
print("")

for name_i, df_i in df_list:
    display_df(
        df_i,
        name_i,
        display_head=False,
        num_spaces=0)

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# -

# # TEST TEST TEST TEST

df_jobs[
    (df_jobs.ads == "oh") & \
    (df_jobs.bulk_id == "81meck64ba") & \
    
    # (df_jobs.compenv == "nersc") & \
    [True for i in range(len(df_jobs))]
    ]

assert False

# +
# job_id = "gasusupo_45"
job_id = "pewehobe_99"

row_jobs = df_jobs.loc[job_id]
row_paths = df_jobs_paths.loc[job_id]

# row_paths.gdrive_path
row_paths.path_full

# +
df_jobs_i = get_other_job_ids_in_set(job_id, df_jobs=df_jobs)

# df_jobs_paths.loc[
df_jobs_data.loc[
    df_jobs_i.index    
    ]

# +
# def get_other_job_ids_in_set():
#     """
#     """
#     row_jobs = df_jobs.loc[job_id]

#     compenv_i = row_jobs.compenv
#     bulk_id_i = row_jobs.bulk_id
#     slab_id_i = row_jobs.slab_id
#     ads_i = row_jobs.ads
#     att_num_i = row_jobs.att_num

#     df_jobs_i = df_jobs[
#         (df_jobs.compenv == compenv_i) & \
#         (df_jobs.bulk_id == bulk_id_i) & \
#         (df_jobs.slab_id == slab_id_i) & \
#         (df_jobs.ads == ads_i) & \
#         (df_jobs.att_num == att_num_i) & \
#         [True for i in range(len(df_jobs))]
#         ]

#     return(df_jobs_i)
# -

assert False

# +
# # "b5cgvsb16w/111/oh/active_site__62/03_attempt/_05/out_data"

# df_jobs[
#     (df_jobs.bulk_id == "b5cgvsb16w") & \
#     (df_jobs.ads == "oh") & \
#     (df_jobs.active_site == 62.) & \
#     (df_jobs.att_num == 3) & \
#     [True for i in range(len(df_jobs))]
#     ]

# +
# job_id =  "wadifowe_41"

# df_jobs.loc[job_id]

# # df_jobs_paths.loc[job_id].gdrive_path
# -

assert False

# +
df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

var = "o"
df_jobs_anal_i = df_jobs_anal_i.query('ads == @var')

for i_cnt, (name_i, row_i) in enumerate(df_jobs_anal_i.iterrows()):
    tmp = 42

# #####################################################
job_id_max_i = row_i.job_id_max
# #####################################################

# #####################################################
row_paths_i = df_jobs_paths.loc[job_id_max_i]
# #####################################################
gdrive_path = row_paths_i.gdrive_path
# #####################################################

in_dir = os.path.join(
    os.environ["PROJ_irox_oer_gdrive"],
    gdrive_path,
    )

out_dir = os.path.join("__test__/completed_*O_slabs")
# -

gdrive_path

assert False

# +
# from methods import read_magmom_comp_data

# magmom_data_dict = read_magmom_comp_data()

# +
# #########################################################
import pickle; import os

# path_i = os.path.join(
#     os.environ[""],
#     "",
#     "")

path_i = "/home/raulf2012/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_OER/dft_workflow/job_analysis/compare_magmoms/out_data/magmom_comparison_data.pickle"


with open(path_i, "rb") as fle:
    magmom_data_dict = pickle.load(fle)
# #########################################################

# +
# magmom_data_dict.

# +
df_jobs.loc["genubeve_94"]

df_magmoms.loc["genubeve_94"]

# +
df_jobs_anal

groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_jobs_anal.groupby(groupby_cols)
for name_i, group in grouped:
    tmp = 42

# grouped.get_group
grouped.get_group(('slac', 'fagumoha_68', 62.0))

# +
compenv_i = "slac"
slab_id_i = "fagumoha_68"
active_site_i = 62.


df_jobs[
    (df_jobs.compenv == compenv_i) & \
    (df_jobs.slab_id == slab_id_i) & \
    (df_jobs.active_site == active_site_i) & \
    [True for i in range(len(df_jobs))]
    ]

# + active=""
#
#
#
#
# -

assert False

# + jupyter={"source_hidden": true}
job_ids = ['rawupuga_30', 'fubemoro_70', 'liluluhu_67', 'dinagiso_25']
for i_cnt, i in enumerate(df_jobs_paths.loc[job_ids].gdrive_path):

    dir_i = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"], i)

    path_i = os.path.join(
        dir_i, "final_with_calculator.traj")

    directory = "__temp__/temp_atoms_write"
    if not os.path.exists(directory): os.makedirs(directory)
    
    # #####################################################
    out_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow",
        "__temp__/temp_atoms_write")

    file_name_prefix = str(i_cnt).zfill(3)
    # out_file = os.path.join(out_path, str(i_cnt).zfill(3) + ".traj")
    out_file = os.path.join(out_path, file_name_prefix + ".traj")

    shutil.copyfile(
        path_i, out_file)

    # #####################################################
    atoms_i = io.read(path_i)
    atoms_i.write(
        os.path.join(
            out_path,
            file_name_prefix + ".cif"
            )
        )

# + jupyter={"source_hidden": true}
df_jobs_data.loc[
    job_ids
    ]

-585.723757 - -585.599615

# + jupyter={"source_hidden": true}
job_id = "tikuboli_46"

df_jobs_paths.loc[job_id].gdrive_path

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
job_id = "bomowoto_88"

df_jobs_paths.loc[job_id].gdrive_path

# + jupyter={"source_hidden": true}
job_ids =  [
    'wulumoha_81',
    'kefasowu_80',
    'supibepa_48',
    'kefadusu_22',
    'butapime_67',
    'lerosove_43',
    'kosivele_15',
    'bomowoto_88',
    'buwinalo_86',
    'tonipita_82',
    'lunosahi_89',
    'dirigufo_28',
    'dofusoba_54',
    'pufanime_49',
    'sebitubi_78',
    'betarara_84',
    'novadesu_38',
    'tuvavali_48',
    'mekififo_83',
    'weratovu_37',
    'mowavuna_20',
    'ronafaki_75',
    'higewedo_88',
    'wodurowa_29',
    'kilalawi_94',
    'kuvapabo_66',
    'dinibone_46',
    'kavuvuhe_72',
    'kusewage_44',
    'kufuwudi_46',
    ]


job_ids = ['kefasowu_80',
 'kefadusu_22',
 'butapime_67',
 'tonipita_82',
 'pufanime_49',
 'sebitubi_78',
 'betarara_84',
 'tuvavali_48',
 'mowavuna_20',
 'higewedo_88',
 'kilalawi_94',
 'kuvapabo_66',
 'dinibone_46']

job_ids = ['kefadusu_22',
 'butapime_67',
 'tonipita_82',
 'pufanime_49',
 'sebitubi_78',
 'betarara_84',
 'tuvavali_48',
 'kilalawi_94',
 'kuvapabo_66',
 'dinibone_46']

for job_id_i, row_i in df_jobs_paths.loc[job_ids].iterrows():
    gdrive_path_i = row_i.gdrive_path

    path_i = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        gdrive_path_i,
        "job.out")
    # print(path_i, "\\")

# + jupyter={"source_hidden": true}
# df_jobs_anal = df_jobs_anal.set_index("job_id_max") 
# df_jobs_anal.loc[job_ids]

df_jobs_data.loc[job_ids]

# + jupyter={"source_hidden": true}
# len(job_ids)
# 30
# + jupyter={"source_hidden": true}



# + jupyter={"source_hidden": true}
# df_jobs_paths.loc["wahumosa_87"].gdrive_path

# df_jobs_paths.loc["kefimeni_60"].gdrive_path

df_jobs_paths.loc["vuvibolo_50"].gdrive_path







# df_jobs.loc["kefadusu_22"]

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
df_jobs_anal[df_jobs_anal.job_id_max == "koduvaka_72"]

# + jupyter={"source_hidden": true}
df_jobs_data.loc[["koduvaka_72"]]


# + jupyter={"source_hidden": true}
df_jobs_paths.loc["koduvaka_72"].gdrive_path

df_jobs.loc["koduvaka_72"]

df_jobs[
    (df_jobs.compenv == "slac") & \
    (df_jobs.slab_id == "fagumoha_68") & \
    (df_jobs.ads == "oh") & \
    (df_jobs.active_site == 62.) & \
    (df_jobs.att_num == 0) & \
    [True for i in range(len(df_jobs))]
    ]

"nebokipa_96",
"hotefihe_55",
"koduvaka_72",

# + jupyter={"source_hidden": true}
df_jobs_anal[df_jobs_anal.job_id_max == "koduvaka_72"]

# + jupyter={"source_hidden": true}
df_jobs[
    (df_jobs.compenv == "slac") & \
    [True for i in range(len(df_jobs))]
    ]

# + jupyter={"source_hidden": true}
df_index = df_jobs_anal.index.to_frame()

df_index_i = df_index[
    (df_index.compenv == "slac") & \
    (df_index_i.ads == "oh") & \
    [True for i in range(len(df_index))]
    ]


df_jobs_anal.loc[
    df_index_i.index
    ]
# + jupyter={"source_hidden": true}



# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
frag_i = "b5cgvsb16w/111/oh/active_site__62/02_attempt"

for job_id_i, row_i in df_jobs_paths.iterrows():
    path_full_i = row_i.path_full

    if frag_i in path_full_i:
        print(job_id_i)

job_ids = [
    "lehowika_51",
    "gonahafo_24",
    "tawawobu_24",
    ]

df_jobs.loc[job_ids]
    
# df_jobs_paths.loc[job_ids].path_short.tolist()
# df_jobs_paths.loc[job_ids].path_rel_to_proj.tolist()
df_jobs_paths.loc[job_ids].gdrive_path.tolist()

# + jupyter={"source_hidden": true}
path_full_i

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
job_id = "hifevufa_15"
df_jobs_paths.loc[job_id].gdrive_path

# df_jobs.loc[job_id]

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
df_jobs[

(df_jobs.slab_id == "mubolemu_18") & \
(df_jobs.compenv == "nersc") & \
(df_jobs.ads == "bare") & \
(df_jobs.att_num == 1)
 
]

# + jupyter={"source_hidden": true}
df_jobs_paths.loc["gipiwata_19"].gdrive_path

# + jupyter={"source_hidden": true}
# df_jobs_data_clusters

for job_id_i, row_i in df_jobs_paths.iterrows():
    frag_i = "xu6ivyvfvf/131/oh/active_site__81/01_attempt/_01"
    if frag_i in row_i.path_full:
        tmp = 42
        # print(row_i)


"wipuhite_59"

# + jupyter={"source_hidden": true}
df_jobs_data.loc["wipuhite_59"]

# + jupyter={"source_hidden": true}
# df_jobs_data_clusters

df_jobs_data_clusters.loc[
    (df_jobs_data_clusters.compenv == "sherlock") & \
    (df_jobs_data_clusters.ads == "oh") & \
    (df_jobs_data_clusters.facet == "010") & \
    (df_jobs_data_clusters.slab_id == "fogalonu_46") & \
    [True for i in range(len(df_jobs_data_clusters))]
    ]

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
job_id_i = "fusoreva_23"
df_job_ids[df_job_ids.job_id == "fusoreva_23"]

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
# df_jobs_data.loc[["vurabamu_02"]]

# df_jobs_data.columns

# + jupyter={"source_hidden": true}
df_jobs_anal_i = df_jobs_anal

df_jobs_anal_i = df_jobs_anal_i[df_jobs_anal_i.job_completely_done == True]

# #########################################################
var = "o"
df_jobs_anal_o = df_jobs_anal_i.query('ads == @var')

var = "oh"
df_jobs_anal_oh = df_jobs_anal_i.query('ads == @var')

var = "bare"
df_jobs_anal_bare = df_jobs_anal_i.query('ads == @var')

# + jupyter={"source_hidden": true}
# df_jobs_anal_bare

# + jupyter={"source_hidden": true}
df_i = df_jobs_anal_oh.index.to_frame(index=False)
# df_i = df_jobs_anal_oh.index.to_frame(index=True)

grouped = df_i.groupby(["compenv", "slab_id", "ads", "active_site", ])
for name, group in grouped:
    print(40 * "*")

    # #####################################################
    df_index_o = df_jobs_anal_o.index.to_frame(index=False)
    df_i = df_index_o
    df_i = df_i[
        (df_i.compenv == name[0]) & \
        (df_i.slab_id == name[1]) & \
        (df_i.ads == "o")
        # (df_i. == name[2]) & \
        ]

    found_o = False
    if df_i.shape[0] > 0:
        found_o = True
        # print("Found an *O slab")

    # #####################################################
    df_index_bare = df_jobs_anal_bare.index.to_frame(index=False)
    df_i = df_index_bare
    df_i = df_i[
        (df_i.compenv == name[0]) & \
        (df_i.slab_id == name[1]) & \
        (df_i.ads == "bare") & \
        # (df_i. == name[2]) & \
        [True for i in range(len(df_i))]
        ]

    found_bare = False
    if df_i.shape[0] > 0:
        found_bare = True
        # print("Found an * slab")
    
    if found_o and found_bare:
        print(name)

    if not found_o:
        print("ISDJFIjidsojfisjdi")

    if not found_bare:
        print("Bare not finished:", name)

# + jupyter={"source_hidden": true}
df_jobs[
    (df_jobs.compenv == "slac") & \
    (df_jobs.slab_id == "mesufagi_19") & \
    (df_jobs.ads == "bare") & \
    (df_jobs.active_site == 33.) & \
    [True for i in range(len(df_jobs))]
    ]



df_jobs_paths.loc[
    "ronafaki_75"
    ].path_full

df_jobs_anal[df_jobs_anal.job_id_max == "ronafaki_75"]

df_jobs_data.loc[["ronafaki_75"]]

# + jupyter={"source_hidden": true}
# name
df_jobs_anal_i = df_jobs_anal

# df_jobs_anal_i

var = "bare"
df_jobs_anal_i = df_jobs_anal_i.query('ads == @var')

df_jobs_anal_i[df_jobs_anal_i.job_completely_done != True]

# + jupyter={"source_hidden": true}
# df_index_o.iloc[0:1]

# + jupyter={"source_hidden": true}
# df_index_bare

# + jupyter={"source_hidden": true}
# # mess_i = "temp"
# # assert df_i.shape[0] == 1, mess_i

# if df_i.shape[0] > 0:
#     print("Found an *O slab")

# + jupyter={"source_hidden": true}
for name_i, row_i in df_jobs_anal_oh.iterrows():

    # name_o = (name_i[0], name_i[1], "o", name_i[3], name_i[4], )
    name_o = (name_i[0], name_i[1], "o", "NaN", name_i[4], )
    name_bare = (name_i[0], name_i[1], "bare", name_i[3], name_i[4], )
    
    tmp = name_o in df_jobs_anal_bare.index.tolist()
    if tmp:
        print("IJDIFIS")
        break


    o_slab_avail = False
    if name_o in df_jobs_anal_o.index:
        o_slab_avail = True

    bare_slab_avail = False
    if name_bare in df_jobs_anal_bare.index:
        bare_slab_avail = True

    if bare_slab_avail and o_slab_avail:
        print("PIJDFIS")

# + jupyter={"source_hidden": true}
# name_i

# + jupyter={"source_hidden": true}
# df_jobs_anal_bare

# + jupyter={"source_hidden": true}
# name_i

# name_o

# df_jobs_anal_bare

# + jupyter={"source_hidden": true}
assert False

# + active=""
#
#
#
#
#
#
#
#
#

# + jupyter={"source_hidden": true}
df_jobs_anal_i = df_jobs_anal

var = "sherlock"
df_jobs_anal_i = df_jobs_anal_i.query('compenv == @var')

var = "putarude_21"
df_jobs_anal_i = df_jobs_anal_i.query('slab_id == @var')

df_jobs_anal_i

# + jupyter={"source_hidden": true}
df_jobs_anal[df_jobs_anal.job_completely_done == True]


df_jobs_paths.loc["vurabamu_02"].path_short

# + jupyter={"source_hidden": true}


# df_job_ids[df_job_ids.job_id == "bisidebi_63"]

# sherlock putarude_21	o	NaN	1	bisidebi_63

df_job_ids[
    (df_job_ids.compenv == "sherlock") & \
    (df_job_ids.ads == "o") & \
    (df_job_ids.slab_id == "putarude_21")
    # (df_job_ids.active_site == "NaN")
    ]

# + jupyter={"source_hidden": true}
# df_jobs.loc["nebubudo_77"]

# + jupyter={"source_hidden": true}
df_jobs_data[df_jobs_data.job_id == "nebubudo_77"]

df_jobs_paths.loc["nebubudo_77"].gdrive_path

# + jupyter={"source_hidden": true}
"nubifoki_79" in df_job_ids.job_id.tolist()

df_job_ids[df_job_ids.job_id == "nubifoki_79"]

# + jupyter={"source_hidden": true}
assert False
# -

# df_jobs_paths.loc["bokedolu_84"]

# + jupyter={"source_hidden": true}



# + jupyter={"source_hidden": true}
# df_jobs[    
# (df_jobs.compenv == "sherlock") & \
# (df_jobs.slab_id == "dekififo_82")
# ]

# df_jobs.loc["dekififo_82"]

# ('sherlock', 'dekififo_82', 0)

# + jupyter={"source_hidden": true}
# df_job_ids[
# #     (df_job_ids.compenv == "sherlock") & \
#     (df_job_ids.job_id == "dekififo_82")
#     ]

# "dekififo_82" in df_job_ids.job_id.tolist()



grouped = df_job_ids.groupby(["compenv", "bulk_id", "facet", "slab_id", "att_num", "rev_num", "ads", "active_site", ])
for name, group in grouped:
    tmp = 42

group

# + jupyter={"source_hidden": true}
df_jobs.shape

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
# df_jobs_anal

var = "mesufagi_19"
df_jobs_anal_i = df_jobs_anal.query('slab_id == @var')
df_jobs_anal_i

# + jupyter={"source_hidden": true}
# df_jobs_anal

var = "rakawavo_17"
df_jobs_anal.query('slab_id == @var')

# + jupyter={"source_hidden": true}
# df_jobs.loc["nasibuka_44"]

# df_jobs_data.loc["nasibuka_44"]

df_jobs

# + active=""
#
#
#
#
#
#
#
#
#
#
#

# + jupyter={"source_hidden": true}
# df_jobs_anal

var = "oh"
df_jobs_anal_i = df_jobs_anal.query('ads == @var')

df_jobs_anal_i[df_jobs_anal_i.job_completely_done == True]

# #########################################################
# #########################################################

# df_jobs_data.loc[[
#     "vomevase_19",
#     "siletibu_45",
#     ]]

# df_jobs_paths.loc[[
#     "vomevase_19",
#     "siletibu_45",
#     ]].path_full.tolist()

# + jupyter={"source_hidden": true}
df_jobs[
    (df_jobs.compenv == "slac") & \
    (df_jobs.slab_id == "mesufagi_19")
#     (df_jobs.active_site == 33)
    ]

# + jupyter={"source_hidden": true}
df_jobs_anal_i = df_jobs_anal

var = "slac"
df_jobs_anal_i.query('compenv == @var')

var = "mesufagi_19"
df_jobs_anal_i.query('slab_id == @var')

# var = 33
# df_jobs_anal_i.query('active_site == @var')

# + jupyter={"source_hidden": true}
df_jobs.loc["gawobenu_61"]

df_jobs_paths.loc["gawobenu_61"].gdrive_path

# + jupyter={"source_hidden": true}
# df_job_ids[
#     () & \
#     ]

# /scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/run_slabs/
# run_oh_covered/out_data/dft_jobs/7h7yns937p/101/oh/active_site__75/01_attempt/_01

df = df_job_ids
df = df[
    (df["compenv"] == "sherlock") &
    (df["bulk_id"] == "7h7yns937p") &
    (df["facet"] == "101") &
    (df["ads"] == "oh") &
    (df["active_site"] == 75.0) &
    (df["att_num"] == 1) &
    [True for i in range(len(df))]
    ]
job_id_i = df.job_id.iloc[0]
df

# + jupyter={"source_hidden": true}
# df_jobs_data[df_jobs_data.job_id == "bobenite_26"]
# df_jobs_data[df_jobs_data.job_id == job_id_i]

display(df_jobs_data[df_jobs_data.job_id == job_id_i])

# df_jobs_data
# df_jobs_anal[df_jobs_anal.job_id_max == "bobenite_26"]

# df_jobs_anal.loc[("sherlock", )]

# display(df_jobs_anal[df_jobs_anal.job_id_max == "bobenite_26"])
display(df_jobs_anal[df_jobs_anal.job_id_max == job_id_i])

# + jupyter={"source_hidden": true}
df_jobs_paths.loc[
    "voburula_03"
    ].gdrive_path

# df_jobs_anal[df_jobs_anal.job_id_max == "voburula_03"]

# + jupyter={"source_hidden": true}
df_job_ids[df_job_ids.job_id == "hutoruwa_20"]

# + jupyter={"source_hidden": true}
df_job_ids[df_job_ids.job_id == "didaveri_93"]

# + jupyter={"source_hidden": true}
df_job_ids.shape

# + jupyter={"source_hidden": true}
df_jobs.shape

# + jupyter={"source_hidden": true}
assert False

# + active=""
#
#
#
#
#
#

# + jupyter={"source_hidden": true}
frag_i = "lafa"
for i in df_jobs.job_id.tolist():
# for i in df_jobs.slab_id.tolist():
    if frag_i in i:
        print(i)
        
df_jobs.loc["lafamiwa_73"]

# + jupyter={"source_hidden": true}
df_jobs_paths.loc["lafamiwa_73"]

# + jupyter={"source_hidden": true}
df_jobs

# + jupyter={"source_hidden": true}
df_jobs.

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
df_jobs_i = df_jobs[
    (df_jobs.ads == "bare") & \
    # (df_jobs.compenv == "slac") & \
    # (df_jobs.slab_id == "kalisule_45") & \
    # (df_jobs.ads == "bare") & \
    [True for i in range(len(df_jobs))]
    ]

# for job_id_i in df_jobs_i.index:
#     tmp = 42
#     tmp = df_jobs_data.loc[[job_id_i]].shape
#     print(tmp)

# + jupyter={"source_hidden": true}
df_jobs_data_i = df_jobs_data.loc[
    df_jobs_i.index
    ]


# df_jobs_data_i.columns.tolist()

# Ignore these


all_cols = [
    "bulk_id",
    "slab_id",
    "job_id",
    "facet",
    "compenv",
    "att_num",
    "rev_num",
    "timed_out",
    "error",
    "error_type",
    "brmix_issue",
    "completed",
    "num_steps",
    "ediff_conv_reached",
    "num_scf_cycles",
    "N_tot",
    "frac_true",
    "frac_false",
    "run_start_time",
    "time_str",
    "run_time",
    "irr_kpts",
    "final_atoms",
    "magmoms",
    "ads_i",
    "unique_key",
    "job_state",
    ]


ignore_cols = [
    "unique_key",
    "ads_i",
    "magmoms",
    "final_atoms",
    "irr_kpts",
    "run_time",
    "time_str",
    "run_start_time",

    "job_id",
    "slab_id",
    "bulk_id",

    "att_num",
    "rev_num",
    "facet",

    "frac_false",
    
    # TEMP
    "completed",
    ]

# + jupyter={"source_hidden": true}
custom_cols = list(set(all_cols) - set(ignore_cols))

df_jobs_data_i = df_jobs_data_i[
    custom_cols
    ]

# + jupyter={"source_hidden": true}
from misc_modules.pandas_methods import reorder_df_columns


col_order_list = [
    "compenv",
    "facet",
    "att_num",
    "job_id",
    "slab_id",
    "bulk_id",
    "rev_num",


    "job_state",
    "completed",

    "N_tot",
    "num_steps",
    "num_scf_cycles",
    "frac_true",
    "frac_false",

    "error",
    "error_type",
    "brmix_issue",
    "ediff_conv_reached",
    "timed_out",
    ]

df_jobs_data_i = reorder_df_columns(col_order_list, df_jobs_data_i)

# + jupyter={"source_hidden": true}
df_jobs_data_i[df_jobs_data_i.num_steps == 0].shape
df_jobs_data_i

# + jupyter={"source_hidden": true}
"voburula_03"

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
grouped = df_jobs.groupby(["slab_id", "att_num"])
for name, group in grouped:
    print(name)
    if name[0] == "kalisule_45":
        break
    tmp = 42

group = group.sort_values("rev_num")

job_ids = group.job_id
# df_jobs_data.loc[job_ids]

# + jupyter={"source_hidden": true}
# df_jobs_data_clusters

df_jobs.slab_id.unique().tolist()

# + jupyter={"source_hidden": true}
# df_jobs_data_clusters

# df_jobs.path_job_root_w_att_rev.tolist()

# frag_i = "b19q9p6k72/101/01_attempt/_03"

# for job_id, row_i in df_jobs.iterrows():
#     path_i = row_i.path_job_root_w_att_rev
#     if frag_i in path_i:
#         print(job_id)

# + jupyter={"source_hidden": true}
df_jobs[df_jobs.ads == "bare"]

df_jobs_data.loc["bopimusi_43"]

# + jupyter={"source_hidden": true}
# job_id_i = "weritidu_20"

# #########################################################
# df_jobs.loc[job_id_i]

# df_jobs.path_rel_to_proj.tolist()

# df_jobs[df_jobs.compenv == "nersc"].shape

# + jupyter={"source_hidden": true}
# df_jobs_data_clusters.loc["weritidu_20"]

# df_jobs_data_clusters[df_jobs_data_clusters.slab_id == "solaleda_75"]

# + jupyter={"source_hidden": true}
# df_jobs_data[df_jobs_data.compenv == "sherlock"]

df_jobs[
    (df_jobs.bulk_id == "7h7yns937p") & \
    [True for i in range(len(df_jobs))]
    ]

df_jobs.loc["tuhiboni_73"]
df_jobs.loc["tuhiboni_73"].path_job_root

# df_jobs_data.loc["tuhiboni_73"]

# + jupyter={"source_hidden": true}
print(df_jobs_data.shape)
print(df_jobs.shape)

df_jobs_data[df_jobs_data.compenv == "sherlock"]

# + jupyter={"source_hidden": true}
# job_id_i = "kusenahi_90"
# print(df_jobs.loc[job_id_i].path_short.tolist()[0])
# print("")
# df_jobs_data.loc[job_id_i]

# + jupyter={"source_hidden": true}
# df_jobs_data

print(df_jobs_data_clusters.shape)
print(df_jobs.shape)

# + jupyter={"source_hidden": true}
# df_jobs_data
# b5cgvsb16w/111/01_attempt/_03
# df_jobs_i = 
# df_jobs.path_short
# df_jobs.path_short.tolist()
# + jupyter={"source_hidden": true}



# + jupyter={"source_hidden": true}
df_jobs_i = df_jobs[df_jobs.compenv == "slac"]

test_path = "b5cgvsb16w/111/01_attempt/_03"
# df_jobs_i

for job_id_i, row_i in df_jobs_i.iterrows():

    path_short_i = row_i.path_short
    if path_short_i == test_path:
        print(row_i)

# + jupyter={"source_hidden": true}
df_jobs_data.loc["bupidafo_75"].iloc[-1]

job_id = "pakehisi_06"

df_jobs_data_clusters.loc[[job_id]]
df_jobs_data.loc[[job_id]]
# + jupyter={"source_hidden": true}
df_jobs_paths.iloc[0].to_dict()
# -

for name_i, row_i in df_slabs_oh.iterrows():

    # #####################################################
    atoms_i = row_i.slab_oh
    magmoms_i = atoms_i.get_initial_magnetic_moments()
    # #####################################################

    # print(magmoms_i)
    # print("---")


