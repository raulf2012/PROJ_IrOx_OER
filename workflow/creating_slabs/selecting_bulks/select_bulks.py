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

# # Analayze local coordination environment of bulk IrOx polymorphs
# ---
#
# This will determine which structures to select for further processing

# # Import Modules

# +
import os
print(os.getcwd())
import sys

from ase.db import connect
import pandas as pd

# #########################################################
from methods import get_df_dft

# +
# Contents will be saved to json
out_dict = dict()

directory = "out_data"
if not os.path.exists(directory):
    os.makedirs(directory)
# -

# # Read file

# +
FinalStructuresdb_file = os.path.join(
    os.environ["PROJ_irox_oer_data"],
    "active_learning_proj_data/FinalStructures_1.db")
db = connect(FinalStructuresdb_file)

data_list = []
for row in db.select():
    row_dict = dict(
        energy=row.get("energy"),
        **row.key_value_pairs)
    data_list.append(row_dict)
df = pd.DataFrame(data_list)

df = df[~df["stoich"].isna()]
df = df.set_index("structure_id")
df = df.drop(columns=["energy", "id_old", ])
# -

df_dft = get_df_dft()

# +
df_i = df[df.coor_env == "O:6"]

print("Number of octahedral AB2:", df_i[df_i.stoich == "AB2"].shape[0])
print("Number of octahedral AB3:", df_i[df_i.stoich == "AB3"].shape[0])

# +
df_dft_i = df_dft.loc[
    df_dft.index.intersection(
        df_i.index.tolist()
        )
    ]

out_dict["bulk_ids__octa_unique"] = df_dft_i.index.tolist()

df_dft_i.head()

# +
import plotly.express as px
fig = px.histogram(df_dft_i, x="num_atoms", nbins=20)

fig.update_layout(title="Number of atoms for unique octahedral IrOx bulk structures")
fig.show()

# +
from plotting.my_plotly import my_plotly_plot

my_plotly_plot(
    figure=fig,
    plot_name="atom_count_histogram_octahedral",
    write_html=True,
    write_png=False,
    png_scale=6.0,
    write_pdf=False,
    write_svg=False,
    try_orca_write=False,
    )
# -

# # Saving data

# #######################################################################
import json
data_path = os.path.join("out_data/data.json")
with open(data_path, "w") as fle:
    json.dump(out_dict, fle, indent=2)
# #######################################################################
