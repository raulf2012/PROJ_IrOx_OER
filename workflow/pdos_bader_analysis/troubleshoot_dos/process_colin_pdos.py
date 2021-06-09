# -*- coding: utf-8 -*-
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

# +
import os
import numpy as np
import pandas as pd

# Read datafile from Colin's SI
colin_file_path = os.path.join(
    os.environ["dropbox"],
    "06_downloads/04_firefox_download",
    "1-s2.0-S003960281830760X-mmc1/SI_Data",
    "IrO2-1c-O.csv")
df = pd.read_csv(colin_file_path)

# Some processing
df = df.rename(columns={
    "Energy relative to fermi level (eV)": "energy",
    "O 2p-PDOS": "pdos_o_2p"})
df = df.set_index("energy")
df = df.drop(columns=["Total DOS", ])

# Taking only range of PDOS from -10 to 2 eV
df = df[
    (df.index > -10) & \
    (df.index < 2.)]

# Performing the trapezoidal rule
rho = df["pdos_o_2p"]
eps = np.array(df.index.tolist())

band_center_i = np.trapz(rho * eps, x=eps) / np.trapz(rho, x=eps)

print(
    "ϵ_2p: ",
    np.round(band_center_i, 4),
    " eV", sep="")

# OUTPUT: ϵ_2p: -3.7607 eV
# According to table in SI, this system (IrO2 C1) is supposed to have a e_2p of -2.44

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df

# + jupyter={"source_hidden": true}
# rho.shape

# + jupyter={"source_hidden": true}
# eps.shape

# + jupyter={"source_hidden": true}
# np.trapz(rho * eps, x=eps)

# + jupyter={"source_hidden": true}
# np.trapz(rho, x=eps)

# + jupyter={"source_hidden": true}
# np.min(
# df.index.tolist())

# np.max(
# df.index.tolist())
