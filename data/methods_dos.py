#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

# | - Import Modules
import os

import re

import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
# __|



class PDOS_Plotting():
    """
    """

    # | - PDOS_Plotting ***********************************
    def __init__(self,
        data_file_dir=".",
        # out_data_file_dir="out_data",
        ):
        """
        """
        # | - __init__
        # Input attributes
        self.data_file_dir = data_file_dir

        # Attributes set during
        self.total_dos_df = None
        self.pdos_df = None
        self.band_gap_df = None
        self.ispin = None


        # System name
        data_file_dir = self.data_file_dir
        sys_name_i = data_file_dir.split("/")[-1]
        self.sys_name = sys_name_i


        self.out_data_file_dir = os.path.join("out_data", sys_name_i)

        self.read_INCAR()

        # self.create_out_data_dir()

        self.read_data_files()

        # __|

    def create_out_data_dir(self):
        """
        """
        # | - create_out_data_dir
        out_folder = self.out_data_file_dir
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        # __|

    def read_INCAR(self):
        """
        """
        # | - read_INCAR
        data_file_dir = self.data_file_dir

        # Check spin polarized calculations:
        try:
            incar_file = open("INCAR", "r")
            ispin = 1  # Non spin polarised calculations.
            for line in incar_file:
                if re.match("(.*)ISPIN(.*)2", line):
                    ispin = 2 # For spin polarised calculations.
        except:
            incar_file = open(data_file_dir + "/INCAR", "r")
            ispin = 1  # Non spin polarised calculations.
            for line in incar_file:
                if re.match("(.*)ISPIN(.*)2", line):
                    ispin = 2 # For spin polarised calculations.

        self.ispin = ispin
        # __|

    def read_data_files(self):
        """
        """
        #| - read_data_files
        data_file_dir = self.data_file_dir

        # var_dir = "rapiDOS_out"

        #| - Methods
        def get_root_pdos_dir(var_dir, var_file):
            """
            """
            #| - get_root_pdos_dir
            root_pdos_dir = Path(
                os.path.join(
                    data_file_dir,
                    var_dir,
                    var_file,
                    )
                )

            return(root_pdos_dir)
            #__|


        def read_data_file(filename):
            """
            """
            #| - read_data_file
            my_file_0 = Path(os.path.join(get_root_pdos_dir("", filename)))
            my_file_1 = Path(os.path.join(get_root_pdos_dir("rapiDOS_out", filename)))

            # print(my_file_1)

            if my_file_0.is_file():
                df = pd.read_csv(my_file_0)
            elif my_file_1.is_file():
                df = pd.read_csv(my_file_1)
            else:
                df = None
                print("Woops! 8ewhrgw7yqewfi")

            return(df)
            #__|

        # __|

        total_dos_df = read_data_file("TotalDOS.csv")
        pdos_df = read_data_file("PDOS.csv")
        band_gap_df = read_data_file("BandGap.csv")

        if total_dos_df is not None and \
                pdos_df is not None and \
            band_gap_df is not None:

            # print("THIS ONE")

            # band_gap = band_gap_df
            band_gap_lower = band_gap_df['Lower Band Gap'][0]
            band_gap_upper = band_gap_df['Upper Band Gap'][0]
            # print('Approx. Band Gap:',np.round(np.abs(band_gap['Band Gap'][0]),3), "eV")

            self.band_gap_lower = band_gap_lower
            self.band_gap_upper = band_gap_upper
            self.in_good_state = True

        else:
            self.in_good_state = False
            print("else here")


        self.total_dos_df = total_dos_df
        self.pdos_df = pdos_df
        self.band_gap_df = band_gap_df

        # __|

    def plot__total_dos(self):
        """
        """
        # | - plot__total_dos
        total_dos_df = self.total_dos_df
        pdos_df = self.pdos_df
        band_gap_df = self.band_gap_df

        band_gap_lower = self.band_gap_lower
        band_gap_upper = self.band_gap_upper
        ispin = self.ispin

        out_data_file_dir = self.out_data_file_dir

        fig = plt.figure(figsize=(10.0,6.0)) # Create figure.
        plt.axvline(x=[0.0], color='k', linestyle='--',linewidth=1.2) # Plot vertical line in Fermi.

        # Plot DOS spin up.
        plt.plot(
            total_dos_df['Energy (E-Ef)'],
            total_dos_df['Total DOS Spin Up'],
            color='C3')

        # Fill between spin up and down.
        plt.fill_between(total_dos_df['Energy (E-Ef)'],
                         0, total_dos_df['Total DOS Spin Up'],
                         facecolor='C7', alpha=0.2, interpolate=True)

        plt.axvspan(band_gap_lower, band_gap_upper, alpha=0.2, color='C5')

        if ispin == 2:
            # Plot DOS spin down
            plt.plot(
                total_dos_df['Energy (E-Ef)'],
                -total_dos_df['Total DOS Spin Down'],
                color='C4')

            # Fill between spin up and down.
            plt.fill_between(total_dos_df['Energy (E-Ef)'],
                         0, -total_dos_df['Total DOS Spin Up'],
                         facecolor='C7', alpha=0.2, interpolate=True)

        plt.legend() # Add legend to the plot.
        plt.xlabel('E - Ef (eV)') # x axis label.
        plt.ylabel('DOS (states/eV)') # x axis label.
        plt.xlim([-8.0,4.0]) # Plot limits.

        # fig.savefig(out_folder + "/" + "Fig1.pdf") # Save figure EPS.
        fig.savefig(out_data_file_dir + "/" + "Fig1.pdf") # Save figure EPS.
        # __|

    def plot__total_pz_dz2(self):
        """
        """
        # | - plot__total_pz_dz2
        total_dos_df = self.total_dos_df
        pdos_df = self.pdos_df
        band_gap_df = self.band_gap_df

        band_gap_lower = self.band_gap_lower
        band_gap_upper = self.band_gap_upper
        ispin = self.ispin

        out_data_file_dir = self.out_data_file_dir


        fig = plt.figure(figsize=(10.0,6.0)) # Create figure.
        pdos_energy_index_df = pdos_df.set_index(['Energy (E-Ef)']) # Set index.
        # Sum same orbitals for all atoms:
        sum_orbitals_df = pdos_energy_index_df.groupby(pdos_energy_index_df.index).sum()

        plt.axvline(x=[0.0], color='k', linestyle='--',linewidth=1.2) # Plot vertical line in Fermi.

        # Spin up.
        plt.plot(sum_orbitals_df['pz_up'],color='C3')
        plt.plot(sum_orbitals_df['dz2_up'],color='C8')

        # Spin down.
        if ispin == 2:
            plt.plot(sum_orbitals_df['pz_down'],color='C3')
            plt.plot(sum_orbitals_df['dz2_down'],color='C8')

        plt.legend() # Add legend to the plot.
        plt.xlabel('E - Ef (eV)') # x axis label.
        plt.ylabel('DOS (states/eV)') # x axis label.
        plt.xlim([-8.0,4.0]) # Plot limits.

        fig.savefig(out_data_file_dir + "/" + "Fig2.pdf") # Save figure EPS.
        # __|

    def plot__spd(self):
        """
        """
        # | - plot__spd
        total_dos_df = self.total_dos_df
        pdos_df = self.pdos_df
        band_gap_df = self.band_gap_df

        band_gap_lower = self.band_gap_lower
        band_gap_upper = self.band_gap_upper
        ispin = self.ispin

        out_data_file_dir = self.out_data_file_dir


        pdos_energy_index_df = pdos_df.set_index(['Energy (E-Ef)']) # Set index.
        # Sum same orbitals for all atoms:
        sum_orbitals_df = pdos_energy_index_df.groupby(pdos_energy_index_df.index).sum()
        # Sum of orbitals for spin up:
        sum_orbitals_df['Total p_up'] = sum_orbitals_df.apply(lambda row: row.px_up + row.py_up + row.pz_up, axis=1)
        sum_orbitals_df['Total d_up'] = sum_orbitals_df.apply(lambda row: row.dxy_up + row.dyz_up + row.dxz_up + row.dz2_up + row.dx2_up, axis=1)


        # Sum of orbitals for spin up:
        if ispin == 2:
            sum_orbitals_df['Total p_down'] = sum_orbitals_df.apply(lambda row: row.px_down + row.py_down + row.pz_down, axis=1)
            sum_orbitals_df['Total d_down'] = sum_orbitals_df.apply(lambda row: row.dxy_down + row.dyz_down + row.dxz_down + row.dz2_down + row.dx2_down, axis=1)

        # Plots:
        fig = plt.figure(figsize=(10.0,6.0)) # Create figure.
        plt.axvline(x=[0.0], color='k', linestyle='--',linewidth=1.2) # Plot vertical line in Fermi.

        # Spin up:
        plt.plot(sum_orbitals_df['s_up'],color='C1')
        plt.plot(sum_orbitals_df['Total p_up'],color='C3')
        plt.plot(sum_orbitals_df['Total d_up'],color='C8')

        # Spin down:
        if ispin == 2:
            plt.plot(sum_orbitals_df['s_down'],color='C1')
            plt.plot(sum_orbitals_df['Total p_down'],color='C3')
            plt.plot(sum_orbitals_df['Total d_down'],color='C8')

        plt.legend() # Add legend to the plot.
        plt.xlabel('E - Ef (eV)') # x axis label.
        plt.ylabel('DOS (states/eV)') # x axis label.
        plt.xlim([-8.0,4.0]) # Plot limits.
        fig.savefig(out_data_file_dir + "/" + "Fig3.pdf") # Save figure EPS.
        # __|

    def plot__atom(self, atom_selected):
        """
        """
        #| - plot__atom
        total_dos_df = self.total_dos_df
        pdos_df = self.pdos_df
        band_gap_df = self.band_gap_df

        band_gap_lower = self.band_gap_lower
        band_gap_upper = self.band_gap_upper
        ispin = self.ispin

        out_data_file_dir = self.out_data_file_dir


        list_of_atoms = list(reversed(pdos_df['Atom Label'].unique()))
        print('List of atoms: ', list_of_atoms)

        """ Select one atom from the previous list. Remember list_of_atoms[0] corresponds to Atom #1,
        list_of_atoms[1] to #2 ..."""

        atom_selected = list_of_atoms[1]  # This is equivalent to atom_selected = 'Cu2' in this example.

        pdos_energy_index_df = pdos_df.set_index(['Energy (E-Ef)']) # Set index.
        only_atom_df = pdos_energy_index_df[pdos_energy_index_df['Atom Label']==atom_selected] # Select only one atom (e.g Cu2).
        atom_spin_up_df = only_atom_df.filter(regex="up").sum(axis=1) # Filter, get all bands with spin up. Then, sum all orbitals.

        if ispin == 2:
            atom_spin_down_df = only_atom_df.filter(regex="down").sum(axis=1) # Filter, get all bands with spin down. Then, sum all orbitals.

        # Plots:
        fig = plt.figure(figsize=(10.0,6.0)) # Create figure.

        plt.plot(atom_spin_up_df,color='C1') # Atom spin up.

        if ispin == 2:
            plt.plot(atom_spin_down_df,color='C3') # Atom spin down.

        plt.axvline(x=[0.0], color='k', linestyle='--',linewidth=1.2) # Plot vertical line in Fermi.
        plt.legend(['Atom spin up']) # Add manually legend to the plot.
        if ispin == 2: plt.legend(['Atom spin up','Atom spin down']) # Add manually legend to the plot.
        plt.xlabel('E - Ef (eV)') # x axis label.
        plt.ylabel('DOS (states/eV)') # x axis label.
        plt.xlim([-8.0,4.0]) # Plot limits.
        fig.savefig(out_data_file_dir + "/" + "Fig4__atom_states.pdf") # Save figure EPS.
        # __|

    # __| *************************************************


def calc_band_center(df):
    """
    """
    #| - calc_band_center
    df = df[
#         (df.index > -10) & \
#         (df.index < 4)

        (df.index > -18) & \
        (df.index < 1.5)






#         (df.index > -10) & \
#         (df.index < 2)

#         (df.index > -8) & \
#         (df.index < 2)

#         (df.index > -6) & \
#         (df.index < 2)

#         (df.index > -4) & \
#         (df.index < 2)

#         (df.index > -2) & \
#         (df.index < 2)

        # #################################################
#         (df.index > -10) & \
#         (df.index < 0)

#         (df.index > -10) & \
#         (df.index < 2)

#         (df.index > -10) & \
#         (df.index < 4)

#         (df.index > -10) & \
#         (df.index < 6)

#         (df.index > -10) & \
#         (df.index < 8)
        ]

    pho_i = df.values
    eps = np.array(df.index.tolist())

    band_center_up = np.trapz(pho_i * eps, x=eps) / np.trapz(pho_i, x=eps)

    return(band_center_up)
    #__|

def process_PDOS(
    PDOS_i=None,
    # atom_name_list=None,
    ):
    """
    """
    #| - process_PDOS

    data_dict_i = dict()

    # #####################################################
    orbitals_to_plot = [
        "px_up",   "py_up",   "pz_up",
        "px_down", "py_down", "pz_down",
        ]

    df_xy = None
    df_band_centers = None
    was_processed = False

    if PDOS_i.in_good_state:
        was_processed = True

        pdos_df = PDOS_i.pdos_df
        pdos_df = pdos_df.set_index(["Energy (E-Ef)"])

        cols_to_keep = orbitals_to_plot + ["Atom Label"]
        df_i = pdos_df[cols_to_keep]

        data = []
        data_xy_dict = dict()
        grouped = df_i.groupby(["Atom Label"])
        for name, group in grouped:
            data_dict_i = dict()

            data_dict_i["name"] = name

            spin_up_cols = [i for i in group.columns.tolist() if "up" in i]
            spin_down_cols = [i for i in group.columns.tolist() if "down" in i]


            px_sum = abs(group.px_up) + abs(group.px_down)
            px_band_center = calc_band_center(px_sum)

            py_sum = abs(group.py_up) + abs(group.py_down)
            py_band_center = calc_band_center(py_sum)

            pz_sum = abs(group.pz_up) + abs(group.pz_down)
            pz_band_center = calc_band_center(pz_sum)

            # #################################################

            p_tot_sum = px_sum + py_sum + pz_sum
            p_tot_band_center = calc_band_center(p_tot_sum)


            data_dict_i["px_band_center"] = px_band_center
            data_dict_i["py_band_center"] = py_band_center
            data_dict_i["pz_band_center"] = pz_band_center
            data_dict_i["p_tot_band_center"] = p_tot_band_center

            data.append(data_dict_i)


            df_xy = pd.DataFrame()
            df_xy["px_sum"] = px_sum
            df_xy["py_sum"] = py_sum
            df_xy["pz_sum"] = pz_sum
            df_xy["p_tot_sum"] = p_tot_sum

            data_xy_dict[name] = df_xy



        df_xy = pd.concat(
            list(data_xy_dict.values()),
            keys=list(data_xy_dict.keys()),
            axis=1)


        df_band_centers = pd.DataFrame(data)

        # | - Processing df_band_centers, deconstruct 'name' property

        def method(row_i):
            # | - method
            # #####################################################
            new_column_values_dict = {
                "element": None,
                "atom_num": None,
                }
            # #####################################################
            name_i = row_i.get("name", None)
            # #####################################################


            # Checking that the first character is a letter
            assert name_i[0].isalpha(), "First character must be a letter (element type)"


            # #####################################################
            alpha_list = []
            numeric_list = []
            # #####################################################
            current_char_is_alpha = True
            # #####################################################
            for char_i in name_i:
                if char_i.isdigit():
                    current_char_is_alpha = False

                if current_char_is_alpha:
                    alpha_list.append(char_i)
                else:
                    numeric_list.append(char_i)

            element_str = "".join(alpha_list)
            atom_num = int("".join(numeric_list))

            # #####################################################
            new_column_values_dict["element"] = element_str
            new_column_values_dict["atom_num"] = atom_num
            # #####################################################
            for key, value in new_column_values_dict.items():
                row_i[key] = value
            return(row_i)
            # __|

        df_i = df_band_centers
        df_i = df_i.apply(
            method,
            axis=1,
            )
        df_band_centers = df_i
        # __|


    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["df_xy"] = df_xy
    out_dict["df_band_centers"] = df_band_centers
    out_dict["was_processed"] = was_processed
    # #####################################################
    return(out_dict)
    # #####################################################
    #__|
