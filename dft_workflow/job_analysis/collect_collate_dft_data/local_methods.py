"""
"""

#| - Import Modules
import os
import sys


sys.path.insert(0, os.path.join(
    os.environ["PROJ_irox"],
    "data"))

from oxr_reaction.oxr_methods import df_calc_adsorption_e

from energetics.dft_energy import Element_Refs

from proj_data_irox import (
    h2_ref,
    h2o_ref,
    )

from proj_data_irox import (
    corrections_dict,
    )

#__|


h2o_ref = h2o_ref
h2_ref = h2_ref

# h2o_ref = h2o_ref + h2o_corr
# h2_ref = h2_ref + h2_corr

Elem_Refs = Element_Refs(
    H2O_dict={
        "gibbs_e": h2o_ref,
        "electronic_e": h2o_ref,
        },

    H2_dict={
        "gibbs_e": h2_ref,
        "electronic_e": h2_ref,
        },
    )

oxy_ref, hyd_ref = Elem_Refs.calc_ref_energies()

oxy_ref = oxy_ref.gibbs_e
hyd_ref = hyd_ref.gibbs_e


def calc_ads_e(group):
    """Calculate species adsorption energy.

    Args:
        group
    """
    # | - calc_ads_e
    df_calc_adsorption_e(
        group,
        oxy_ref,
        hyd_ref,
        group[
            group["ads"] == "bare"
            ]["pot_e"].iloc[0],
        corrections_mode="corr_dict",
        corrections_dict=corrections_dict,
        adsorbate_key_var="ads",
        dft_energy_key_var="pot_e",
        )

    return(group)
    #__|


