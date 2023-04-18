import numpy as np
import pandas as pd

element_list = [
    "H",    "D",    "He",    "C",    "N",   "O",    "F",    "P",
    "S",    "Cl",    "Li",    "Na",    "Mg",    "Ca",
    "Si",    "PAH",    "15N",    "13C",    "18O", "E-"
]

species_list = [
    "E-",    "HI",    "HII",    "Hm",    "HeI",
    "HeII",
    "HeIII",
    "CI",
    "CII",
    "CIII",
    "CIV",
    "CV",
    "CVI",
    "CVII",
    "Cm",
    "NI",
    "NII",
    "NIII",
    "NIV",
    "N",
    "NVI",
    "NVII",
    "NVIII",
    "OI",
    "OII",
    "OIII",
    "OIV",
    "OV",
    "OVI",
    "OVII",
    "OVIII",
    "OIX",
    "Om",
    "NeI",
    "NeII",
    "NeIII",
    "NeIV",
    "NeV",
    "NeVI",
    "NeVII",
    "NeVIII",
    "NeIX",
    "NeX",
    "NeXI",
    "MgI",
    "MgII",
    "MgIII",
    "MgIV",
    "MgV",
    "MgVI",
    "MgVII",
    "MgVIII",
    "MgIX",
    "MgX",
    "MgXI",
    "MgXII",
    "MgXIII",
    "SiI",
    "SiII",
    "SiIII",
    "SiIV",
    "SiV",
    "SiVI",
    "SiVII",
    "SiVIII",
    "SiIX",
    "SiX",
    "SiXI",
    "SiXII",
    "SiXIII",
    "SXIV",
    "SiXV",
    "SI",
    "SII",
    "SIII",
    "SIV",
    "SV",
    "SVI",
    "SVII",
    "SVIII",
    "SIX",
    "SX",
    "SXI",
    "SXII",
    "SXIII",
    "SXIV",
    "SXV",
    "SXVI",
    "SXVI",
    "CaI",
    "CaII",
    "CaIII",
    "CaIV",
    "CaV",
    "CaVI",
    "CaVII",
    "CaVIII",
    "CaIX",
    "CaX",
    "CaXI",
    "CaXII",
    "CaXIII",
    "CaXIV",
    "CaXV",
    "CaXV",
    "CaXVII",
    "CaXVIII",
    "CaXIX",
    "CaXX",
    "CaXXI",
    "FeI",
    "FeII",
    "FeIII",
    "FeIV",
    "FeV",
    "FeVI",
    "FeVII",
    "FeVIII",
    "FeIX",
    "FeX",
    "FeI",
    "FeXII",
    "FeXIII",
    "FeXIV",
    "FeXV",
    "FeXVI",
    "FeXVII",
    "FeXVIII",
    "FeXIX",
    "FeXX",
    "FeXXI",
    "FeXXII",
    "FeXXIII",
    "FeXXIV",
    "FeXXV",
    "FeXXVI",
    "FeXXVII",
    "H2",
    "H2p",
    "H3p",
    "OH",
    "H2O",
    "C2",
    "O2",
    "HCOp",
    "CH",
    "CH2",
    "CH3p",
    "CO",
    "CHp",
    "CH2p",
    "OHp",
    "H2Op",
    "HOp",
    "COp",
    "HOCp",
    "O2p",
]

def get_elemental_abundances(abundance_array):
    df=pd.DataFrame(abundance_array)
    df.columns = species_list
    element_df=pd.DataFrame()
    for element in element_list:
        count=count_element(df.columns,element)
        abunds=df.apply(lambda x: (x.values*count.values).sum(),axis=1)
        element_df[element]=abunds
    return element_df

# def get_elemental_abundances(abundance_array):
#     """Assuming the species in species_list, converts the array of abundances from the GIZMO h5 file
#     into a dictionary of elemental abundances

#     Args:
#         abundance_array (h5 table): Abundances of species_list in columns, different times in rows

#     Returns:
#         dict: Elemental abundance of elements in element_list
#     """
#     df = pd.DataFrame(abundance_array)
#     df.columns = species_list
#     abundances = []
#     for element in element_list:
#         abundances.append(total_element_abundance(element, df))
#     return dict(zip(element_list, abundances))


def count_element(species_list, element):
    """
    Count the number of atoms of an element that appear in each of a list of species,
    return the array of counts

    :param  species_list: (iterable, str), list of species names
    :param element: (str), element

    :return: sums (ndarray) array where each element represents the number of atoms of the chemical element in the corresponding element of species_list
    """
    species_list = pd.Series(species_list)
    # confuse list contains elements whose symbols contain the target eg CL for C
    # We count both sets of species and remove the confuse list counts.
    confuse_list = [x for x in element_list if element in x]
    confuse_list = sorted(confuse_list, key=lambda x: len(x), reverse=True)
    confuse_list.remove(element)
    sums = species_list.str.count(element)
    for i in range(2, 10):
        sums += np.where(species_list.str.contains(element + f"{i:.0f}"), i - 1, 0)
    for spec in confuse_list:
        sums += np.where(species_list.str.contains(spec), -1, 0)
    return sums


def total_element_abundance(element, df):
    """
    Calculates that the total elemental abundance of a species as a function of time. Allows you to check conservation.

    :param element: (str) Element symbol. eg "C"
    :param df: (pandas dataframe) UCLCHEM output in format from `read_output_file`

    :return: Series containing the total abundance of an element at every time step of your output
    """
    sums = count_element(df.columns, element)
    return df.mul(sums.values, axis=1).sum(axis=1).mean()
