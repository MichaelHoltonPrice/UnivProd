import numpy as np
from pandas import read_csv
import pickle
import json
import os
import pandas as pd
import progressbar
import lmdb
import copy
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import optuna
from sklearn.linear_model import LinearRegression
import datetime

def fos_byte_string2list(fos_byte_string, num_fos):
    """ Convert a field of study (fos) byte string to a list.

    The input is a fos byte string; that is, it is an integer in byte string
    format that stores whether each paper contains each of the fields of study,
    of which there are num_fos (number of fields of study). The integer is
    converted to binary, where the leftmost bit is the least significant bit.
    Each bit indicates whether each fos is in the paper. This function returns
    a  list where each element if False if the corresponding bit is 0 and True
    if it is 1. This is best illustrated with an example showing the
    corresponding values of the fos byte string, bit representation, and list
    representation.

    fos_byte_string    b"6"
    have_fos_binary    01100...
    have_fos           [False,True,True,False,False...]

    Args:
        fos_byte_string: The field of study byte string to convert to a list.
        num_fos: Number of fields of study.

    Returns:
        A list of length num_fos containing True/False, indicating whether the
        paper contains each field of study.
    """

    # Create a format string for the conversion from int to bits
    format_string =  "{:0" + str(num_fos) + "b}"

    # Create have_fos, reversing the the result of formatting with [::-1] to
    # make the least significant bit leftmost rather than rightmost.
    have_fos = \
        [v == "1" for v in format_string.format(int(fos_byte_string))[::-1]]
    return have_fos

def list2fos_byte_string(have_fos):
    """ Convert a field of study (fos) list to a byte string

    Given a list that indicates whether a paper has a given field of study,
    return a byte string that encodes that last as integer. See the
    documentation for def fos_byte_string2list for full details.

    Args:
        have_fos: A list of containing True/False, indicating whether the
            paper contains each field of study.

    Returns:
        The field of study (fos) byte string
    """

    # Build the fos integer
    fos_int = 0
    for value in reversed(have_fos):
        if value:
            bit = 1
        else:
            bit = 0
        fos_int = (fos_int << 1) | bit

    # Convert the fos integer to a fos byte string
    fos_byte_string = str(fos_int)
    fos_byte_string = fos_byte_string.encode("ascii")
    return fos_byte_string

def update_fos_byte_string(new_fos_id,fos_byte_string, top_level_ids):
    """ Update a field of study (fos) byte string to so that it contains a new
    field of study.

    For details on the various representations see the documentation for
    fos_byte_string2list.

    As an example, consider the following set of inputs:

    new_fos_id       b"129"
    fos_byte_string  b"4"
    top_level_ids    [b"129",b"130",b"131"]

    The input fos byte string has a bit representation 001. The new fos ID is
    the first entry in top_level_ids, so the new bit representation is 101,
    which corresponds to the new byte string b"5" (the output).


    Args:
        new_fos_id: The field of study (fos) ID to add
        fos_byte_string: The starting fos byte string
        top_level_ids: A list of all the type level IDs (must include
            new_fos_id)

    Returns:
        The update field of study (fos) byte string
    """
    # Raise an exception if the new fos ID is not in the top level IDs
    if not new_fos_id in top_level_ids:
        raise ValueError("new_fos_id is not in top_level_ids")

    # Convert the input fos byte string to a list
    have_fos = fos_byte_string2list(fos_byte_string, len(top_level_ids))

    # Get the index of the new fos ID. Raise an exception if it is already in
    # have_fos
    index = top_level_ids.index(new_fos_id)
    if have_fos[index]:
        raise ValueError("Field of Study has already been added")

    # Add the new field of study to have_topic
    have_fos[index] = True

    # Convert from a list back to a byte string
    fos_byte_string = list2fos_byte_string(have_fos)

    return fos_byte_string

def affildata_byte2lists(s):
    """ For affiliation data, convert from a byte string to an equivalent set
    of lists.

    The input byte string, s, contains author / affiliation pairs separated
    by spaces, for which the author and affiliation IDs are separated by a
    dash. For example, consider a paper with three authors (b"1", b"2", and
    b"3") and three affiliations (b"A", b"B", and b"C"), for which author 1
    is affiliated with both A and B. The byte string representing this is:

    s = b"1-A 1-B 2-B 3-C"

    This function converts the preceding representation to two lists,
    authr_list and affil_list:

    authr_list = [b"1", b"1", b"2", b"3"]
    affil_list = [b"A", b"B", b"B", b"C"]

    Args:
        s: The byte string with the affiliation data

    Returns:
        authr_list A list of author IDs as byte strings
        affil_list A list of affiliation IDs as byte strings
    """

    # If the input is a blank string, return two empty lists
    if s == b"":
        return( (list(),list()) )

    # Decode the byte string so that it is a regular string
    s = s.decode()

    # Split into tokens, where each token is an author-affiliation pair
    # separated by a dash
    tokens = s.split(" ")

    # Use list comprehension to parse each author-affiliation pair into a list
    # with two elements (the author ID and affiliation ID).
    pair_data = [t.split('-') for t in tokens]

    # Extract the author and affiliation IDs from pair_data, and encode as
    # byte strings
    authr_list = [pd[0].encode("ascii") for pd in pair_data]
    affil_list = [pd[1].encode("ascii") for pd in pair_data]

    # Return the output lists
    return authr_list, affil_list

def affildata_lists2byte(authr_list, affil_list):
    """ For affiliation data, return the byte string representation of
    affiliations given the input author and affiliation ID lists.

    See the documentation for affildata_byte2lists for details on the two
    different representations. If the author and affiliation lists are

    authr_list = [b"1", b"1", b"2", b"3"]
    affil_list = [b"A", b"B", b"B", b"C"]

    then the return list is

    s = b"1-A 1-B 2-B 3-C"

    Args:
        authr_list A list of author IDs as byte strings
        affil_list A list of affiliation IDs as byte strings

    Returns:
        s: The byte string with the affiliation data
    """
    s = b" ".join([authr_list[n] + b'-' + affil_list[n]
                   for n in range(0, len(authr_list))])
    return(s)

def get_fractional_contributions(authr_list, affil_list, all_affil):
    """ Determine the fractional contribution of each affiliation to a paper

    The inputs are a list of author IDs and affiliation IDs, plus a list of
    all the affiliation IDs, all_affil (this third input is used to determine
    the index of each affiliation in all_affil).

    For each affiliation, determine the weighting to assign to each affiliation
    (its fractional contribution). Each author is assumed to contribute an
    equal amount, so each author has a total fractional contribution of 1
    divided by the number of authors. If an author has multiple affiliations,
    their contribution is allocated equally among those affilations.

    For example, consider the following author and affiliation lists:

    authr_list = [b"1", b"1", b"2", b"3"]
    affil_list = [b"A", b"B", b"B", b"C"]

    Author 1 contributes 1/3, but that contribution is divided across two
    affiliations (A and B). Author 2 contributes 1/3 to B and author 3
    contributes 1/3 to C. Hence, the fractional contributions are

    Affiliation A: 1/3 * 1/2       = 1/6
    Affiliation B: 1/3 * 1/2 + 1/3 = 3/6
    Affiliation C:             1/3 = 2+/6

    Aside form calculating these weights, the index in all_affil of the
    affiliations that make a non-zero contribution to the weights are returned
    as the first output, ind_affil.

    Args:
        authr_list A list of author IDs as byte strings
        affil_list A list of affiliation IDs as byte strings

    Returns:
        ind_affil: The indices in all_affil of the weights in weight_affil
        weigth_affil: The weights (fractional contributions) for each
            affiliation
    """

    # Identify the number of authors
    num_auth = len(set(authr_list))

    # Intialize the index and weight lists
    ind_affil = list()
    weight_affil = list()

    # Iterate over entries
    for n in range(0, len(authr_list)):
        # Get the number of affiliations for this author, then calculate the
        # weight this entry contributs
        num_affil_for_auth = np.sum([a == authr_list[n] for a in authr_list])
        weight = 1/num_auth/num_affil_for_auth
        ind = all_affil.index(affil_list[n])

        # If this affiliation has hat to be encountered, add it to the output
        # lists
        if not(ind in ind_affil):
            ind_affil.append(ind)
            weight_affil.append(0)

        # Add the weight to the output
        z = ind_affil.index(ind)
        weight_affil[z] = weight_affil[z] + weight

    return ind_affil, weight_affil

def save_prod_array(prod_array, file_path):
    """ Save a 3D production array to an .npz file

    Args:
        prod_array: The 3D production array to save
    Returns:
        prod_array: The production array
    """
    np.savez_compressed(file_path, prod_array)

def load_prod_array(file_path):
    """ Load a 3D production array from an .npz file

    Args:
        Ny: The number of years
        Na: The number of affiliations
    """
    dict_data = np.load(file_path)
    return(dict_data["arr_0"])

class UnivDataManager:
    """A class for managing and parsing the various university data sets
    Attributes:
        univ_data_dir: The path to the University_Data_Central folder.
        results_dir: The path to the results folder
        verbose: Should summary and status information be printed out?
        mag_mapping: A dataframe with information on mapping MAD IDs to IPEDS
            IDs
        starting_ipeds_ids: The full set of ipeds_ids in mag_mapping
        delta: A dataframe containing the Delta data
        cw: A dictionary of dataframes containing Crosswalk data, where the
            keys are years.
                    (1) years / self.mag_years [Ny]
        (2) all_affil.json / self.all_affil [length Na]
        (3) top_level_names.txt / self.top_level_names [length num_fos]
        (4) pap_prod_array.csv / self.pap_prod_array [Ny x Na x num_fos]
        (5) cit_prod_array.csv / self.cit_prod_array [Ny x Na x num_fos]

        mag_years: A list of years (length Ny)
        all_affil: A list of MAG affiliation IDs (length Na)
        top_level_names: A list of top-level MAG field names (length num_fos)
        pap_prod_array: Paper production array (Ny x Na x num_fos)
        cit_prod_array: Citation production array (Ny x Na x num_fos)
        chetty_college_level: Information for each college (or institution) in
            the Chetty dataset.
        chetty_longitudinal: Longitudinal (cohort-/year- dependent) Chetty data
            are years.
        chetty_ope_ids: The college level OPE IDs as a list
        screened_ipeds_ids: A set of IPEDS IDs screened to be one-to-one with
            MAG IDs and to remove potentially problematic entries
        stable_id_triplets: Triplets (ipeds_id / mag_id / ope_id6) for which
            an OPEID / IPEDS ID match has been made in the Crosswalk (and which
            do not change in any year).
        final_id_triplets: Triplets (ipeds_id / mag_id / ope_id6) for which the
            super_opeid associated with ope_id6 is acceptable.
    """

    def __init__(self, univ_data_dir: str, results_dir: str, verbose=False):
        """Initialize the UnivDataManager class.
        Call a set of "helper" methods to initialize the class.
        Args:
            data_dir: The path to the University_Data_Central folder
            results_dir: The path to the results directory with production data
            verbose: Should summary and status information be printed out?
            :type univ_data_dir: str
            :type results_dir: str
            :type verbose: bool
        """
        self.univ_data_dir = univ_data_dir
        self.results_dir = results_dir
        self.verbose = verbose

        # Load MAG ID matching data. This creates the following attributes:
        # self.mag_mapping
        # self.starting_ipeds_ids
        self.load_mag_id_matching_data()

        # Load Delta data. This creates the following attribute (a dataframe):
        # self.delta
        self.load_delta_data()

        # Load Crosswalk data. This creates the following attribute (a
        # dictionary with years as keys):
        # self.cw
        self.load_crosswalk_data()

        # Load MAG paper and citation production data. This creates the
        # following attributes:
        # self.mag_years
        # self.all_affil
        # self.top_level_names
        # self.pap_prod_array
        # self.cit_prod_array
        self.load_mag_prod_data()

        # Load MAG paper and citation production data. This creates the
        # following attributes:
        # self.chetty_college_level
        # self.chetty_longitudinal
        # self.chetty_ope_ids
        self.load_chetty_data()

        # Screen the IPEDS-MAG matches in self.mag_mapping to require all
        # matches to be one-to-one and not be (potentially) problematic based
        # on key fields. This creates the following attributes:
        # self.screened_ipeds_ids
        self.screen_ipeds_mag_matches()

        # The output of the preceding screening is a set of IPEDS IDs,
        # self.screened_ipeds_ids. In the next step, require that each IPEDS ID
        # be associated with a single OPE ID6 (overarching OPE ID) that is
        # stable across the Crosswalk years. This results in a triplet for each
        # "surviving" ipeds_id, [ipeds_id,mag_id,ope_id6]. These are stored
        # in the following attribute:
        # self.stable_id_triplets
        self.filter_for_stable_ope_id6()

        # The output of the preceding filtering is a list of triplets
        # (ipeds_id / mag_id / ope_id6). In the next step, require that each
        # ope_id6 in the triplet (a) does not have a corresponding -1 for its
        # super_opeid (which means the Chetty team could not reliably match
        # or incorporate it) and (b) does not have an associated Chetty
        # super group. This .creates a new list of triplet, stored in the
        # following attribute:
        # self.final_id_triplets
        self.filter_for_chetty_super_groups()


    def load_mag_id_matching_data(self):
        """Load mag ID matching data from file into self.mag_mapping
        """
        if self.verbose:
            print("Loading MAG ID matching data")
        file_name = \
            os.path.join(self.univ_data_dir,
                         "original_data",
                         "mag_id_matching",
                         "mag_matches_to_unitid_with_mag_title.xlsx")
        self.mag_mapping = \
            pd.read_excel(file_name,
                          dtype={'unitid': str, 'magID': str})

        self.starting_ipeds_ids = set(self.mag_mapping.unitid)

    def load_delta_data(self):
        """Load Delta data from file into the dataframe self.delta
        """
        if self.verbose:
            print("Loading Delta data")

        # The Delta directory
        delta_dir = os.path.join(self.univ_data_dir, "original_data", "delta")

        # There are two Delta .csv files that are combined into a single
        # dataframe: (1) delta_public_release_87_99.csv covers academic years
        # 1986/87 through 1998/99 and (2) delta_public_release_00_15.csv covers
        # academic years 1999/2000 through 2014/15. Create one large dataframe,
        # ensuring that the columns in the input files are all identical.

        # Read the .csv file for the older time period
        path_old = os.path.join(delta_dir, "delta_public_release_87_99.csv")
        delta_old = read_csv(path_old, sep="\t", dtype="str", encoding='latin1')

        path_new = os.path.join(delta_dir, "delta_public_release_00_15.csv")
        delta_new = read_csv(path_new, sep="\t", dtype="str", encoding='latin1')

        if delta_old.shape[1] != delta_new.shape[1]:
            raise Exception(
                "Delta dataframes do not have the same numbers of columns")

        if np.sum(delta_old.columns == delta_new.columns) != delta_old.shape[1]:
            raise Exception("Delta column names do not match")

        self.delta = delta_old.append(delta_new)

    def load_crosswalk_data(self):
        """Load crosswalk data from file into the dictionary self.cw[year]
        """
        # The Crosswalk directory
        cw_dir = os.path.join(self.univ_data_dir, "original_data",
                              "college_scorecard",
                              "CollegeScorecard_Raw_Data_01192021",
                              "Raw_Data_Files", "Crosswalks")
        # Note: In IPEDS / Delta, the year 2014 corresponds to the academic
        #       year 2013-2014. The same convention is adopted here.

        # A dictionary for the crosswalk data
        self.cw = dict()
        if self.verbose:
            print("Loading crosswalk data")
        for year in range(2000, 2018):
            if self.verbose:
                print(year)

            # The Crosswalk file for this year:
            cw_file = os.path.join(cw_dir, "CW" + str(year) + ".xlsx")
            self.cw[year] = pd.read_excel(cw_file,
                                          sheet_name="Crosswalk",
                                          dtype="str")
            # Check that there are exactly five columns that begin with "COA".
            # Some methods rely on this.
            candidate_columns = [col_name for col_name in self.cw[year]
                                 if col_name.startswith("COA")]
            if candidate_columns != ["COA" + str(n) for n in range(1, 6)]:
                raise ValueError("Unexpected COA column situation")

    def load_mag_prod_data(self):
        """Load MAG production data using self.results_dir
        Load the MAG production data, which consist of:
        (1) years / self.mag_years [Ny]
        (2) all_affil.json / self.all_affil [length Na]
        (3) top_level_names.txt / self.top_level_names [length num_fos]
        (4) pap_prod_array.csv / self.pap_prod_array [Ny x Na x num_fos]
        (5) cit_prod_array.csv / self.cit_prod_array [Ny x Na x num_fos]
        """
        if self.verbose:
            print("Loading MAG production data")

        # The productions
        self.years = json.load(open(os.path.join(self.results_dir,
                                                 "years.json")))
        self.years = [int(y) for y in self.years]
        Ny = len(self.years)

        self.all_affil = json.load(open(os.path.join(self.results_dir,
                                                 "all_affil.json")))
        Na = len(self.all_affil)

        with open(os.path.join(self.results_dir,
                               "top_level_names.txt"),
                  "r") as file_handle:
            self.top_level_names = [v.strip() for v in list(file_handle)]

        self.pap_prod_array = load_prod_array(
            os.path.join(self.results_dir,
                         "pap_prod_array.npz"))

        self.cit_prod_array = load_prod_array(
            os.path.join(self.results_dir,
                         "cit_prod_array.npz"))

    def load_chetty_data(self):
        """Load Chetty data using self.univ_data_dir
        Load the Chetty data, which consist of:
        (1) mrc_table3.csv / self.chetty_longitudinal
        (2) mrc_table11.csv / self.chetty_college_level
        That is, mrc_table3.csv provides the year- (cohort-) dependent
        longitudinal data and mrc_table11.csv provides the college level data,
        such as whether a super_opeid is used for the institution.
        """
        if self.verbose:
            print("Loading Chetty data")

        longi_file = os.path.join(self.univ_data_dir, "original_data",
                                  "chetty", "mrc_table3.csv")
        collg_file = os.path.join(self.univ_data_dir, "original_data",
                                  "chetty", "mrc_table11.csv")
        self.chetty_college_level = read_csv(collg_file)
        self.chetty_longitudinal = read_csv(longi_file)
        # For faster subsequent matching, create a list with the Chetty college
        # level OPE IDs
        self.chetty_ope_ids = self.chetty_college_level.opeid.to_list()

    def screen_ipeds_mag_matches(self):
        """Screen the IPEDS-MAG matches in self.mag_mapping
        A match is accepted if (a) wrong_match_corrected is True if wrongMatch
        is True, (b) remaining matches are one-to-one,  and (c) certain key
        fields identifying potential problems are all blank (see code comments
        for details).
        """
        if self.verbose:
            print("Screening IPEDS-MAG matches")

        # First, remove entries (rows) in self.mag_mapping if wrongMatch = 1
        # and wrong_match_corrected = 0. Use a for loop rather than, say, list
        # comprehension because of the peculiarities of how pandas deals with
        # missing values
        good_ind = list()
        for n in range(0, self.mag_mapping.shape[0]):
            # Filtration for stability has not yet been done
            ipeds_id = self.mag_mapping.unitid[n]
            if not pd.isna(self.mag_mapping.WrongMatch[n]):
                if self.mag_mapping.WrongMatch[n] == 1:
                    if not pd.isna(self.mag_mapping.wrong_match_corrected[n]):
                        if pd.isna(
                                self.mag_mapping.wrong_match_corrected[n]
                        ) == 1:
                            good_ind.append(n)
                    else:
                        good_ind.append(n)
                else:
                    good_ind.append(n)
            else:
                good_ind.append(n)

        # Iterate over remaining matches to
        # identify those with a one-to-one mapping (only one IPEDS ID per
        # MAG ID, and vice versa). Also exclude entries for which any of the
        # following six fields (range(4,10) of the columns) are not blank:
        #
        # IPEDS_has_Multiple_magID
        # WrongMatch
        # wrong_match_corrected
        # MagBlank
        # MagBlank Corrected
        # no_longer_in_IPEDS
        # need to subset here by good IDs
        subset_mapping = self.mag_mapping.iloc[good_ind, :]

        self.screened_ipeds_ids = set()
        mag_ids = list(set(subset_mapping.magID))
        for mag_id in mag_ids:
            # Require only one unitID corresponding to magID
            matches = [n for n, m in enumerate(subset_mapping.magID)
                       if m == mag_id]
            if len(matches) == 1:
                n = matches[0]
                ipeds_id = subset_mapping.iloc[n, 0]
                if np.sum(pd.isna(subset_mapping.iloc[n, range(4, 10)])) == 6:
                    # Not a special case. Add to dictionary.
                    # column 0 is ipeds_id (unitid)
                    self.screened_ipeds_ids.add(ipeds_id)
            else:
                # If there is not exactly one match, make sure that it has been
                # flagged in column 4, IPEDS_has_Multiple_magID
                if np.sum(
                        [subset_mapping.iloc[n, 4] == 1 for n in matches]) != \
                        len(matches):
                    raise Exception('Found an unhandled case')

    def filter_for_stable_ope_id6(self):
        """Filter self.screened_ipeds_ids to identify those with a stable,
        associated ope_id6.
        self.screened_ipeds_ids is the result of an initial screening of
        IPEDS-MAG ID pairs based only on the information in self.mag_mapping.
        Further filter these IPEDS IDs to based on information in the College
        Scorecard Crosswalk. In particular, for each year there must be exactly
        one associated OPE ID6. Further, this OPE ID6 must not change across
        years (though it acceptable for no match to be found in a given year
        or years). The result is, for each ID that survives the filtering, a
        triplet [ipeds_id,mag_id,ope_id6] that can be used to link Delta, MAG,
        and Chetty data. These triplets are stored in self.stable_id_triplets.
        There is one special case that must be handled: an unidentified IPEDS ID
        change for NYU School of medicine.
        """
        if self.verbose:
            print("Filtering for stable OPE ID6's")
        # For 2000-2004, NYU School of Medicine appears to have an
        # uncaught IPEDS ID change: 189158 -> 193900. Only the latter is in
        # self.screened_ipeds_ids. Copy the starting set of IPEDS IDs and
        # remove "193900".
        ipeds_ids_to_check = self.screened_ipeds_ids.copy()
        ipeds_ids_to_check.remove("193900")

        self.stable_id_triplets = list()
        for ipeds_id in ipeds_ids_to_check:
            # Match this ipeds_id to a single OPE ID6 that does not change
            # across years
            ope_id6_by_year = list()
            for year in self.cw.keys():
                matches = [n for n, x in
                           enumerate(list(self.cw[year]["IPEDSMatch"]))
                           if x == ipeds_id]
                ope_id8s = list(self.cw[year]["OPEID"].iloc[matches])
                ope_id6 = UnivDataManager.get_opeid6_id(ope_id8s)
                ope_id6_by_year.append(ope_id6)
            unique_matches = set([x for x in ope_id6_by_year if x is not None])

            if len(unique_matches) == 1:
                # Get the associated mag_id (making sure it is unique)
                ope_id6 = list(unique_matches)[0]
                mag_matches = [self.mag_mapping.magID[n] for n, x
                               in enumerate(self.mag_mapping.unitid)
                               if x == ipeds_id]
                if len(mag_matches) != 1:
                    raise Exception("Wrong number of MAG ID matches")
                mag_id = mag_matches[0]
                self.stable_id_triplets.append([ipeds_id, mag_id, ope_id6])

    def filter_for_chetty_super_groups(self):
        """Filter self.stable_id_triplets to identify and remove problematic OPE
        ID 6's.
        self.stable_id_triplets is the result of a call to
        self.filter_for_stable_ope_id6. Some of these triplets have an
        associated ope_id6 that is problematic for one of two reasons: the
        associated super_opeid is (a) -1 or (b) part of a Chetty super group.
        Create a new list, self.final_id_triplets, without these triplets.
        Aside from (a) and (b), the ope_id6 must be in the Chetty dataset.
        """
        if self.verbose:
            print("Filtering for Chetty super groups")
        self.final_id_triplets = list()
        for triplet in self.stable_id_triplets:
            mag_id, ipeds_id, ope_id6 = triplet
            if ope_id6 is None:
                raise ValueError("ope_id6 of triplet is None")

            chetty_ope_id = int(ope_id6[1:-2])
            if chetty_ope_id in self.chetty_ope_ids:
                index = self.chetty_ope_ids.index(chetty_ope_id)
                super_ope_id =\
                    self.chetty_college_level["super_opeid"].iloc[index]
                if super_ope_id != -1:
                    if not self.is_in_chetty_super_group(ope_id6):
                        self.final_id_triplets.append(triplet)

    @staticmethod
    def get_opeid6_id(ope_id8s):
        """Given an input set of ope_id8s, return the one that it is an OPE ID6
        (or None if there is none in the set).
        Args:
            ope_id8s: A set of length 8 strings
        Return:
            The OPE ID6 (as a length 8 string) or None
        Raises:
            ValueError: If there is more than one OPE ID6 in the input set.
        """
        matches = [UnivDataManager.is_opeid6(ope_id8) for ope_id8 in ope_id8s]

        if sum(matches) == 1:
            return ope_id8s[matches.index(True)]
        elif sum(matches) == 0:
            return None
        else:
            raise ValueError("There should be exactly 0 or 1 ope_id6s in"
                             "ope_id8s")

    @staticmethod
    def is_opeid6(ope_id8):
        """Is the input OPEID8 an OPEID6 (end with "00")?
        Args:
            ope_id8: A length 8 string
        Return:
            True or False
        """
        if len(ope_id8) != 8:
            raise ValueError("ope_id8 is not length 8")
        # return ope_id8[-2:] == "00"
        return ope_id8[0] == "0" and ope_id8[-2:] == "00"

    @staticmethod
    def accept_crosswalk_row(ipeds_match: str, coa1: str, coa2: str, coa3: str,
                             coa4: str, coa5: str) -> bool:
        """Is this crosswalk entry (row) accepted?
        An entry is accepted if "IPEDSMatch" is not "No match" and all the
        columns that begin with "COA" are "-2".
        Args:
            ipeds_match: Value of the "IPEDSMatch" field
            coa1: Value of the "COA1" field
            coa2: Value of the "COA2" field
            coa3: Value of the "COA3" field
            coa4: Value of the "COA4" field
            coa5: Value of the "COA5" field
        Returns:
            bool: True or False
        """
        if ipeds_match == "No match":
            return False
        if coa1 != "-2":
            return False
        if coa2 != "-2":
            return False
        if coa3 != "-2":
            return False
        if coa4 != "-2":
            return False
        if coa5 != "-2":
            return False
        return True

    def create_ipeds_id_to_mag_id(self):
        """Create the dictionary self.ipeds_id_to_mag_id.
        self.ipeds_id_to_mag_id maps IPEDS IDs onto MAG IDs. That is:
        mag_id = self.ipeds_id_to_mag_id[mag_Id]. It is merely a reversal of
        self.mag_id_to_ipeds_id.
        """

        if self.verbose:
            print("Creating ipeds_id_to_mag_id")

        self.ipeds_id_to_mag_id = dict()

        # Store the ipeds_ids in a set to check (below) that the mag-ipeds
        # mapping is one-to-one.
        ipeds_ids = set()
        for mag_id in self.mag_id_to_ipeds_id:
            ipeds_id = self.mag_id_to_ipeds_id[mag_id]
            ipeds_ids.add(ipeds_id)
            self.ipeds_id_to_mag_id[ipeds_id] = mag_id

        if len(ipeds_ids) != len(self.mag_id_to_ipeds_id):
            raise Exception("mag-ipeds mapping is not one-to-one")

    def create_mag_id_to_ope_ids(self):
        """Create the dictionary self.mag_id_to_ope_ids
        self.mag_id_to_ipeids_id maps MAG IDs onto IPEDS IDs. That is:
        ope_ids = self.mag_id_to_ope_ids[mag_Id][year]. Unlike with
        self.mag_id_to_ipeds_id, the mapping is not one-to-one. Rather, each
        magId can map to one or more ope IDs and is (potentially) year
        dependent.
        """

        if self.verbose:
            print("Creating mag_id_to_ope_ids")

        # Extract the mag IDs for which an "acceptable" one-to-one mapping
        # exists with ope IDs.
        mag_ids = list(self.mag_id_to_ipeds_id.keys())

        # Initialize the dictionary of dictionaries
        self.mag_id_to_ope_ids = dict()

        # Iterate over MAG IDs then years in the Crosswalk
        for mag_id in mag_ids:
            ipeds_id = self.mag_id_to_ipeds_id[mag_id]
            sub_dict = dict()  # Initiliaze the sub-dictionary
            for year in self.cw.keys():
                # Find matches to this ipeds_id in the Crosswalk for this year.
                matches = [n for n, x in enumerate(self.cw[year]["IPEDSMatch"])
                           if x == ipeds_id]
                # Initialize this year's entry in the sub_dict
                sub_dict[year] = set()

                # Iterate over matches to add OPE IDs (if necessary)
                if len(matches) > 0:
                    for n in matches:
                        # Get the value to add from this year's Crosswalk
                        ope_id_to_add = self.cw[year]["OPEID"][n]
                        sub_dict[year].add(ope_id_to_add)
            # Add the sub-dictionary (years are keys) to the main dictionary
            # magIDs are keys)
            self.mag_id_to_ope_ids[mag_id] = sub_dict

    def create_merged_production_data(self, output_file=None):
        """Create merged production data (and, optionally, write to file)
        For institution-year pairs for which a match can be established, add
        columns to the Delta data for MAG production and Chetty.
        Args:
            output_file: An optional output file (.csv)
        Returns:
            DataFrame: A dataframe of merged data
        """

        if self.verbose:
            print("Create merged production function dataframe")

        # Check that the production arrays sum to one in each year (this could
        # be checked when they are generated, but is currently not checked
        # then). The tolerance is 1e-6 for each value.
        check_tol = 1e-6
        pap_prod_by_year = [np.sum(self.pap_prod_array[y, :, :]) for y in
                            range(self.pap_prod_array.shape[0])]
        cit_prod_by_year = [np.sum(self.cit_prod_array[y, :, :]) for y in
                            range(self.cit_prod_array.shape[0])]

        for n_y, v_y in enumerate(pap_prod_by_year):
            if abs(v_y - float(round(v_y))) > check_tol:
                raise ValueError("Paper production failed check for " +
                                 "year = " + str(self.years[n_y]))

        for n_y, v_y in enumerate(cit_prod_by_year):
            if abs(v_y - float(round(v_y))) > check_tol:
                raise ValueError("Citation production failed check for " +
                                 "year = " + str(self.years[n_y]))

        # Copy self.delta
        delta_modified = self.delta.copy(deep=True)

        # Add blank production columns to delta_modified by iterating over
        # fields of interest
        all_fos = self.top_level_names.copy()
        all_fos.insert(0, "no_fos")
        for n_fos, fos in enumerate(all_fos):
            delta_modified.insert(delta_modified.shape[1],
                                  "mag_paper_production_" + fos,
                                  "", allow_duplicates=False)
            delta_modified.insert(delta_modified.shape[1],
                                  "mag_citation_production_" + fos,
                                  "", allow_duplicates=False)

        # Iterate over rows to add production data
        final_ipeds_ids = [x[0] for x in self.final_id_triplets]
        prog_bar = progressbar.ProgressBar(max_value=len(all_fos))
        for n_fos, fos in enumerate(all_fos):
            nprod = delta_modified.columns.get_loc("mag_paper_production_" +
                                                   fos)
            ncita = delta_modified.columns.get_loc("mag_citation_production_" +
                                                   fos)

            for r in range(0, delta_modified.shape[0]):
                year = int(delta_modified["academicyear"].iloc[r])
                if year in self.years:
                    ipeds_id = delta_modified["unitid"].iloc[r]
                    if ipeds_id in final_ipeds_ids:
                        index_in_final = final_ipeds_ids.index(ipeds_id)
                        mag_id = self.final_id_triplets[index_in_final][1]
                        affil_index_in_mag = self.all_affil.index(mag_id)
                        year_index_in_mag = self.years.index(year)
                        delta_modified.iloc[r, nprod] =\
                            self.pap_prod_array[year_index_in_mag,
                                                affil_index_in_mag,
                                                n_fos]
                        delta_modified.iloc[r, ncita] = \
                            self.cit_prod_array[year_index_in_mag,
                                                affil_index_in_mag,
                                                n_fos]
            prog_bar.update(n_fos+1)

        # Add Chetty columns. "iclevel" and "state" are common, so do not add
        # those (the Delta values are "accepted").
        columns_to_add = self.chetty_longitudinal.columns.to_list()
        columns_to_add.remove("iclevel")
        columns_to_add.remove("state")
        for chetty_column in columns_to_add:
            delta_modified.insert(delta_modified.shape[1],
                                  chetty_column,
                                  "", allow_duplicates=False)

        col_ind_chetty = [n for n in
                          range(delta_modified.shape[1] - len(columns_to_add),
                                delta_modified.shape[1])]
        print("Iterating over delta rows to add Chetty data")
        prog_bar = progressbar.ProgressBar(max_value=delta_modified.shape[0])
        final_ipeds_ids = [x[0] for x in self.final_id_triplets]
        for r in range(0, delta_modified.shape[0]):
            year = int(delta_modified["academicyear"].iloc[r])
            ipeds_id = delta_modified["unitid"].iloc[r]
            if year in self.years:
                if ipeds_id in final_ipeds_ids:
                    index_in_final = final_ipeds_ids.index(ipeds_id)
                    ope_id6 = self.final_id_triplets[index_in_final][2]
                    chetty_ope_id = int(ope_id6[1:-2])
                    if chetty_ope_id in self.chetty_ope_ids:
                        index = self.chetty_ope_ids.index(chetty_ope_id)
                        super_ope_id =\
                            self.chetty_college_level["super_opeid"].iloc[index]
                        cohort_year = year - 19
                        match = self.chetty_longitudinal[
                            (
                            self.chetty_longitudinal.super_opeid == super_ope_id
                            ) & (
                            self.chetty_longitudinal.cohort == cohort_year
                            )
                        ]

                        if match.shape[0] == 1:
                            # add

                            match.pop("iclevel")
                            match.pop("state")
                            for m, c in enumerate(col_ind_chetty):
                                delta_modified.iloc[r, c] = match.iloc[0, m]
                        elif match.shape[0] > 1:
                            raise Exception("More than 1 match for Chetty row")
            prog_bar.update(r+1)

        if output_file is not None:
            delta_modified.to_csv(output_file, index=False)
        return delta_modified

    def summarize_matching(self, ipeds_id, year):
        """Summarize matching for a given ipeds_id and year.
        Args:
            ipeds_id: The IPEDS ID
            year: The year
        """

        print("--------------------------------------------------------------")
        print("Input IPEDS ID: " + ipeds_id)
        print("Input year    : " + str(year))

        all_ipeds_ids = [x[0] for x in self.final_id_triplets]
        if not ipeds_id in all_ipeds_ids:
            print("No matching IPEDS ID")

        if not year in list(self.cw.keys()):
            print("No matching year")

        index = all_ipeds_ids.index(ipeds_id)
        mag_id = self.final_id_triplets[index][1]
        # (Length 8, but an OPE ID6)
        ope_id6 = self.final_id_triplets[index][2]

        print("MAG ID        : " + mag_id)
        print("OPE ID6       : " + ope_id6)

        # Link Delta data. While Delta codes institutional data using single
        # ipeds_ids that correspond to just the main, ope_id6 of a group, in
        # fact the data are usually for an entire institution, which consists
        # of additional ipeds_ids and ope_ids.
        matched_row = self.delta[(self.delta.academicyear == str(year)) &
                                 (self.delta.unitid == ipeds_id)]

        if matched_row.shape[0] != 1:
            raise Exception("ipeds_id should match exactly one row")

        # fte_count
        # ftall
        print("instname      : " + matched_row["instname"].iloc[0])
        print("FTE           : " + matched_row["fte_count"].iloc[0])

        # Link Chetty. Chetty uses shortened versions of the ope_id6 as an
        # identifier. In particular, it drops the leading "0" and trailing
        # "00", eliminates any remaining, leading 0s. For example:
        # Aveda Institute Covington
        # ope_id        ipeds_id
        # 02600900      160320
        #  26009 <--- Chetty OPE ID

        chetty_ope_id = str(int(ope_id6[1:-2]))
        if self.is_in_chetty_super_group(ope_id6):
            # This should not be true for triplets in self.final_id_triplets.
            print(ope_id6 + " is in a Chetty super group")
        else:
            print(ope_id6 + " is not in a Chetty super group")

        ope_id8_set = self.get_ope_id8_set(ope_id6, year)
        all_ope_id8s = list(ope_id8_set)
        all_ope_id8s.sort()

        ope_id8s_in_crosswalk = self.cw[year].OPEID.tolist()
        # Get the IPEDS IDs that match the full set of OPE IDs
        all_ipeds_ids = list()
        for oid in all_ope_id8s:
            index = ope_id8s_in_crosswalk.index(oid)
            all_ipeds_ids.append(self.cw[year]["IPEDSMatch"].iloc[index])

        for ope_id8 in all_ope_id8s:
            matched_row = self.cw[year][self.cw[year].OPEID == ope_id8]
            if matched_row.shape[0] != 1:
                raise Exception("ope_id8 should match exactly one row")
            print("ope_id8/PEPSLocname: " +
                  matched_row["OPEID"].iloc[0] +
                  "    " +
                  matched_row["PEPSLocname"].iloc[0])

        # An individual born in 1980 (cohort_year) modally enters college in
        # 1998. This is the academic year 1998-99, which Delta / IPEDS treat
        # code as year = 1999; we adopt the same convention. The publication
        # data are also for 1999. Hence, cohort_year = year - 19.
        cohort_year = year - 19

        chetty_ope_id = int(ope_id6[1:-2])
        index = self.chetty_ope_ids.index(chetty_ope_id)
        super_ope_id = self.chetty_college_level["super_opeid"].iloc[index]

        print("super_opeid   : " + str(super_ope_id))
        matches = self.chetty_longitudinal[
            (self.chetty_longitudinal.cohort == cohort_year) &
            (self.chetty_longitudinal.super_opeid == super_ope_id)
            ]

        if matches.shape[0] == 0:
            print("No Chetty match")
        elif matches.shape[0] == 1:
            print(matches[["count", "k_rank", "k_mean"]])
        else:
            raise Exception("More than one match for Chetty data")

        # Link MAG data
        i = self.years.index(year)
        j = self.all_affil.index(mag_id)
        all_fos = self.top_level_names.copy()
        all_fos.insert(0, "no_fos")
        for n_fos, fos in enumerate(all_fos):
            print(fos)
            pap_prod = self.pap_prod_array[i, j, n_fos]
            cit_prod = self.cit_prod_array[i, j, n_fos]
            print("Paper Prod.   : " + str(pap_prod))
            print("Cit.  Prod.   : " + str(cit_prod))

    def get_ope_id8_set(self, ope_id6, year):
        all_ope_id_main = self.cw[year]["OPEIDMain"].to_list()
        matches = [self.cw[year]["OPEID"].iloc[n]
                   for n, x in enumerate(all_ope_id_main)
                   if x == ope_id6]
        return set(matches)

    def is_in_chetty_super_group(self, ope_id):
        """Is this ope_id in a Chetty super group?
        Args:
            ope_id: An ope_id that is an OPE ID6
        Return:
            True or False (or None if the input ope_id is not found)
        Raises:
            Exception: If the super_opeid is -1
        """

        if not UnivDataManager.is_opeid6(ope_id):
            raise ValueError(f"ope_id = {ope_id} is not a valid OPE ID6")

        chetty_ope_id = int(ope_id[1:-2])

        # self.chetty_ope_ids was set to
        # self.chetty_college_level.opeid.to_list() by __init__ for faster
        # matching.
        if not chetty_ope_id in self.chetty_ope_ids:
            return None

        index = self.chetty_ope_ids.index(chetty_ope_id)

        if self.chetty_college_level["super_opeid"].iloc[index] == -1:
            raise Exception("super_opeid is -1")

        return self.chetty_college_level["super_opeid"].iloc[index]\
               < chetty_ope_id

    def summarize_filtering(self):
        print("Unique IPEDS IDs in starting IPEDS-MAG mapping")
        print(len(set(self.mag_mapping.unitid)))

        print("IPEDS IDs after screening mapping")
        print(len(self.screened_ipeds_ids))

        print("IPEDS IDs after ensuring Crosswalk stability")
        print(len(self.stable_id_triplets))

        print("IPEDS IDs after filtering for Chetty super groups")
        print(len(self.final_id_triplets))

def lmdb_to_dict(db_path):
    env = lmdb.open(db_path)
    txn = env.begin()
    length = txn.stat()['entries']
    txn = env.begin()
    cursor = txn.cursor()
    output_dict = dict()
    while cursor.next():
        output_dict[cursor.key()] = cursor.value()
    return output_dict

def clean_data(input_df_path,
               run_name,
               flag_tuition=True,
               flag_salary=False,
               flag_reliance=True,
               flag_faculty_ratio=True,
               flag_grad=True,
               flag_field_aggregate=False,
               flag_field_ratio=True,
               flag_field_citation_per_paper=True,
               flag_output_decompose=False,
               flag_citation_as_output=False,
               flag_only_r=False,
               start_year=None,
               agg_year=1,
               flag_imp_time=True,
               n_imp=2,
               drop_mandate_var=False,
               drop_other_var=False):
    """Clean data in preparation for neural network cost function fitting

    Args:
        input_df_path: Path to the input data frame,
            delta_with_MAG_and_chetty.csv
        run_name: Name of the "experiment"
        flag_tuition: whether to include tuition as an institutional
            characteristic
        flag_salary: whether to include faculty salary as an institutional
            characteristic
        flag_reliance: whether to include tuition reliance as an institutional
            characteristic
        flag_faculty_ratio: whether to include faculty-student ratio as an
            institutional characteristic
        flag_grad: whether to include master/doctoral degrees as an an
            institutional characteristic
        flag_field_aggregate: whether to aggregate fields into
            Humanities/Social science/Science
        flag_field_ratio: whether to include the field decomposition of
            publications as an institutional characteristic
        flag_field_citation_per_paper: whether to include citations per paper
            in each field as a quality variable
        flag_output_decompose: if True, use field-specific publications as
            a multi-dimensional output instead of total publication.
        flag_citation_as_output: if True, use citations as the output instead
            of publications
        flag_only_r: Whether to keep only Carnegie categories 15 and 16
            (Doctorate-granting universities with very high or high research
            activity)
        start_year: the year to start the dataset. If None, all the years are
           included.
        agg_year: if >1, use a rolling average of variables except for total
           cost.
        flag_imp_time: if True, impute missing years
        n_imp: maximum number of missing variables allowed to be missing and
            imputed by multiple-imputation method.
        drop_mandate_var: Whether to drop mandate variables
        drop_other_var: Whether to drop non-mandate control variables
    """

    # Make the results directory and add it to the Python path
    Path("./results/{}".format(run_name)).mkdir(parents=True,
                                                exist_ok=True)
    # Make the results/figures directory and add it to the Python path
    Path("./results/{}/figures".format(run_name)).mkdir(exist_ok=True,
                                                        parents=True)

    delta = pd.read_csv(input_df_path)

    # Remove Grand Canyon University (unitid = 104717). Remove it because its
    # sale in 2004 leads to a non-sensical entry for 2004 (2004 is the only
    # entry that "survives" subsequent cleaning)
    delta = delta[delta['instname'] != 'Grand Canyon University']

    # Extract research outputs (papers and citation) by field
    publications_by_field = [col for col in delta.keys()
                             if col.startswith('mag_paper_production_')]
    citations_by_field = [col for col in delta.keys()
                          if col.startswith('mag_citation_production_')]

    # Sum the per field production values to yield the total production
    delta['mag_paper_production'] = delta[publications_by_field].sum(axis=1)
    delta['mag_citation_production'] = delta[citations_by_field].sum(axis=1)

    # Calculate the citations per paper
    delta['citation_per_paper'] = \
        delta['mag_citation_production'] /delta['mag_paper_production']

    # If necessary, aggregate into "super-fields"
    if flag_field_aggregate:
        field_list = ['Humanities',
                      'Social science',
                      'Science and engineering']
        field_agg = [['Art', 'Philosophy', 'History'],
                     ['Business', 'Economics', 'Psychology',
                      'Political science', 'Sociology'],
                     ['Biology', 'Chemistry', 'Computer science', 'Engineering',
                      'Environmental science', 'Geography', 'Geology',
                      'Materials science', 'Mathematics']]
        publications_by_field = ['mag_citation_production_' + field
                                 for field in field_list]
        for i, field in enumerate(field_list):
            for f in field_agg[i]:
                pub_in_field = ['mag_paper_production_'+f]
                ci_in_field = ['mag_citation_production_'+f]
            delta['mag_paper_production_'+field] =\
                delta[pub_in_field].sum(axis=1)
            delta['mag_citation_production_'+field] = \
                delta[ci_in_field].sum(axis=1)
            if flag_field_citation_per_paper:
                delta['citation_per_paper_' + field] = \
                    (delta['mag_citation_production_'+field] /
                     delta['mag_paper_production_'+field]).fillna(0)
            if flag_field_ratio:
                delta['paper_ratio_'+field] = \
                    delta['mag_paper_production_'+field]\
                    /delta['mag_paper_production']
            delta = delta.drop(columns = pub_in_field+ci_in_field)

    else: # flag_field_aggregate is False
        field_list = []
        for col in publications_by_field:
            if col == 'mag_paper_production_no_fos':
                field = 'Other'
            else:
                field = col.split('_')[-1]
            field_list.append(field)
            if col == 'mag_paper_production_no_fos':
                if flag_field_citation_per_paper:
                    delta['citation_per_paper_' + field] =\
                        (delta['mag_citation_production_no_fos'] /
                         delta[col]).fillna(0)
            else:
                if flag_field_citation_per_paper:
                    delta['citation_per_paper_' + field] = \
                        (delta['mag_citation_production_'+field] /
                         delta[col]).fillna(0)
                if flag_field_ratio:
                    delta['paper_ratio_' + field] = \
                        delta[col]/delta['mag_paper_production']

    # Define control variables to be included
    control_vars = ['hbcu',
                    'hsi',
                    'medical',
                    'hospital',
                    'state',
                    'sector_revised',
                    'fed_grant_pct',
                    'state_grant_pct',
                    'inst_grant_pct',
                    'grad_rate_150_p',
                    'instr_sal_as_pct_instrtot',] + \
                   ['satmt25','satmt75','satvr25','satvr75'] +\
                   ['tuitionfee01_tf',
                    'tuitionfee02_tf',
                    'tuitionfee03_tf']*flag_tuition +\
                   ['ft_faculty_salary']*flag_salary +\
                   ['tuition_reliance_a1',
                    'govt_reliance_a']*flag_reliance +\
                   ['ft_faculty_per_100fte']*flag_faculty_ratio +\
                   ['paper_ratio_' + field for field
                    in (set(field_list)-set(['Other']))] * flag_field_ratio

    # Define student production variable, handling whether or not graduate
    # students count.
    output_vars = ['bachelordegrees' ] + \
                  ['masterdegrees', 'doctordegrees']*flag_grad

    # Define research production varaible, handling whether or not to decompose
    # by field and whether to use citations (as opposed to papers).
    if flag_output_decompose and (not flag_citation_as_output):
        output_vars = output_vars + publications_by_field
    elif (not flag_output_decompose) and flag_citation_as_output:
        output_vars = output_vars + ['mag_citation_production']
    elif flag_output_decompose and  flag_citation_as_output:
        output_vars = output_vars + citations_by_field
    else:
        output_vars = output_vars + ['mag_paper_production']

    quality_vars = ['k_median_nozero']
    if not flag_citation_as_output:
        if flag_field_citation_per_paper:
            quality_vars = quality_vars + ['citation_per_paper_' + field for field in field_list]
        else:
            quality_vars = quality_vars + ['citation_per_paper']


    # total01 is the total cost
    total_cost_var = ['total01']

    # Create a list of variables to be inflation-adjusted
    vars_inflation_adjust = total_cost_var + \
                            ['tuitionfee01_tf',
                             'tuitionfee02_tf',
                             'tuitionfee03_tf']*flag_tuition +\
                            ['ft_faculty_salary']*flag_salary


    # Subset and clean the dataframe
    # Keep only schools with sector_revised values between 1 and 3 (four year
    # colleges and universities)
    raw_data = delta.loc[(delta['sector_revised'] <= 3) &
                         (delta['sector_revised'] >= 1)]

    # Drop schools with too little output
    raw_data = raw_data[raw_data['mag_paper_production'] >= 5]
    raw_data = raw_data[raw_data['bachelordegrees'] >= 5]

    # If necessary, keep only Carnegie categories 15 and 16 (Doctorate-granting
    # universities with very high or high research activity)
    if flag_only_r:
        raw_data = raw_data[raw_data['carnegie2010'].isin([15, 16])]

    # mini_delta is the subset and cleaned delta data with the added production
    # columns
    mini_delta = raw_data[['cpi_scalar_2015',
                           'academicyear',
                           'unitid',
                           'instname'] +
                          total_cost_var+output_vars+quality_vars+control_vars]

    # Interpret missing as 0 for master and doctoral degrees
    if 'masterdegrees' in mini_delta.keys():
        mini_delta.loc[:, 'masterdegrees'] = \
            mini_delta['masterdegrees'].fillna(0)
    if 'doctordegrees' in mini_delta.keys():
        mini_delta.loc[:, 'doctordegrees'] =\
            mini_delta['doctordegrees'].fillna(0)

    # If ACT is reported but not SAT, impute SAT from ACT
    for s, a in zip(['satmt25', 'satmt75', 'satvr25', 'satvr75'],
                    ['actmt25', 'actmt75', 'acten25', 'acten75']):
        temp = raw_data[[s, a]].dropna()
        lr = LinearRegression().fit(temp[[a]], temp[s])
        idx_fill = raw_data[s].isna() & (~raw_data[a].isna())
        mini_delta.loc[idx_fill, s] = lr.predict(raw_data.loc[idx_fill, [a]])


    # For the presence of a hospital and medical school, interpret values
    # less than 0 as missing
    mini_delta.loc[mini_delta['hospital']<0,'hospital'] = np.nan
    mini_delta.loc[mini_delta['medical']<0,'medical'] = np.nan
    #(1,2)=(yes,no) in the data. Converting it to be (1,0)=(yes,no)
    mini_delta.loc[:,'hospital'] = 2 - mini_delta['hospital']
    #(1,2)=(yes,no) in the data. Converting it to be (1,0)=(yes,no)
    mini_delta.loc[:,'medical'] = 2 - mini_delta['medical']

    # Specify the start year
    if start_year !=None:
        mini_delta = mini_delta[mini_delta['academicyear'] >= start_year]
    start_year = mini_delta.academicyear.min()
    end_year = mini_delta.academicyear.max()

    # If necessary, do the time imputation
    if flag_imp_time:
        # Impute time series.
        df_list = []
        for unitid in mini_delta.unitid.unique():
            df_school = mini_delta[mini_delta['unitid'] == unitid]
            instname = df_school['instname'].iloc[0]
            df_school = df_school.set_index('academicyear').\
                reindex(pd.Index(np.arange(start_year, end_year+1),
                                 name='academicyear'))
            df_school.index = pd.to_datetime(df_school.index, format='%Y')
            df_school = df_school.interpolate(method='time',
                                              limit=2,
                                              limit_area='inside',
                                              downcast='infer').reset_index()
            df_school['academicyear'] = df_school['academicyear'].dt.year
            df_school['instname'] = instname
            df_list.append(df_school)

        mini_delta = pd.concat(df_list, axis=0,ignore_index=True)
        mini_delta['unitid'] = mini_delta['unitid'].astype(pd.Int64Dtype())

    #mini_delta = mini_delta[~mini_delta['hospital'].isna()]

    #Drop schools with missing cost data
    mini_delta = mini_delta[~mini_delta['total01'].isna()]

    # Drop schools with more than n_imp missing values. Missing value will be
    # imputed later.
    mini_delta = mini_delta.dropna(axis=0,thresh=mini_delta.shape[1]-n_imp)

    # Sector and State are dummy variables
    mini_delta = mini_delta.drop(columns='state').\
        join(pd.get_dummies(mini_delta['state'], drop_first=True))
    mini_delta = mini_delta.drop(columns='sector_revised').\
        join(pd.get_dummies(mini_delta['sector_revised'],
                            prefix='sector',
                            drop_first=True))

    # Account for inflation
    for var in vars_inflation_adjust:
        mini_delta[var] = mini_delta[var]/mini_delta['cpi_scalar_2015']

    # Apply logs to some variables
    mini_delta['log_tc'] = mini_delta[total_cost_var].apply(np.log10)
    for col in output_vars+quality_vars:
        if col == 'bachelordegrees':
            mini_delta['log_degrees'] = \
                (mini_delta['bachelordegrees']+1).apply(np.log10)
            mini_delta =\
                mini_delta.drop(columns=total_cost_var+['bachelordegrees'])
        if col == 'masterdegrees':
            mini_delta['log_master_degrees'] = \
                (mini_delta['masterdegrees']+1).apply(np.log10)
            mini_delta = mini_delta.drop(columns='masterdegrees')
        if col == 'doctordegrees':
            mini_delta['log_doctor_degrees'] =\
                (mini_delta['doctordegrees']+1).apply(np.log10)
            mini_delta = mini_delta.drop(columns='doctordegrees')
        if col == 'k_median_nozero':
            mini_delta['log_k_median_nozero'] =\
                (mini_delta['k_median_nozero']+1).apply(np.log10)
            mini_delta = mini_delta.drop(columns='k_median_nozero')
        if col == 'mag_paper_production':
            mini_delta['log_papers'] =\
                (mini_delta['mag_paper_production']+1).apply(np.log10)
            mini_delta = mini_delta.drop(columns=['mag_paper_production'])
        if col == 'mag_citation_production':
            mini_delta['log_citation'] = \
                (mini_delta['mag_citation_production']+1).apply(np.log10)
            mini_delta = mini_delta.drop(columns=['mag_citation_production'])
        if col == 'citation_per_paper':
            mini_delta['log_citation_per_paper'] =\
                (mini_delta['citation_per_paper']+1).apply(np.log10)
            mini_delta = mini_delta.drop(columns=['citation_per_paper'])
        if col.startswith('citation_per_paper_'):
            field = col.split('_')[-1]
            mini_delta['log_citation_per_paper_' + field] =\
                (mini_delta[col]+1).apply(np.log10)
            mini_delta = mini_delta.drop(columns=col)
        if col in citations_by_field:
            if col == 'mag_citation_production_no_fos':
                field = 'Other'
            else:
                field = col.split('_')[-1]
            mini_delta['log_citation_' + field] =\
                (mini_delta[col]+1).apply(np.log10)
            mini_delta = mini_delta.drop(columns=col)
        if col in publications_by_field:
            if col == 'mag_paper_production_no_fos':
                field = 'Other'
            else:
                field = col.split('_')[-1]
            mini_delta['log_papers_' + field] =\
                (mini_delta[col]+1).apply(np.log10)
            mini_delta = mini_delta.drop(columns=col)

    # Drop cpi_scalar_2015
    mini_delta = mini_delta.drop(columns=['cpi_scalar_2015'])

    # Impute the missing values.
    full_data_temp = copy.deepcopy(mini_delta.drop(columns=['academicyear',
                                                            'unitid',
                                                            'instname',
                                                            'log_tc']))
    imp = IterativeImputer(max_iter=5, random_state=0)
    full_data_imp = imp.fit_transform(full_data_temp)
    mini_delta_imp = pd.DataFrame(data=full_data_imp,
                                  columns=full_data_temp.keys())
    for v in ['academicyear', 'unitid', 'instname', 'log_tc']:
        mini_delta_imp[v] = mini_delta[v].values


    if agg_year!=1:
        # Impute time series.
        df_list = []
        for unitid in mini_delta_imp.unitid.unique():
            df_school = mini_delta_imp[mini_delta_imp['unitid'] == unitid]
            # Blance panel once
            df_school = df_school.set_index('academicyear').\
                reindex(pd.Index(np.arange(start_year, end_year+1),
                                 name='academicyear'))
            instname = df_school.instname.dropna().iloc[0]
            df_school = df_school.drop(columns = ['instname','unitid'])
            df_school_agg = df_school.rolling(window=agg_year,
                                              min_periods=agg_year).mean()
            df_school_agg.index = df_school_agg.index - (agg_year-1)

            df_school_agg_except_tc = df_school_agg.drop(columns=['log_tc']).\
                merge(df_school['log_tc'], left_index=True, right_index=True).\
                dropna().reset_index()
            df_school_agg_except_tc['unitid'] = unitid
            df_school_agg_except_tc['instname'] = instname

            df_list.append(df_school_agg_except_tc)

        mini_delta_imp = pd.concat(df_list, axis=0,ignore_index=True)

    # If necessary, remove variables
    # TODO: implement soft/hard mandate
    if drop_mandate_var or drop_other_var:
        # There are six categories of variables
        # log_tc          The thing we're predicting (total cost)
        # output          The output variables
        # obs_info        Information that specifies this observation
        #                 (academicyear, unitid, and instname)
        # output_quality  The quality variables for the outputs
        # mandate_control Control variables that imply a mission mandate
        # other_control   Control variables that do not imply a mission mandate
        #
        # The first four types of variables are never dropped. The last two
        # types of variables are optionally dropped.

        mandate_var = ["hbcu",    # historically black colleges and universities
                       "hsi",     # hispanic serving instiutions
                       "sector_2",# Private  nonprofit 4-year or above dummy
                       "sector_3"]# Private for-profit 4-year or above dummy
        kept_var = ["log_tc", "log_papers", "log_k_median_nozero",
                    'log_degrees', 'log_master_degrees', 'log_doctor_degrees',
                    "academicyear", "unitid", "instname"]
        # The research quality variables all begin with log_citation_per_paper
        for var in mini_delta_imp.columns:
            if var.startswith("log_citation_per_paper"):
                kept_var.append(var)

        # All other variables belong to other_control
        other_control = []
        for var in mini_delta_imp.columns:
            if (not var in kept_var) and (not var in mandate_var):
                other_control.append(var)

        # Check that all variables have been assigned
        print(kept_var)
        print(mandate_var)
        print(other_control)
        assert len(mini_delta_imp.columns ==
                   len(kept_var) + len(mandate_var) + len(other_control))

        # Now that we have defined all variables, handle the drops
        if drop_mandate_var:
            mini_delta_imp = mini_delta_imp.drop(columns=mandate_var)
        if drop_other_var:
            mini_delta_imp = mini_delta_imp.drop(columns=other_control)

    # Save the unnormalized data.
    assert mini_delta_imp.isnull().sum().sum() == 0
    mini_delta_imp.to_csv('results/{}/processed_data_full.csv'.
                          format(run_name), index=False)

def train_cost_nn(run_name):
    """Train the cost function neural network

    Args:
        run_name: Name of the "experiment"
    """

    # Set the seed
    torch.manual_seed(0)
    np.random.seed(0)

    '''
    0. Read and normalize data
    '''
    mini_delta_imp = pd.read_csv('results/{}/processed_data_full.csv'.format(run_name))

    # Normalize the data.
    full_data = copy.deepcopy(mini_delta_imp.drop(columns=['academicyear',
                                                           'unitid',
                                                           'instname']))

    scaler = StandardScaler()
    scaler.fit(full_data)
    with open('results/{}/full_data_scaler.pickle'.format(run_name),'wb') as f:
        pickle.dump(scaler, f)
    full_data[full_data.keys()] = scaler.transform(full_data)
    train_data, valid_data = train_test_split(full_data, test_size=.2)
    with open('results/{}/train_data_scaler.pickle'.format(run_name),'wb') as f:
        pickle.dump(train_data, f)
    with open('results/{}/valid_data_scaler.pickle'.format(run_name),'wb') as f:
        pickle.dump(valid_data, f)

    '''
    1. Train the neural network using architecture search by Optuna.
    '''
    DEVICE = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    BATCHSIZE = 256
    #DIR = os.getcwd()
    EPOCHS = 1000
    #LOG_INTERVAL = 20
    N_TRAIN_EXAMPLES = BATCHSIZE * 30
    N_VALID_EXAMPLES = BATCHSIZE * 10
    N_TRIAL = 500

    # define_model is inside train_cost_nn to use local variables
    def define_model(trial):
        # We optimize the number of layers, hidden units and dropout ratio in
        # each layer.
        n_layers = trial.suggest_int("n_layers", 2, 5)
        layers = []

        in_features = full_data.shape[1] - 1  # -1 for the target var.
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 16, 128)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.Sigmoid())
            p = trial.suggest_float("dropout_l{}".format(i), 0.0, 0.5)
            layers.append(nn.Dropout(p))
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))

        return nn.Sequential(*layers)

    # objective is inside train_cost_nn to use local variables
    def objective(trial):
        # Generate the model.
        model = define_model(trial).to(DEVICE)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # Get the data loader
        train_target = torch.tensor(train_data['log_tc'].
                                    values.astype(np.float32))
        train = torch.tensor(train_data.drop('log_tc', axis = 1).
                             values.astype(np.float32))
        train_tensor = torch.utils.data.TensorDataset(train, train_target)
        train_loader = torch.utils.data.DataLoader(dataset=train_tensor,
                                                   batch_size=BATCHSIZE,
                                                   shuffle=True)

        valid_target = torch.tensor(valid_data['log_tc'].
                                    values.astype(np.float32))
        valid = torch.tensor(valid_data.drop('log_tc', axis=1).
                             values.astype(np.float32))
        valid_tensor = torch.utils.data.TensorDataset(valid, valid_target)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_tensor,
                                                   batch_size=BATCHSIZE,
                                                   shuffle=True)

        # Training of the model.
        loss_func = nn.MSELoss()
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # Limiting training data for faster epochs.
                if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                    break

                data, target = data.view(data.size(0), -1).to(DEVICE),\
                               target.to(DEVICE)

                optimizer.zero_grad()
                output = model(data).squeeze()
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()

            # Validation of the model.
            model.eval()
            total_val_loss = 0.
            n_val = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):
                    # Limiting validation data.
                    if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                        break
                    data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                    output = model(data).squeeze()
                    # Get the index of the max log-probability.
                    total_val_loss +=loss_func(output, target) * data.size(0)
                    n_val += data.size(0)
            mse_val = total_val_loss/n_val
            trial.report(mse_val, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # return accuracy
        return mse_val

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIAL)#, timeout=600

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    best_trial = study.best_trial
    with open('results/{}/BestCostFunctionSpecification.pickle'.
                      format(run_name), 'wb') as f:
        pickle.dump(best_trial, f)

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    with open('results/{}/optuna_study.pickle'.format(run_name), 'wb') as f:
        pickle.dump(study, f)

    '''
    2. Train the best model with the full dataset.
    '''
    with open('results/{}/BestCostFunctionSpecification.pickle'.
                      format(run_name), 'rb') as f:
        best_trial = pickle.load(f)

    # train_on_full_sample is inside train_cost_nn to use local variables
    def train_on_full_sample(trial):
        model = define_model(trial).to(DEVICE)

        # Generate the optimizers.
        optimizer_name = trial.params['optimizer']
        lr = trial.params['lr']
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # Get the data loader
        target = torch.tensor(full_data['log_tc'].values.astype(np.float32))
        X = torch.tensor(full_data.drop('log_tc', axis=1).
                         values.astype(np.float32))
        full_tensor = torch.utils.data.TensorDataset(X, target)
        full_loader = torch.utils.data.DataLoader(dataset=full_tensor,
                                                  batch_size=BATCHSIZE,
                                                  shuffle=True)

        # Training of the model.
        loss_func = nn.MSELoss()
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx, (data, target) in enumerate(full_loader):
                # Limiting training data for faster epochs.
                if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                    break

                data, target = data.view(data.size(0), -1).to(DEVICE),\
                               target.to(DEVICE)

                optimizer.zero_grad()
                output = model(data).squeeze()
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()

        model.eval()
        return model

    best_model_trained = train_on_full_sample(best_trial)
    torch.save(best_model_trained, 'results/{}/best_model_trained.pt'.
               format(run_name))

    target = torch.tensor(full_data['log_tc'].
                          values.astype(np.float32)).to(DEVICE)
    X = torch.tensor(full_data.drop('log_tc', axis=1).values.
                     astype(np.float32)).to(DEVICE)
    epsilon = target - best_model_trained.forward(X).squeeze()
    eps_df = copy.deepcopy(mini_delta_imp[['academicyear',
                                           'unitid',
                                           'instname']])
    eps_df['epsilon'] = epsilon.detach().cpu().numpy()
    eps_df.to_csv('results/{}/epsilon.csv'.format(run_name))

def build_run_spec(run_settings):
    """
    Build a string summarizing a cost function specification given the input
    run_settings.

    Args:
        run_settings: The settings for this run

    Return:
        spec: The run specification
    """
    spec = '-salary' * run_settings["flag_salary"] + \
           '-faculty_ratio' * run_settings["flag_faculty_ratio"] + \
           '-grad' * run_settings["flag_grad"] + \
           '-field_aggregate' * run_settings["flag_field_aggregate"] + \
           '-field_ratio' * run_settings["flag_field_ratio"] + \
           '-field_citation_per_paper' * \
           run_settings["flag_field_citation_per_paper"] + \
           '-output_decompose' * run_settings["flag_output_decompose"] + \
           '-citation_as_output' * run_settings["flag_citation_as_output"] + \
           '-only_r' * run_settings["flag_only_r"] + \
           '-agg_year{}'.format(run_settings["agg_year"]) * \
           (run_settings["agg_year"] != 1) + \
           '-imp_time' * run_settings["flag_imp_time"] + \
           '-imp{}'.format(run_settings["n_imp"]) + \
           '-drop_mandate_var' * run_settings["drop_mandate_var"] + \
           '-drop_other_var' * run_settings["drop_other_var"]
    return spec

def extract_run_datetime(run_name):
    """ Given an input run_name, extract the datetime

    Args:
        run_name: The name for this run

    Returns:
        date_time: The date time
    """

    # Example run_name:
    # 2021-08-27-12-13-49-ratio-grad-field_ratio-field_citation_per_paper-imp2
    date_time = datetime.datetime.strptime("-".join(run_name.split("-")[0:6]),
                                           '%Y-%m-%d-%H-%M-%S')
    return(date_time)

def extract_run_spec(run_name):
    """ Given an input run_name, extract the run specification

    Args:
        run_name: The name for this run

    Returns:
        run_spec: The run specification
    """

    # Example run_name:
    # 2021-08-27-12-13-49-ratio-grad-field_ratio-field_citation_per_paper-imp2
    tokens = run_name.split("-")

    if len(tokens) <= 6:
        return None

    # Drop the first six entries, which are for the date time
    tokens = tokens[6:len(tokens)]
    # The leading "-" is part of the run_spec
    return "-" + "-".join(tokens)

def get_run_dir(run_settings):
    """
    Get the path the directory with saved cost function results given the
    input run_settings. The directory with the newest time stamp is returned.

    Args:
        run_settings: The settings for this run

    Returns:
        run_dir: The run directory
    """
    spec = build_run_spec(run_settings)
    contents = os.listdir("results")
    run_specs = [extract_run_spec(run_spec) for run_spec in contents]
    matches = [n for n, item in enumerate(run_specs) if item == spec]
    if len(matches) == 0:
        raise Exception("run_spec = " + spec + " not found")
    date_times = [extract_run_datetime(contents[n]) for n in matches]
    n_most_recent = matches[np.argmax(date_times)]
    return os.path.join("results", contents[n_most_recent])

def have_run(run_settings):
    """
    Determine whether a run has been done (based on whether a directory for it
    exists)

    Args:
        run_settings: The settings for this run

    Returns:
        have_run: Whether the run has been done
    """
    spec = build_run_spec(run_settings)
    contents = os.listdir("results")
    run_specs = [extract_run_spec(run_spec) for run_spec in contents]
    matches = [n for n, item in enumerate(run_specs) if item == spec]
    return len(matches) != 0

def define_best_model(run_dir):
    """
    Define the best model using the information that has been saved.

    Args:
        run_dir: The run directory with save files

    Returns:
        best_model: The best model (a neural network object,
            torch.nn.Sequential)
    """
    with open( os.path.join(run_dir,
                            "BestCostFunctionSpecification.pickle"),
               "rb") as f:
        best_trial = pickle.load(f)
    # We optimize the number of layers, hidden units and dropout ratio in
    # each layer.
    n_layers = best_trial.suggest_int("n_layers", 2, 5)
    layers = []

    with open(os.path.join(run_dir, "train_data_scaler.pickle"), "rb") as f:
        train_data = pickle.load(f)

    in_features = train_data.shape[1] - 1  # -1 for the target var.
    for i in range(n_layers):
        out_features = best_trial.suggest_int("n_units_l{}".format(i), 16, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.Sigmoid())
        p = best_trial.suggest_float("dropout_l{}".format(i), 0.0, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, 1))

    return nn.Sequential(*layers)

def train_on_training_data(run_dir):
    """
    Train the best architecture on just the training data

    Args:
        run_dir: The run directory with save files

    Returns:
        model: The trained model
    """
    DEVICE = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    BATCHSIZE = 256
    EPOCHS = 1000
    N_TRAIN_EXAMPLES = BATCHSIZE * 30
    #N_VALID_EXAMPLES = BATCHSIZE * 10
    #N_TRIAL = 500

    model = define_best_model(run_dir).to(DEVICE)

    with open( os.path.join(run_dir,
                            "BestCostFunctionSpecification.pickle"),
               "rb") as f:
        best_trial = pickle.load(f)

    with open(os.path.join(run_dir, "train_data_scaler.pickle"), "rb") as f:
        train_data = pickle.load(f)

    # Generate the optimizers.
    optimizer_name = best_trial.params['optimizer']
    lr = best_trial.params['lr']
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the data loader
    target = torch.tensor(train_data['log_tc'].values.astype(np.float32))
    X = torch.tensor(train_data.drop('log_tc', axis=1).
                     values.astype(np.float32))
    full_tensor = torch.utils.data.TensorDataset(X, target)
    full_loader = torch.utils.data.DataLoader(dataset=full_tensor,
                                              batch_size=BATCHSIZE,
                                              shuffle=True)

    # Training of the model.
    loss_func = nn.MSELoss()
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(full_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE),\
                           target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    return model

def do_trans_log_fit(run_dir):
    """
    Do a trans-log fit on the training data (and calculate the root mean
    square error, RMSE, of the validation data).

    Args:
        run_dir: The run directory with save files

    Returns:
        lin_reg: The linear regression object for the trans log fit
        var_names: The variable names associated with each independent variable
            in the trans log fit
        rmse: The root mean square error (RMSE) of the training data
        rmse_valid: The out-of-sample root mean square error (RMSE) of the
            validation data
    """
    # The equation to be fit is:
    #
    # log(C) = alphar * log(yr)   + alhpae * log(ye) +
    #          betar  * log(yr)^2 + betae  * log(ye)^2 +
    #          phi    * log(yr)*log(ye) + Z .* gamma + epsilon
    #
    # The preceding equation is linear in the parameters.

    # Load the data from file
    with open(os.path.join(run_dir, "train_data_scaler.pickle"), "rb") as f:
        train_data0 = pickle.load(f)
    with open(os.path.join(run_dir, "valid_data_scaler.pickle"), "rb") as f:
        valid_data0 = pickle.load(f)

    # Create the initial parameter vector
    train_data = train_data0.copy()
    valid_data = valid_data0.copy()
    y = train_data["log_tc"].to_numpy()
    yvalid = valid_data["log_tc"].to_numpy()
    X = train_data.copy()
    Xvalid = valid_data.copy()
    # Delete the y-variable (log_tc)
    del X["log_tc"]
    del Xvalid["log_tc"]
    # Add three additional variables (the quadratic terms)
    X["log_papers_sq"] = X["log_papers"] * X["log_papers"]
    Xvalid["log_papers_sq"] = Xvalid["log_papers"] * Xvalid["log_papers"]
    X["log_degrees_sq"] = X["log_degrees"] * X["log_degrees"]
    Xvalid["log_degrees_sq"] = Xvalid["log_degrees"] * Xvalid["log_degrees"]
    X["log_papers_log_degrees"] = X["log_papers"] * X["log_degrees"]
    Xvalid["log_papers_log_degrees"] = \
        Xvalid["log_papers"] * Xvalid["log_degrees"]

    # Save the variable names
    var_names = X.columns

    X = X.to_numpy()
    Xvalid = Xvalid.to_numpy()
    lin_reg = LinearRegression(fit_intercept=False).fit(X, y)
    # Extract the linear coefficients
    th = lin_reg.coef_
    # Calculate the predicted y-value
    y_pred = np.matmul(X, th)
    yvalid_pred = np.matmul(Xvalid, th)

    # Calculate the vector of validation residuals
    resid = y_pred - y
    resid_valid = yvalid_pred - yvalid
    rmse = np.sqrt(np.mean(resid * resid))
    rmse_valid = np.sqrt(np.mean(resid_valid * resid_valid))
    return lin_reg, var_names, rmse, rmse_valid, resid_valid


class DataObject:
    def __init__(self, run_name):
        self.run_name = run_name
        self.flag_field_aggregate = ('field_aggregate' in run_name)
        self.flag_field = any([(flag in run_name)
                               for flag in ['field_aggregate',
                                            'field_ratio',
                                            'field_citation_per_paper',
                                            'field_decompose']])
        self.flag_citation_as_output = ('citation_as_output' in run_name)
        self.flag_field_decompose = ('field_decompose' in run_name)
        self.flag_field_citation_per_paper = \
            ('field_citation_per_paper' in run_name)

        self.mini_delta = \
            pd.read_csv('results/{}/processed_data_full.csv'.format(run_name)).\
                set_index(['unitid','academicyear'])
        self.full_data =\
            copy.deepcopy(self.mini_delta.drop(columns=['instname']))

        self.epsilon_df = pd.read_csv('results/{}/epsilon.csv'.format(run_name))
        self.epsilon_df.index = self.mini_delta.index
        with open('results/{}/full_data_scaler.pickle'.format(run_name), 'rb') \
                as f:
            self.scaler = pickle.load(f)
        full_data_scaled = self.scaler.transform(self.full_data)
        delta_path = os.path.join("..",
                                 "..",
                                 "delta_with_paper_production",
                                 "2021_06_15_updated_results",
                                  "delta_with_MAG_and_chetty.csv")
        delta = pd.read_csv(delta_path)
        raw_data = delta.loc[delta['sector_revised'] <= 3]
        self.unitid_name_match = \
            self.mini_delta.groupby('unitid').first()['instname'].reset_index()

        self.data_all =\
            self.mini_delta.merge(raw_data,
                                  on=['unitid', 'academicyear'],
                                  suffixes=('', '_y')).\
                set_index(['unitid', 'academicyear'])
        self.data_all.drop(self.data_all.filter(regex='_y$').columns.tolist(),
                           axis=1,
                           inplace=True)

        self.citation_logged = \
            ('log_citation_per_paper' in self.full_data.keys())
        if 'log_citation_per_paper' in self.full_data.keys():
            self.citation_var = 'log_citation_per_paper'
            self.citation_var_label = 'log citation per paper'
        elif 'citation_per_paper' in self.full_data.keys():
            self.citation_var = 'citation_per_paper'
            self.citation_var_label = 'citation per paper'
        else:
            self.citation_var = None

        self.median_school = self.full_data.median()
        self.school_mean = self.mini_delta.groupby('unitid').mean()
        self.data_all_mean = self.data_all.groupby('unitid').mean()#.reset_index()
        self.data_all_mean = \
            self.data_all_mean.reset_index().\
                merge(self.unitid_name_match, on='unitid').set_index('unitid')

        self.ivy_plus_median = \
            self.full_data[self.data_all['tier']==1].median()
        self.elite_median = self.full_data[self.data_all['tier']==2].median()
        self.highly_selective_median =\
            self.full_data[self.data_all['tier'].isin([3,4])].median()
        self.selective_median = \
            self.full_data[self.data_all['tier'].isin([5,6])].median()
        self.nonselective_median = \
            self.full_data[~self.data_all['tier'].isin(range(7))].median()

        self.nonprofit_private_median = \
            self.full_data[self.data_all['sector_revised']==2].median()
        self.public_median = \
            self.full_data[self.data_all['sector_revised']==1].median()

        try:
            with open('results/{}/univ_mu.pickle'.format(run_name),'rb') as f:
                self.univ_mu = pickle.load(f).set_index(['unitid'])
            with open('results/{}/year_mu.pickle'.format(run_name),'rb') as f:
                self.year_mu = pickle.load(f).set_index(['academicyear'])
        except:
            school_list = self.epsilon_df['unitid'].unique()
            year_list = self.epsilon_df['academicyear'].unique()
            n_school = len(school_list)
            n_year = len(year_list)
            y = self.epsilon_df['epsilon']
            x = pd.concat((pd.get_dummies(self.epsilon_df['unitid'],
                                          drop_first=True),
                           pd.get_dummies(self.epsilon_df['academicyear'],
                                          drop_first=True)),
                          axis=1)
            lr = LinearRegression(fit_intercept=False)
            lr.fit(x, y)
            # The first entry for the unitid and academicyear dummies are
            # dropped, and thus implicitly zero (hence add 0. to the
            # beginning of each).
            mu_i = np.concatenate(([0.], lr.coef_[:n_school-1]))
            mu_t = np.concatenate(([0.], lr.coef_[n_school-1:]))
            self.univ_mu =\
                pd.DataFrame(data={'unitid': school_list,
                                   'mu_i': mu_i}).set_index(['unitid'])
            self.year_mu = \
                pd.DataFrame(data={'academicyear': year_list,
                                   'mu_t': mu_t}).set_index(['academicyear'])
            with open('results/{}/univ_mu.pickle'.format(run_name), 'wb') as f:
                pickle.dump(self.univ_mu, f)
            with open('results/{}/year_mu.pickle'.format(run_name), 'wb') as f:
                pickle.dump(self.year_mu, f)

        self.data_all = self.data_all.merge(self.univ_mu, on=['unitid'])

        self.median_mu = self.data_all.groupby('unitid').mean().median()['mu_i']
        self.nonprofit_private_median_mu = \
            self.data_all[self.data_all['sector_revised'] == 2].\
                groupby('unitid').mean().median()['mu_i']
        self.public_median_mu = \
            self.data_all[self.data_all['sector_revised'] == 1].\
                groupby('unitid').mean().median()['mu_i']

        self.ivy_plus_median_mu = \
            self.data_all[self.data_all['tier'] == 1].\
                groupby('unitid').mean().median()['mu_i']
        self.elite_median_mu = self.data_all[self.data_all['tier'] == 2].\
            groupby('unitid').mean().median()['mu_i']
        self.highly_selective_median_mu = \
            self.data_all[self.data_all['tier'].isin([3, 4])].\
                groupby('unitid').mean().median()['mu_i']
        self.selective_median_mu = \
            self.data_all[self.data_all['tier'].isin([5, 6])].\
                groupby('unitid').mean().median()['mu_i']
        self.nonselective_median_mu =\
            self.data_all[~self.data_all['tier'].isin(range(7))].\
                groupby('unitid').mean().median()['mu_i']

        if self.flag_field_aggregate:
            self.field_list = ['Humanities',
                               'Social science',
                               'Science and engineering']
        elif self.flag_field:
            self.field_list = ['Art',
                               'Biology',
                               'Business',
                               'Chemistry',
                               'Computer science',
                               'Economics',
                               'Engineering',
                               'Environmental science',
                               'Geography',
                               'Geology',
                               'History',
                               'Materials science',
                               'Mathematics',
                               'Medicine',
                               'Philosophy',
                               'Physics',
                               'Political science',
                               'Psychology',
                               'Sociology']
        else:
            self.field_list = None

class GenResults:
    def __init__(self, data_obj, model, run_name):
        self.data_obj = data_obj
        self.model = model
        self.run_name = run_name

    def replace_var(self,
                    data,
                    varname,
                    val,
                    data_scaled=False,
                    return_scaled=False):
        temp = copy.deepcopy(data)
        if data_scaled:
            temp = self.data_obj.scaler.inverse_transform(temp)
        unscaled_data_replace = pd.DataFrame(data=temp, columns=data.keys())
        unscaled_data_replace[varname] = val
        if return_scaled:
            data_replaced = \
                pd.DataFrame(data=self.data_obj.scaler.\
                             transform(unscaled_data_replace),
                             columns=data.keys())
        else:
            data_replaced = unscaled_data_replace
        return data_replaced

    def scale_then_drop(self, data, col):
        temp = pd.DataFrame(data=self.data_obj.scaler.transform(data),
                            columns=data.keys(),
                            index=data.index)
        return temp.drop(columns=col)

    def scale(self, df):
        return pd.DataFrame(data=self.data_obj.scaler.transform(df),
                            columns=df.keys(),
                            index=df.index)

    def unscale(self, array, keys=None, index=None):
        if keys is not None:
            return pd.DataFrame(data=self.data_obj.\
                                scaler.inverse_transform(array),
                                columns=keys,
                                index=index)
        else:
            return pd.DataFrame(data=self.data_obj.scaler.\
                                inverse_transform(array),
                                columns=array.keys(),
                                index=array.index)

    def gen_hypo_field_paper_grid(self, data_i, field, hypo_paper_grid):
        n_grid = len(hypo_paper_grid)
        temp = pd.DataFrame()
        for f in self.data_obj.field_list:
            if len(data_i.shape)==1:
                temp['paper_count_{}'.format(f)] =\
                    np.repeat(data_i['paper_ratio_{}'.format(f)] *
                              (np.power(10, data_i['log_papers'])-1), n_grid)
            else:
                temp['paper_count_{}'.format(f)] =\
                    data_i['paper_ratio_{}'.format(f)] * \
                    (np.power(10, data_i['log_papers'])-1)
        diff = hypo_paper_grid - temp['paper_count_{}'.format(field)]
        temp['paper_count_{}'.format(field)] = hypo_paper_grid

        if len(data_i.shape) == 1:
            data_ii = pd.DataFrame()
            for col in data_i.keys():
                data_ii[col] = data_i[col].repeat(n_grid)
        else:
            data_ii = copy.deepcopy(data_i)
        data_ii['log_papers'] = \
            np.log10((np.power(10, data_i['log_papers'])-1) + diff + 1)
        for f in self.data_obj.field_list:
            data_ii['paper_ratio_{}'.format(f)] =\
                temp['paper_count_{}'.format(f)]/\
                (np.power(10, data_ii['log_papers'])-1)
        return data_ii



import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
from torch.nn import functional as F

class VanillaGrad(object):

    def __init__(self, pretrained_model, cuda=False):
        self.pretrained_model = pretrained_model
        #self.features = pretrained_model.features
        self.cuda = cuda
        #self.pretrained_model.eval()

    def __call__(self, x, index=None, task='classification'):
        output = self.pretrained_model(x)

        if task=='classification':
            if index is None:
                index = np.argmax(output.data.cpu().numpy())
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            if self.cuda:
                one_hot = Variable(torch.from_numpy(one_hot).cuda(), requires_grad=True)
            else:
                one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
            one_hot = torch.sum(one_hot * output)

            one_hot.backward(retain_graph=True)

            grad = x.grad.data.cpu().numpy()
            grad = grad[0, :, :, :]

        elif task=='regression':
            output_sum = torch.sum(output)
            output_sum.backward(retain_graph=True)
            grad = x.grad.data.cpu().numpy()


        return grad


class SmoothGrad(VanillaGrad):
    def __init__(self, pretrained_model, cuda=False, stdev_spread=0.15,
                 n_samples=25, magnitude=False, task='regression'):
        super(SmoothGrad, self).__init__(pretrained_model, cuda)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitutde = magnitude
        self.task = task

    def __call__(self, x, index=None):
        x = x.data.cpu().numpy()
        stdev = self.stdev_spread * (np.max(x) - np.min(x))
        #total_gradients = np.zeros_like(x)
        if self.cuda:
            total_gradients = torch.from_numpy( np.zeros_like(x) ).cuda() #For computing the second derivative.
        else:
            total_gradients = torch.from_numpy( np.zeros_like(x) ) #For computing the second derivative.

        for i in range(self.n_samples):
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            if self.cuda:
                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).cuda(), requires_grad=True)
            else:
                x_plus_noise = Variable(torch.from_numpy(x_plus_noise), requires_grad=True)
            output = self.pretrained_model(x_plus_noise)

            if self.task=='classification':
                if index is None:
                    index = np.argmax(output.data.cpu().numpy())

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][index] = 1
                if self.cuda:
                    one_hot = Variable(torch.from_numpy(one_hot).cuda(), requires_grad=True)
                else:
                    one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
                one_hot = torch.sum(one_hot * output)

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                #one_hot.backward(retain_variables=True)
            elif self.task=='regression':
                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                output_sum = torch.sum(output)
                #output_sum.backward(retain_graph=True)

            #grad = x_plus_noise.grad
            #grad = x_plus_noise.grad.data.cpu().numpy()
            #grad2 = grd( output_sum, x_plus_noise, create_graph=True )[0]
            gradient = grad( output_sum, x_plus_noise, create_graph=True, allow_unused=True )[0]

            if self.magnitutde:
                total_gradients += (gradient*gradient)
                #total_gradients += (grad * grad)
                #total_gradients2 += (grad2 * grad2)
            else:
                #total_gradients += grad
                #total_gradients2 += grad2
                total_gradients += gradient

            #if self.visdom:
        if self.task=='classification':
            avg_gradients = total_gradients[0, :, :, :] / self.n_samples
            #avg_gradients2 = total_gradients2[0, :, :, :] / self.n_samples
        elif self.task=='regression':
            avg_gradients = total_gradients / self.n_samples
            #avg_gradients2 = total_gradients2 / self.n_samples

        return avg_gradients.data.cpu().numpy()


class SmoothGrad2(VanillaGrad):
    '''
    Compute the second derivative of the model by applying SmoothGrad twice.
    The second derivative is taken for a specific column specified as target_col
    '''
    def __init__(self, pretrained_model, target_col, cuda=False, stdev_spread_1st=0.01, stdev_spread_2nd=0.05,
                 n_samples_1st=25, n_samples_2nd=25,magnitude=False, task='regression'):
        super(SmoothGrad2, self).__init__(pretrained_model, cuda)
        self.target_col = target_col
        self.stdev_spread_1st = stdev_spread_1st
        self.stdev_spread_2nd = stdev_spread_2nd
        self.n_samples_1st = n_samples_1st
        self.n_samples_2nd = n_samples_2nd
        self.magnitutde = magnitude
        self.task = task

    def __call__(self, x, index=None):
        x = x.data.cpu().numpy()
        stdev_1st = self.stdev_spread_1st * (np.max(x) - np.min(x))
        stdev_2nd = self.stdev_spread_2nd * (np.max(x) - np.min(x))
        #total_gradients = np.zeros_like(x)

        total_2nd_gradients = torch.zeros_like(torch.tensor(x) )
        if self.cuda:
            total_2nd_gradients = torch.zeros_like(torch.from_numpy(x).cuda() ) #For computing the second derivative.
        else:
            total_2nd_gradients = torch.zeros_like(torch.from_numpy(x) ) #For computing the second derivative.

        for j in range(self.n_samples_2nd):
            noise2 = np.random.normal(0, stdev_2nd, x.shape).astype(np.float32)
            x_base = x + noise2
            ##Start SmoothGrad at x_base, which is x+noise2
            if self.cuda:
                x_base = Variable(torch.from_numpy(x_base).cuda(), requires_grad=True)
            else:
                x_base = Variable(torch.from_numpy(x_base), requires_grad=True)
            total_gradients = torch.zeros_like(x_base) #For computing the second derivative.


            for i in range(self.n_samples_1st):
                noise = np.random.normal(0, stdev_1st, x_base.shape).astype(np.float32)
                x_plus_noise = x_base + torch.tensor(noise)
                x_plus_noise.retain_grad()
                output = self.pretrained_model(x_plus_noise)

                if self.task=='classification':
                    if index is None:
                        index = np.argmax(output.data.cpu().numpy())
                    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                    one_hot[0][index] = 1
                    if self.cuda:
                        one_hot = Variable(torch.from_numpy(one_hot).cuda(), requires_grad=True)
                    else:
                        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
                    one_hot = torch.sum(one_hot * output)
                    if x_plus_noise.grad is not None:
                        x_plus_noise.grad.data.zero_()
                    #one_hot.backward(retain_variables=True)
                elif self.task=='regression':
                    if x_plus_noise.grad is not None:
                        x_plus_noise.grad.data.zero_()
                    output_sum = torch.sum(output)

                gradient = grad( output_sum, x_plus_noise, create_graph=True, retain_graph=True )[0]
                total_gradients += gradient

                #if self.visdom:
            if self.task=='classification':
                avg_gradients = total_gradients[0, :, :, :] / self.n_samples_1st
            elif self.task=='regression':
                avg_gradients = total_gradients / self.n_samples_1st

            ##Compute second derivative
            sum_avg_gradient_target = torch.sum( avg_gradients[:,self.target_col] )
            gradient2nd = grad( sum_avg_gradient_target, x_base, create_graph=True, retain_graph=True, allow_unused=True )[0]
            total_2nd_gradients += gradient2nd
        if self.task=='classification':
            avg_2nd_gradients = total_2nd_gradients[0, :, :, :] / self.n_samples_2nd
        elif self.task=='regression':
            avg_2nd_gradients = total_2nd_gradients / self.n_samples_2nd

        return avg_2nd_gradients.data.cpu().numpy()


if __name__ == '__main__':
    X = torch.tensor( np.random.randn(1000,10).astype(np.float32) )
    #model = nn.Linear(10,1 )
    model = nn.Sequential( nn.Linear(10,5 ), nn.Linear(5,5),nn.Linear(5,1), nn.ReLU() )
    FirstDerivativeModel = SmoothGrad(model, cuda=False, \
    stdev_spread=0.15,n_samples=25, magnitude=False, task='regression')
    FirstDerivative = FirstDerivativeModel(X)

    target_col = 0
    SecondDerivativeModel = SmoothGrad2(model, target_col, cuda=False, stdev_spread_1st=0.01, stdev_spread_2nd=0.05,\
             n_samples_1st=25, n_samples_2nd=25, magnitude=False, task='regression')
    SecondDerivative = SecondDerivativeModel(X)

    print(SecondDerivative)

    '''
    x = torch.tensor([3.], requires_grad=True)
    y = x**4
    for i in range(5):
        print(i,y)
        grads = grad(y, x, create_graph=True)[0]
        y = grads.sum()
    #Works fine


    x= torch.tensor( np.arange(8).reshape([4,2]).astype(np.float32), requires_grad=True )
    #model = nn.Sequential( nn.Linear(2,2), nn.Linear(2,1) ) #same if model = nn.Linear(2,1)
    model = nn.Sequential( nn.Linear(2,2), nn.Sigmoid(),nn.Linear(2,1) )
    output = model(x)
    obj = torch.sum(output)
    grads = grad(obj, x, create_graph=True )[0]
    gradsum = grads.sum()
    grads2 = grad(gradsum, x, create_graph=True)[0]
    #One of the differentiated Tensors appears to not have been used in the graph.
    #With allow_unused=True in grad, grads2 becomes all zero.

    x= torch.tensor( np.arange(8).reshape([4,2]).astype(np.float32), requires_grad=True )
    output = x**2
    obj = torch.sum(output)
    grads = grad(obj, x, create_graph=True )[0]
    gradsum = grads.sum()
    grads2 = grad(gradsum, x, create_graph=True)[0]
    #Works fine

    x= torch.tensor( np.arange(8).reshape([4,2]).astype(np.float32), requires_grad=True )
    model = nn.Sequential( nn.Linear(2,2), nn.Linear(2,1) ) #same if model = nn.Linear(2,1)
    output = model(x)
    obj = torch.sum(output)
    grads = grad(obj, model.parameters(), create_graph=True, retain_graph=True )[0]
    gradsum = grads.sum()
    grads2 = grad(gradsum, model.parameters(), create_graph=True)[0]



    '''
