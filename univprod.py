import numpy as np
from pandas import read_csv
import json
import os
import pandas as pd
import progressbar
import lmdb

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

        data_dir: The path to the input data directory (likely /zenodo_inputs)
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

    def __init__(self, data_dir: str, results_dir: str, verbose=False):
        """Initialize the UnivDataManager class.
        Call a set of "helper" methods to initialize the class.
        Args:
            data_dir: The path to the input data directory (likely
                /zenodo_inputs)
            results_dir: The path to the results directory with production data
            verbose: Should summary and status information be printed out?
            :type data_dir: str
            :type results_dir: str
            :type verbose: bool
        """
        self.data_dir = data_dir
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
            os.path.join(self.data_dir,
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
        delta_dir = self.data_dir

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
        cw_dir = self.data_dir
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
        """Load Chetty data using self.data_dir
        Load the Chetty data, which consist of:
        (1) mrc_table3.csv / self.chetty_longitudinal
        (2) mrc_table11.csv / self.chetty_college_level
        That is, mrc_table3.csv provides the year- (cohort-) dependent
        longitudinal data and mrc_table11.csv provides the college level data,
        such as whether a super_opeid is used for the institution.
        """
        if self.verbose:
            print("Loading Chetty data")

        longi_file = os.path.join(self.data_dir, "mrc_table3.csv")
        collg_file = os.path.join(self.data_dir, "mrc_table11.csv")
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