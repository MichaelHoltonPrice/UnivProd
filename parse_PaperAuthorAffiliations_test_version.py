from datetime import datetime
import os
import lmdb
import progressbar
import numpy as np
import json
import yaml
from univprod import *

# The path information containing mag_dir, mag_version, and results_dir are
# loaded from the file path_dict.yaml that should have been created already by
# parse_Papers.py. Throw an exception if the file does not exist.
path_dict_file = "path_dict.yaml"
if not os.path.exists(path_dict_file):
    raise Exception('path_dict.yaml is not at the expected location')

with open(path_dict_file, 'r') as f:
  path_dict = yaml.load(f)

mag_dir = path_dict["mag_dir"]
mag_version = path_dict["mag_version"]
results_dir = path_dict["results_dir"]

# The PaperAuthorAffiliations .txt file should be located at
# /mag_dir/mag_version/mag/PaperAuthorAffiliations.txt. Raise an exception if
# it is not there.
paa_file = os.path.join(mag_dir, mag_version,
                        "mag", "PaperAuthorAffiliations.txt")
if not os.path.exists(paa_file):
    raise Exception('PaperAuthorAffiliations.txt '
                    'is not at the expected location')

# The paper hash is located in the directory results_dir/paper_hash. Throw an
# error if the file data.mdb does not exist in that directory.
db_dir = os.path.join(results_dir, "paper_hash")
if not os.path.isfile(os.path.join(db_dir, "data.mdb")):
    raise Exception('The file data.mdb does not exist in ' + db_dir)

# For the 2021-02-15 version of MAG, there are 677903527 entries in
# PaperReferences.txt. Initialize the progress bar, which is approximate if a
# different version of MAG is used.
num_paa = 9
prog_bar = progressbar.ProgressBar(max_value=num_paa)

# Set the commit period
commit_period = 100000

# Set the report period
report_period = 100000

# To balance memory and computation, two passes are made through the file to be
# parsed, paaFile0. In the first pass, the number of records per paper is
# stored in the dictionary p2numRec. In the second pass, the affiliation data
# for each record are stored in the dictionary p2ad, then written to an lmdb
# database once all data have been accumulated for a paper (which is determined
# by checking p2numRec.

# Store all the affiliations encountered in affil_set
affil_set = set()

# A dictionary that maps paper IDs to the number of records for that paper ID
# in PaperAuthorAffiliations.txt
p2num_rec = dict()

print("Iterating over PaperAuthorAffiliations.txt for the first time")
with open(paa_file, encoding="utf-8") as infile:
    for m, line in enumerate(infile):
        tokens = line.split('\t')
        paper_id = tokens[0]
        paper_id = paper_id.strip()
        paper_id = paper_id.encode('ascii')
        affil_id = tokens[2]
        affil_id = affil_id.strip()
        affil_id = affil_id.encode('ascii')
        affil_set.add(affil_id)
        if paper_id in p2num_rec:
            p2num_rec[paper_id] = p2num_rec[paper_id] + 1
        else:
            p2num_rec[paper_id] = 1
        # If the report period has been reached, update the progress bar
        if m % report_period == 0:
            # If m+1 exceeds the current maximum value of the progress bar
            # increase the max value to m+1
            if m + 1 > prog_bar.max_value:
                prog_bar.max_value = m + 1
            prog_bar.update(m + 1)

# A dictionary that maps paper IDs onto accumulated affiliation data
# represented as a byte string in the format expected by
# univprod.affildata_byte2lists.
p2ad = dict()

# Convert the set of all affiliations encountered to a list
all_affil = list(affil_set) # one entry is b''
all_affil.sort()

# Write the list of all affiliations to file.
file_path = os.path.join(results_dir,"all_affil.json")
with open(file_path, "w") as file_handle:
    file_handle.write(json.dumps([affil.decode() for affil in all_affil]))

# Na is the number of affiliations
Na = len(all_affil)

# Accumulate data for the years 1800 to 2020 (though the citation data require
# that a paper have five years of potential citations for comparisons to be
# meaningful).
minYear = 1800
maxYear = 2020

# A list of all the years
years = list(np.arange(minYear, maxYear + 1))

# Write years to file
file_path = os.path.join(results_dir,"years.json")
with open(file_path, "w") as file_handle:
    file_handle.write(json.dumps([str(y) for y in years]))

# Ny is the number of years
Ny = len(years)

# Load top level field of study IDs and names from file
file_path = os.path.join(results_dir, "top_level_ids.txt")
with open(file_path) as file_handle:
    top_level_ids = [x.strip().encode("ascii") for x in file_handle.readlines()]

file_path = os.path.join(results_dir, "top_level_names.txt")
with open(file_path) as file_handle:
    top_level_names = [x.strip() for x in file_handle.readlines()]

if len(top_level_ids) != len(top_level_names):
    raise ValueError("Lengths of IDs and names do not match for FoS")
num_fos = len(top_level_ids)

# Initialize variables to store the paper production data (the citation
# production data are generated by a separate script).
# papProdArray has has dimensions Ny x Na x (1+Nf). It is weighted by the
# contribution of each institution (affiliation) to the paper and by field of
# study. For example, for two authors and three affiliations where the first
# author has two affiliations, the weighting by affiliation is
# [0.25, 0.25, 0.50]. The field of study weighting is "on top of" the
# affiliation weighting, and is 1/num_fos_for_paper, where num_fos_for_paper is
# the number of top level fields of study for the paper. If no top level fields
# of study are found for the paper, the paper is added to the first element of
# the third dimension (hence why that dimension has 1+Nf, not Nf elements);
# that is, the first index accumulates "other" papers with non field of study
# associations.
pap_prod_array = np.zeros((Ny, Na, 1 + num_fos))
cit_prod_array = np.zeros((Ny, Na, 1 + num_fos))

# Initialize the progress bar
prog_bar = progressbar.ProgressBar(max_value=num_paa)

start_time = datetime.now()
print("Iterating over PaperAuthorAffiliations.txt for the second time")
with open(paa_file, encoding="utf-8") as infile:
    # Open the lmdb database with a sufficiently large map_size (80 GB)
    env = lmdb.open(db_dir, map_size=1e7, lock=False)
    # Begin a new set of transactions
    txn = env.begin(write=True)
    for m, line in enumerate(infile):
        tokens = line.split('\t')
        paper_id = tokens[0]
        paper_id = paper_id.strip()
        paper_id = paper_id.encode('ascii')
        authr_id = tokens[1]
        authr_id = authr_id.strip()
        authr_id = authr_id.encode('ascii')
        affil_id = tokens[2]
        affil_id = affil_id.strip()
        affil_id = affil_id.encode('ascii')

        # If this paper is already in the dictionary p2ad, get the
        # corresponding lists representing the affiliation data. If the paper
        # is not in p2ad, create empty lists.
        if paper_id in p2ad:
            s = p2ad[paper_id]
            authr_list, affil_list = affildata_byte2lists(s)
        else:
            authr_list = list()
            affil_list = list()

        # Update the author/affiliation lists
        authr_list.append(authr_id)
        affil_list.append(affil_id)

        # If all entries for this paper have been accumulated, add it to the
        # production arrays (so long as the year is not empty and is in the
        # required year range).
        if len(authr_list) == p2num_rec[paper_id]:
            ind_affil, weight_affil = get_fractional_contributions(authr_list,
                                                                   affil_list,
                                                                   all_affil)
            value = txn.get(paper_id)

            # Get year
            tokens = value.split(b"-")
            y = tokens[0]
            # Add the entry if the year is not blank and is in the range of
            # years
            if y != b'':
                y = int(y.decode())
                if y in years:
                    # Number of citations is the second element in value
                    num_cite = tokens[1]
                    num_cite = int(num_cite.decode())

                    # The fos byte string is the third element in value
                    fos_list = fos_byte_string2list(tokens[2], num_fos)
                    num_fos_for_paper = sum(fos_list)
                    if num_fos_for_paper > 0:
                        fos_prop = 1 / num_fos_for_paper
                    for n in range(0, len(ind_affil)):
                        na = ind_affil[n]
                        ny = years.index(y)

                        if num_fos_for_paper == 0:
                            # Since this paper has no associated FoS, add all
                            # the weight to :,:,0
                            pap_prod_array[ny, na, 0] =\
                                pap_prod_array[ny, na, 0] + weight_affil[n]
                            cit_prod_array[ny, na, 0] = \
                                cit_prod_array[ny, na, 0] + \
                                weight_affil[n] * num_cite
                        else:
                            # Since this paper has at least one associated FoS,
                            # add them after :,:,0
                            for n_fos in range(0,num_fos):
                                if fos_list[n_fos]:
                                    pap_prod_array[ny, na, 1 + n_fos] = \
                                        pap_prod_array[ny, na, 1 + n_fos] + \
                                        weight_affil[n] * fos_prop
                                    cit_prod_array[ny, na, 1 + n_fos] = \
                                        cit_prod_array[ny, na, 1 + n_fos] + \
                                        weight_affil[n] * fos_prop * num_cite

            # To save memory, remove entry from dictionary (if necessary; if
            # there is one entry for this paper, it was never added to p2ad.)
            if len(authr_list) > 1:
                del p2ad[paper_id]
        else:
            # More entries need to be accumulated. Update p2ad.
            s = affildata_lists2byte(authr_list, affil_list)
            p2ad[paper_id] = s

        # If the commit period has been reached, commit the transaction and
        # begin a new set of transactions.
        if (m % commit_period) == (commit_period - 1):
            txn.commit()
            txn = env.begin(write=True)
        # If the report period has been reached, update the progress bar
        if m % report_period == 0:
            # If m+1 exceeds the current maximum value of the progress bar
            # increase the max value to m+1
            if m + 1 > prog_bar.max_value:
                prog_bar.max_value = m + 1
            prog_bar.update(m + 1)

# Commit any remaining transactions and close the database
txn.commit()
env.close()

end_time = datetime.now()
print("Second pass to parse took: " + str(end_time-start_time))

# Make sure all records were processed
if len(p2ad) != 0:
  print("Not all records processed")

# Write pap_prod_array to file
file_path = os.path.join(results_dir, "pap_prod_array.npz")
save_prod_array(pap_prod_array,file_path)

# Write cit_prod_array to file
file_path = os.path.join(results_dir, "cit_prod_array.npz")
save_prod_array(cit_prod_array,file_path)
