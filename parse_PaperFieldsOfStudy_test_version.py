from datetime import datetime
import os
import lmdb
import progressbar
import numpy as np
import yaml
from univprod import *

# The path information containing mag_dir, mag_version, and results_dir are
# loaded from the file path_dict.yaml that should have been created already by
# parse_Papers.py. Throw an exception if the file does not exists.
path_dict_file = "path_dict.yaml"
if not os.path.exists(path_dict_file):
    raise Exception('path_dict.yaml is not at the expected location')

with open(path_dict_file, 'r') as f:
  path_dict = yaml.load(f)

mag_dir = path_dict["mag_dir"]
mag_version = path_dict["mag_version"]
results_dir = path_dict["results_dir"]

# The PaperFieldsOfStudy.txt and FieldsOfStudy.txt files should be located in
# /mag_dir/mag_version/advanced/. Raise an exception if they are not there.
pf_file = os.path.join(mag_dir,
                       mag_version,
                       "advanced",
                       "PaperFieldsOfStudy.txt")
if not os.path.exists(pf_file):
    raise Exception('PaperFieldsOfStudy.txt is not at the expected location')

f_file = os.path.join(mag_dir, mag_version, "advanced", "FieldsOfStudy.txt")
if not os.path.exists(f_file):
    raise Exception('FieldsOfStudy.txt is not at the expected location')

# The paper hash is located in the directory results_dir/paper_hash. Throw an
# error if the file data.mdb does not exists in that directory.
db_dir = os.path.join(results_dir, "paper_hash")
if not os.path.isfile(os.path.join(db_dir, "data.mdb")):
    raise Exception('The file data.mdb does not exist in ' + db_dir)

print("Iterating over FieldsOfStudy.txt to identify top level fields")
top_level_ids = list()
top_level_names = list()
with open(f_file, encoding="utf-8") as infile:
    # Iterate over lines to add extract top level fields
    for m, line in enumerate(infile):
        # The tokens are:
        # FieldOfStudyId, Rank, NormalizedName, DisplayName, MainType, Level,
        # PaperCount, PaperFamilyCount, CitationCount, CreatedDate
        tokens = line.split('\t')
        if tokens[5] == '0':
            top_level_ids.append(tokens[0].encode("ascii"))
            top_level_names.append(tokens[3])

# Sort the lists
ind = np.argsort(top_level_names)
top_level_ids = [top_level_ids[k] for k in ind]
top_level_names = [top_level_names[k] for k in ind]

# Write top_level_ids to file
file_path = os.path.join(results_dir, "top_level_ids.txt")
with open(file_path, "w") as text_file:
    text_file.writelines([x.decode()+"\n" for x in top_level_ids])

# Write top_level_names to file
file_path = os.path.join(results_dir, "top_level_names.txt")
with open(file_path, "w") as text_file:
    text_file.writelines([x+"\n" for x in top_level_names])

# Set num_fos (for number of fields of study) to the length of top_level_names
num_fos = len(top_level_names)

# Create a dictionary that maps field of study IDs to their indices in
# top_level_ids and top_level_names
fos_id2index = dict()
for k, fos_id in enumerate(top_level_ids):
    fos_id2index[fos_id] = k

# For the 2021-02-15 version of MAG, there are 1458885638 entries in
# PaperFieldsOfStudy.txt. Initialize the the progress bar, which is approximate
# if a different version of MAG is used.
num_pf = 10
prog_bar = progressbar.ProgressBar(max_value=num_pf)

# Require 0.0 for the confidence of the field of study identification (this is
# not the same thing as the probability)
confidence_cutoff = 0.0

print("Iterating over PaperFieldsOfStudy.txt")
start_time = datetime.now()
with open(pf_file) as infile:
    # Open the lmdb database with a sufficiently large map_size (80 GB)
    env = lmdb.open(db_dir, map_size=1e7, lock=False)

    # Set the commit period
    commit_period = 100000

    # Set the report period
    report_period = 100000

    # Begin a new set of transactions
    txn = env.begin(write=True)

    # Iterate over lines to add them to the lmdb database
    top_level_lines = 0
    for m, line in enumerate(infile):
        # PaperId, FieldOfStudyId, Score
        tokens = line.split('\t')

        if float(tokens[2]) >= confidence_cutoff:
            fos_id = tokens[1].encode("ascii")
            if fos_id in top_level_ids:
                top_level_lines = top_level_lines + 1
                paper_id = tokens[0].encode("ascii")
                # Get old byte string from hash
                hash_value = txn.get(paper_id)
                hash_value = hash_value.split(b"-")
                fos_byte_string = hash_value[2]

                # Update byte string with new field
                fos_byte_string = update_fos_byte_string(fos_id,
                                                         fos_byte_string,
                                                         top_level_ids)

                # Create the new hash value
                new_hash_value = hash_value[0] + b"-" + \
                                 hash_value[1] + b"-" + \
                                 fos_byte_string

                # Add the hash to the database
                txn.put(paper_id, new_hash_value)

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

# Print out some summary info
print("--")
print("Number of lines:")
print(m+1)
print("--")
print("top_level_lines:")
print(top_level_lines)
print("Parsing PaperFieldsOfStudy.txt took: " + str(end_time-start_time))