from datetime import datetime
import os
import lmdb
import progressbar
import numpy as np
import yaml

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

# Citations must be within five years to count
year_window = 5
# The PaperReferences.txt file should be located at
# /mag_dir/mag_version/mag/PaperReferences.txt. Raise an exception if it is not
# there.
pr_file = os.path.join(mag_dir, mag_version, "mag", "PaperReferences.txt")
if not os.path.exists(pr_file):
    raise Exception('PaperReferences.txt is not at the expected location')

# The paper hash is located in the directory results_dir/paper_hash. Throw an
# error if the file data.mdb does not exists in that directory.
db_dir = os.path.join(results_dir, "paper_hash")
if not os.path.isfile(os.path.join(db_dir, "data.mdb")):
    raise Exception('The file data.mdb does not exist in ' + db_dir)

# For the 2021-02-05 version of MAG, there are 1744495483 entries in
# PaperReferences.txt. Initialize the progress bar, which is approximate if a
# different version of MAG is used.
num_pr = 10
prog_bar = progressbar.ProgressBar(max_value=num_pr)

start_time = datetime.now()
print("Iterating over PaperReferences.txt")
with open(pr_file) as infile:
    # Open the lmdb database with a sufficiently large map_size (80 GB)
    env = lmdb.open(db_dir, map_size=1e7, lock=False)

    # Set the commit period
    commit_period = 100000

    # Set the report period
    report_period = 100000

    # Begin a new set of transactions
    txn = env.begin(write=True)

    # Iterate over lines to add them to the lmdb database
    for m, line in enumerate(infile):
        tokens = line.split('\t')
        id_new = tokens[0]  # the citing paper
        id_old = tokens[1].strip()  # the paper being cited

        id_new = id_new.encode('ascii')
        id_old = id_old.encode('ascii')

        # The hash has already had citation data appended to it
        # e.g., b"1970-4-2"
        hash_value_old = txn.get(id_old)

        # e.g., [b"1970",b"4",b"2"]
        hash_value_old = hash_value_old.split(b'-')
        year_old = hash_value_old[0]

        # If this entry has a year for the original paper, add it
        if year_old != b'':
            count = np.int(hash_value_old[1])
            hash_value_new = txn.get(id_new)
            year_old = np.int(year_old)

            # Is the year range (yearWindow) satisfied?
            hash_value_new = hash_value_new.split(b'-')
            year_new = np.int(hash_value_new[0])
            if (year_new - year_old) <= year_window:
                # Update the citation count of id_old
                count = count + 1
                year_old = str(year_old)
                year_old = year_old.encode('Ascii')
                count = str(count)
                count = count.encode('Ascii')
                updated_value_old =\
                    year_old + b"-" + count + b"-" + hash_value_old[2]
                txn.put(id_old, updated_value_old)

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
print("Parsing PaperReferences.txt took: " + str(end_time-start_time))