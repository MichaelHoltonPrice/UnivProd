from datetime import datetime
import os
import lmdb
import progressbar
import yaml

# Set the directory with MAG data and the directory where results will be
# placed. The results directory should already have been created.
mag_dir = os.path.join(".", "test_mag")

# The Papers.txt file should be located at /mag_dir/mag_version/mag/Papers.txt.
# Raise an exception if it is not there.
mag_version = "test-version"
papers_file = os.path.join(mag_dir, mag_version, "mag", "Papers.txt")
if not os.path.exists(papers_file):
    raise Exception('Papers.txt is not at the expected location')

results_dir = os.path.join(os.path.expanduser("~"), "shimao_et_al_results")
# Raise an exception if results_dir does not exist
if not os.path.isdir(results_dir):
    raise Exception('results_dir does not exist at the expected location')

# The paper hash is located in the directory results_dir/paper_hash. Throw an
# error if the directory already exists.
db_dir = os.path.join(results_dir, "paper_hash")
if os.path.isdir(db_dir):
    raise Exception('The directory ' + db_dir + ' already exists')

# Write mag_dir, mag_version, and results_dir to a yaml file that is read in by
# subsequent scripts.
path_dict = {"mag_dir": mag_dir,
             "mag_version": mag_version,
             "results_dir": results_dir}
with open('path_dict.yaml', 'w') as file:
    yaml.dump(path_dict, file)

# For the 2021-02-05 version of MAG, there are 252109820 entries in Papers.txt.
# Initialize the the progress bar, which is approximate if a different version
# of MAG is used.
num_papers = 12
prog_bar = progressbar.ProgressBar(max_value=num_papers)

print("Iterating over Papers.txt")
start_time = datetime.now()
with open(papers_file, encoding='utf-8') as infile:
    # Open the lmdb database with a sufficiently large map_size (80 GB)
    env = lmdb.open(db_dir, map_size=1e7, lock=False)

    # Set the commit period
    commit_period = 1000

    # Set the report period
    report_period = 10000

    # Begin a new set of transactions
    txn = env.begin(write=True)

    # Iterate over lines to add them to the lmdb database
    for m, line0 in enumerate(infile):
        line = line0.encode()  # to byte
        tokens = line.split(b'\t')

        # Get the id
        id_bytestring = tokens[0]

        # Get the year
        yr_bytestring = tokens[7]

        # Add b'-0-0' to initialize the citation count and field of study (FoS)
        # "record". The FoS record is an integer that is converted to binary to
        # indicate whether each top-level MAG FoS is associated with this paper.
        value = yr_bytestring + b"-0-0"

        # Add the transaction
        txn.put(id_bytestring, value)

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
print("Parsing Papers.txt took: " + str(end_time-start_time))
