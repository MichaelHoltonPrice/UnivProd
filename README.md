# Overview
This README describes how to run the Python analysis code for the following
article:

Price et al. 2021 (in review) -- UnivProd: A university production dataset

The analysis is divided into three steps:

1. Download input files from Zenodo
2. Utilize the Microsoft Academic Graph (MAG) to create production arrays for
   research institutions
3. Merge MAG, Delta, and Chetty data using a crosswalk
4. Add Liberal Arts colleges as a dummy variable

Note: We are not allowed to deposit the MAG files on Zenodo. They are, however,
available by request [here](https://docs.microsoft.com/en-us/academic-services/graph/get-started-setup-provisioning).

This README assumes a terminal/command line and proficiency using it. To obtain
the analysis the code, clone the git repository and change directory into it:

```
git https://github.com/MichaelHoltonPrice/UnivProd
cd UnivProd
```

The MAG schema can be found at this [link]
(https://docs.microsoft.com/en-us/academic-services/graph/reference-data-schema)
or in the file entity-relationship-diagram.png in this github repository. 

# Downloading the input files from Zenodo
To download the inputs files (aside from the MAG files) from Zenodo run the
following script at the command line:

```
python .\download_zenodo_inputs.py
```

This will create a directory named /zenodo_inputs and download all necessary 
non-MAG input files into the directory.

# Creating production arrays using MAG
The production arrays are created by running, in sequence, the following five
steps:

1. Run tests
2. Run parse_Papers.py
3. Run parse_PaperReferences.py
4. Run parse_PaperFieldsOfStudy.py
5. Run parse_PaperAuthorAffiliations.py

## Running the tests (validation)
This step is optional. Ensure that the folder UnivProd exists in the home 
directory and is empty. Then, run the following set of commands to
test MAG:

```
python .\unit_tests.py
python .\parse_Papers_test_version.py
python .\test_parse_Papers.py
python .\parse_PaperReferences_test_version.py
python .\test_parse_PaperReferences.py
python .\parse_PaperFieldsOfStudy_test_version.py
python .\test_parse_PaperFieldsOfStudy.py
python .\parse_PaperAuthorAffiliations_test_version.py
python .\test_parse_PaperAuthorAffiliations.py
```

The files ending \_test_version.py are minimally modified versions of the
parsing scripts, for which none of the comments have been modified (only the
actual commands). The files beginning test\_ check that only needed
modifications are present in the modified parsing scripts.

## parse_Papers.py
If necessary (e.g., if the tests in the preceding folder were run, delete
the files in the results folder (if paper_hash remains, an error will be 
thrown). Parse the MAG Papers.txt file with the following command:

```
python .\parse_Papers.py
```

This script creates an lmdb, hard drive based hash map for which the keys are
paper IDs and the values store three things: the publication year, number of
citations of the paper within a five year window, and a field of study (fos or
FoS) record that stores whether each top-level MAG FoS is associated with this
paper (see below for details on the FoS record). These three things are stored
as byte strings separated by dashes. For example, a paper published in 1991
with 14 citations and a FoS record b"6" has the hash value b"1991-14-6".

parse_Papers.py iterates over the Papers.txt file to obtain the publication
year, but does also initialize the second two entries of the hash value, which
are updated by subsequent scripts. For the preceding example, the hash value
after running parse_Papers.py is b"1991-0-0".

parse_Papers.py interacts with three directories: (a) the MAG directory, (b) a
results directory, and (c) the active directory in which the script is running
(the base directory of the git repository). The first two directories must be
specified in the script by setting the variables mag_dir and results_dir. In
addition, the MAG version, mag_version, (e.g., "2021-02-15") must be set. These
three variables are stored in the active directory in the file path_dict.yaml
so that they do not have to be set for subsequent scripts.

The Papers.txt file is located at: mag_dir/mag_version/mag/Papers.txt.

A new lmdb database is created at: results_dir/paper_hash

If hard disk space is limited, the processing can be done with only the MAG
input files needed for each pertinent step. For example, only the file
Papers.txt needs to be at /mag_dir/mag_version/mag for parse_Papers.txt. Once
parse_Papers.py has been run, Papers.txt can be deleted and PaperReferences.txt
can be copied to the directory prior to running the next script,
parse_PaperReferences.py

## parse_PaperReferences.py
Parse the MAG PaperReferences.txt file with the following command:

```
python .\parse_PaperReferences.py
```

This script updates the hash values in the lmdb hash map to have, for each
paper, the number of citations within a five year window. For the preceding
example, the hash value after running parse_Papers.py is b"1991-14-6".

The PaperReferences.txt file is located at: mag_dir/mag_version/mag.

For this and subsequent parsing scripts, the path information written to
path_dict.yaml by parse_Papers.txt is loaded from file to set mag_dir,
results_dir, and mag_version.

## parse_PaperFieldsOfStudy.py
Parse the MAG PaperFieldsOfStudy.txt file with the following command:

```
python .\parse_PaperFieldsOfStudy.py
```

This script updates the hash values in the lmdb hash map to have, for each
paper, the FoS record. For the preceding example, the hash value after running
parse_PaperFieldsOfStudy.py is b"1991-14-0".

Two files are needed for this step, FieldsOfStudy.txt and
PaperFieldsOfStudy.txt, both located at: mag_dir/mag_version/advanced. The
first defines the fields of study, which have multiple levels, and the second
indicates which papers are associated with which fields of study. For this
analysis, we only use the 19 highest level fields.

Each paper's FoS associations are stored as integer FoS byte strings. That is,
fos_byte_string is an integer that stores whether each paper contains each of
the fields of study, of which there are num_fos = 19. The integer is converted
to binary, where the leftmost bit is the least significant bit. Each bit
indicates whether each fos is in the paper. This is best illustrated with
an example. If fos_byte_string = b"06", then the bit representation is 01100...
For further details see the documentation for fos_byte_string2list in
univprod.py

## parse_PaperAuthorAffiliations.py
Parse the MAG PaperAuthorAffiliations.txt file with the following command:

```
python .\parse_PaperAuthorAffiliations.py
```

This script makes two passes through the PaperAuthorAffiliations.txt file in
order to parse the affiliation data in PaperAuthorAffiliations.py. In the
first pass, the number of affiliations that exist for each paper are counted
and stored in the dictionary p2num_rec (paper ID to number of records). In the
second pass, the affiliation data are "accumulated" in a second dictionary,
pdad (paper ID to affiliation data) until all records for a paper have been
accumulated. At that point, the production data for each affiliation
(institution) are updated, which involves two weightings: (1) a weighting for
fractional contribution of each institution to the paper and (2) a weighting
for FoS.

Institutional weighting is done first by author and then by
affiliation. For example, consider the byte string, s = b"1-A 1-B 2-B 3-C",
that has been accumulated and stored in p2aa. In s, author / affiliation pairs
are separated by spaces, and for each pair the author and affiliation IDs are
separated by a dash. In this case, author 1 is affiliated with two institutions,
A and B, and authors 2 and 3 with a single institution each (B and C,
respectively). Author 1 contributes 1/3, but that contribution is divided
across two affiliations (A and B). Author 2 contributes 1/3 to B and author 3 
contributes 1/3 to C. Hence, the fractional contributions are

Affiliation A: 1/3 * 1/2       = 1/6

Affiliation B: 1/3 * 1/2 + 1/3 = 3/6

Affiliation C:             1/3 = 2+/6

On top of the affiliation weighting, there is a compounding weighting for
fields of study that multiply the affiliation weightings. For the FoS
weighting, each field of study associated with a paper contributes equally.
For example, if a paper has three fields of study, each field is assigned one
third of the weight. If a paper has no field of study, it is placed in a
special bin in the production arrays at the first index (see below; there is
also a blank institution ID, b"", for papers with no identified affiliations).

The main output of parse_PaperAuthorAffiliations.py is two arrays,
pap_prod_array and cit_prod_array, which are saved as .csv files in the results
directory. The first array is the production of papers by year, institution and
FoS, whereas the second array is the production of citations (within the five
year  window). These two arrays have the same dimensions,
Ny x Na x 1 + num_fos, where Ny is the number of years, Na, is the number of
affiliations, and num_fos is the number of fields of study. The first entry
of the third dimension is for papers with no associated top level fields of
study. A similar entry is not needed for affiliations because, as noted above,
one of the institutions is the blank institution.

# Do the crosswalk

```
python .\do_crosswalk.py
```

This script loads and merges three separate types of data (MAG production data,
Delta data, and Chetty data). The merge is handled by a class called 
UnivDataManager in the file univprod.py. Full details of the merge are 
available in the documentation for the class, but we also summarize the process
here.

The necessary input data are located in two places, which are
inputs to the initialization method of UnivDataManager: (1) data_dir, 
which contains new data needed for the merge (in the do_crosswalk.py script,
data_dir is set equal to /zenodo_inputs) and (2) results_dir, which contains
the results of previous processing to create the MAG production data. In 
particular, the following input files are needed at the following locations:

1. /data_dir/mag_matches_to_unitid_with_mag_title.xlsx
2. /data_dir/delta_public_release_87_99.csv
3. /data_dir/delta_public_release_00_15.csv
4. /data_dir/CW*
5. /results_dir/years.json
6. /results_dir/all_affil.json
7. /results_dir/top_level_names.txt
8. /results_dir/pap_prod_array.npz
9. /results_dir/cit_prod_array.npz
10. /data_dir/mrc_table3.csv
11. /data_dir/mrc_table11.csv

The college scorecard Crosswalk files are of the form CW2000.xlsx and span the 
years 2000 through 2018.

The initialization method of class UnivDataManager accomplishes the initial 
steps of the merge by calling the following eight methods:

1. load_mag_id_matching_data
2. load_delta_data
3. load_crosswalk_data
4. load_mag_prod_data
5. load_chetty_data
6. screen_ipeds_mag_matches
7. filter_for_stable_ope_id6
8. filter_for_chetty_super_groups

The first five methods merely load the input data.

Method 6, screen_ipeds_mag_matches, screens the matches we made between MAG
IDs and IPEDS IDs to accept only those we are very confident in. This yields a
new variable, self.screened_ipeds_ids (of base class set), that contains the
IPEDS IDs that survived the screening.

Method 7, filter_for_stable_ope_id6, filters the set of IPEDS from the previous
step (self.screened_ipeds_ids) based on information in the College Scorecard
crosswalk. In particular, it requires that each IPEDS ID is associated with
exactly one OPE ID6 and, further, that that OPE ID6 does not change across 
years (2000 through 2018). The result is a reduced set of IPEDS IDs, along with
a unique MAG ID and OPE ID6 for each IPEDS ID. These are stored in the newly
created variable self.stable_id_triplets.

Method 8, filter_for_chetty_super_groups, removes entries in
self.stable_id_triplets for which the super OPEID is either -1 or part of a
Chetty super group. This yields a new set of triplets, self.final_id_triplets.

In summary, the initialization method of class UnivDataManager loads input data
and creates a final set of ID triplets (IPEDS, MAG, and Chetty). The actual
merge is accomplished by calling the method create_merged_production_data,
adds MAG and chetty data to Delta to create a final, merged data frame, which
is saved as results_dir/delta_with_MAG_and_chetty.csv.

# Add the liberal arts dummy variable
The final step is to append a column to the final dataframe with a dummy
variable (binary indicator) of whether each institution is a liberal arts
college. The source of the list of liberal arts colleges is the US News and
World report rankings of liberal arts colleges, which has been collated by
Andrew G. Reiter and posted [on his website](https://andyreiter.com/wp-content/uploads/2021/09/US-News-Rankings-Liberal-Arts-Colleges-Through-2022.xlsx).
This file is also permanently archived on Zenodo with out other input files 
(the only files not archived on Zenodo are the MAG files, which we do not have
the permission to post there.)

To append the liberal arts dummy run the following script, which creates a new
file, delta_with_MAG_and_chetty_and_la.csv, in the results folder.

```
python .\add_liberal_arts_dummy.py
```