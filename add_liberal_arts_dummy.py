import yaml
import os
import pandas as pd

# The path information containing mag_dir, mag_version, and results_dir is
# loaded from the file path_dict.yaml that should have been created already by
# parse_Papers.py. Throw an exception if the file does not exist.
path_dict_file = "path_dict.yaml"
if not os.path.exists(path_dict_file):
    raise Exception('path_dict.yaml is not at the expected location')

with open(path_dict_file, 'r') as f:
  path_dict = yaml.load(f)

# For now, hard code the results directory to the development file
results_dir = path_dict["results_dir"]

path_to_delta = os.path.join(results_dir, "delta_with_MAG_and_chetty.csv")
delta = pd.read_csv(path_to_delta)

# Read the Liberal Arts colleges rankings and create a list of IPEDS /unit IDs
path_to_la =\
    os.path.join("zenodo_inputs",
                 "US-News-Rankings-Liberal-Arts-Colleges-Through-2022.xlsx")
la = pd.read_excel(path_to_la, sheet_name="Rankings", skiprows=1)
unitids = set(la["IPEDS ID"].to_list())

# Iterate over the delta+ data frame to create the dummy column
la_dummy = list()
for i, row in enumerate(delta.iterrows()):
    row_label, row = row
    la_dummy.append(row.get("unitid") in unitids)

delta["is_la"] = la_dummy
output_file = os.path.join(results_dir, "delta_with_MAG_and_chetty_and_la.csv")
delta.to_csv(output_file, index=False)