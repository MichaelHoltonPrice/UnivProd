import yaml
from univprod import *
import os

# The path information containing mag_dir, mag_version, and results_dir is
# loaded from the file path_dict.yaml that should have been created already by
# parse_Papers.py. Throw an exception if the file does not exist.
path_dict_file = "path_dict.yaml"
if not os.path.exists(path_dict_file):
    raise Exception('path_dict.yaml is not at the expected location')

with open(path_dict_file, 'r') as f:
  path_dict = yaml.load(f)

results_dir = path_dict["results_dir"]

data_dir = "zenodo_inputs"
udm = UnivDataManager(data_dir, results_dir, verbose=True)

udm.summarize_filtering()
pcc_ipeds_id = "209746"
hmc_ipeds_id = "115409"
stanford_ipeds_id = "243744"
penn_state_ipeds_id = "214777"

year = 2005
udm.summarize_matching(pcc_ipeds_id, year)
udm.summarize_matching(hmc_ipeds_id, year)
udm.summarize_matching(stanford_ipeds_id, year)
udm.summarize_matching(penn_state_ipeds_id, year)

output_file = os.path.join(results_dir, "delta_with_MAG_and_chetty.csv")
merged = udm.create_merged_production_data(output_file)

print("Finished with crosswalk")