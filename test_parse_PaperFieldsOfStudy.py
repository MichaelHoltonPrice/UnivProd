import unittest
from univprod import *
import os
import difflib
import yaml

class TestParsePapers(unittest.TestCase):
    # Check that the two scripts parse_PaperFieldsOfStudy.py and
    # parse_PaperFieldsOfStudy_test_version.py differ only in the essentials
    def test_parse_PaperFieldsOfStudy_diff(self):
        with open('parse_PaperFieldsOfStudy.py', 'r') as main_script:
            main_script_text = main_script.readlines()
        with open('parse_PaperFieldsOfStudy_test_version.py', 'r') as test_script:
            test_script_text = test_script.readlines()
        # Find and print the diff:
        lines = list()
        for line in difflib.unified_diff(main_script_text,
                                         test_script_text,
                                         fromfile='main',
                                         tofile='test',
                                         lineterm='\n'):
            lines.append(line)

        # Create expected diff as a list, line by line
        # Set the directory with MAG data and the directory where results will be
        # placed. The results directory should already have been created.
        expected_lines = list()
        expected_lines.append("--- main\n")
        expected_lines.append("+++ test\n")
        expected_lines.append("@@ -80,7 +80,7 @@\n")
#        expected_lines.append("@@ -91,7 +91,7 @@\n")
        expected_lines.append(" # For the 2021-02-05 version of MAG, there are 1458885638 entries in\n")
        expected_lines.append(" # PaperFieldsOfStudy.txt. Initialize the the progress bar, which is approximate\n")
        expected_lines.append(" # if a different version of MAG is used.\n")
        expected_lines.append("-num_pf = 1458885638\n")
        expected_lines.append("+num_pf = 10\n")
        expected_lines.append(" prog_bar = progressbar.ProgressBar(max_value=num_pf)\n")
        expected_lines.append(" \n")
        expected_lines.append(" # Require 0.0 for the confidence of the field of study identification (this is\n")
        expected_lines.append("@@ -91,7 +91,7 @@\n")
        expected_lines.append(" start_time = datetime.now()\n")
        expected_lines.append(" with open(pf_file) as infile:\n")
        expected_lines.append("     # Open the lmdb database with a sufficiently large map_size (80 GB)\n")
        expected_lines.append("-    env = lmdb.open(db_dir, map_size=80e9, lock=False)\n")
        expected_lines.append("+    env = lmdb.open(db_dir, map_size=1e7, lock=False)\n")
        expected_lines.append(" \n")
        expected_lines.append("     # Set the commit period\n")
        expected_lines.append("     commit_period = 100000\n")

        self.assertEqual(
            lines,
            expected_lines
        )

    # Check that calling parse_Papers_test_version.py created the correct results
    def test_parse_PaperFieldsOfStudy(self):
        path_dict_file = "path_dict.yaml"
        with open(path_dict_file, 'r') as f:
            path_dict = yaml.load(f)

        results_dir = path_dict["results_dir"]
        db_dir = os.path.join(results_dir, "paper_hash")
        paper_data = lmdb_to_dict(db_dir)

        self.assertEqual(
            paper_data,
            {b'01': b'1972-2-7',
             b'02': b'1974-2-5',
             b'03': b'1976-0-8',
             b'04': b'1978-1-0',
             b'05': b'1979-0-0',
             b'06': b'1980-0-0',
             b'07': b'1900-0-0',
             b'08': b'1990-2-0',
             b'09': b'1991-0-0',
             b'10': b'1992-0-0',
             b'11': b'1990-1-0',
             b'12': b'1991-0-0'}
        )

        file_path = os.path.join(results_dir, "top_level_ids.txt")
        with open(file_path) as file_handle:
            top_level_ids = [x.strip().encode("ascii") for x in
                             file_handle.readlines()]

        self.assertEqual(
            top_level_ids,
            [b'01', b'02', b'03', b'04']
        )

        file_path = os.path.join(results_dir, "top_level_names.txt")
        with open(file_path) as file_handle:
            top_level_names = [x.strip() for x in file_handle.readlines()]

        self.assertEqual(
            top_level_names,
            ['Field 1', 'Field 2', 'Field 3', 'Field 4']
        )

if __name__ == '__main__':
    unittest.main()
