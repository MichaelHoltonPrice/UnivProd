import unittest
from univprod import *
import os
import difflib
import yaml

class TestParsePapers(unittest.TestCase):
    # Check that the two scripts parse_Papers.py and
    # parse_Papers_test_version.py differ only in the essentials
    def test_parse_Papers_diff(self):
        with open('parse_Papers.py', 'r') as main_script:
            main_script_text = main_script.readlines()
        with open('parse_Papers_test_version.py', 'r') as test_script:
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
        expected_lines.append("@@ -6,11 +6,11 @@\n")
        expected_lines.append(" \n")
        expected_lines.append(" # Set the directory with MAG data and the directory where results will be\n")
        expected_lines.append(" # placed. The results directory should already have been created.\n")
        expected_lines.append("-mag_dir = os.path.join(os.path.expanduser(\"~\"), \"localMAG\")\n")
        expected_lines.append("+mag_dir = os.path.join(\".\", \"test_mag\")\n")
        expected_lines.append(" \n")
        expected_lines.append(" # The Papers.txt file should be located at /mag_dir/mag_version/mag/Papers.txt.\n")
        expected_lines.append(" # Raise an exception if it is not there.\n")
        expected_lines.append("-mag_version = \"2021-02-15\"\n")
        expected_lines.append("+mag_version = \"test-version\"\n")
        expected_lines.append(" papers_file = os.path.join(mag_dir, mag_version, \"mag\", \"Papers.txt\")\n")
        expected_lines.append(" if not os.path.exists(papers_file):\n")
        expected_lines.append("     raise Exception(\'Papers.txt is not at the expected location\')\n")
        expected_lines.append("@@ -37,14 +37,14 @@\n")
        expected_lines.append(" # For the 2021-02-15 version of MAG, there are 252109820 entries in Papers.txt.\n")
        expected_lines.append(" # Initialize the the progress bar, which is approximate if a different version\n")
        expected_lines.append(" # of MAG is used.\n")
        expected_lines.append("-num_papers = 252109820\n")
        expected_lines.append("+num_papers = 12\n")
        expected_lines.append(" prog_bar = progressbar.ProgressBar(max_value=num_papers)\n")
        expected_lines.append(" \n")
        expected_lines.append(" print(\"Iterating over Papers.txt\")\n")
        expected_lines.append(" start_time = datetime.now()\n")
        expected_lines.append(" with open(papers_file, encoding=\'utf-8\') as infile:\n")
        expected_lines.append("     # Open the lmdb database with a sufficiently large map_size (80 GB)\n")
        expected_lines.append("-    env = lmdb.open(db_dir, map_size=80e9, lock=False)\n")
        expected_lines.append("+    env = lmdb.open(db_dir, map_size=1e7, lock=False)\n")
        expected_lines.append(" \n")
        expected_lines.append("     # Set the commit period\n")
        expected_lines.append("     commit_period = 1000\n")

        self.assertEqual(
            lines,
            expected_lines
        )

    # Check that calling parse_Papers_test_version.py created the correct results
    def test_parse_Papers(self):
        path_dict_file = "path_dict.yaml"
        with open(path_dict_file, 'r') as f:
            path_dict = yaml.load(f)

        results_dir = path_dict["results_dir"]
        db_dir = os.path.join(results_dir, "paper_hash")
        paper_data = lmdb_to_dict(db_dir)

        self.assertEqual(
            paper_data,
            {b'01': b'1972-0-0',
             b'02': b'1974-0-0',
             b'03': b'1976-0-0',
             b'04': b'1978-0-0',
             b'05': b'1979-0-0',
             b'06': b'1980-0-0',
             b'07': b'1900-0-0',
             b'08': b'1990-0-0',
             b'09': b'1991-0-0',
             b'10': b'1992-0-0',
             b'11': b'1990-0-0',
             b'12': b'1991-0-0'}
        )

if __name__ == '__main__':
    unittest.main()
