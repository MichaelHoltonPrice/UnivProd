import unittest
from univprod import *
import os
import difflib
import yaml

class TestParsePapers(unittest.TestCase):
    # Check that the two scripts parse_PaperReferences.py and
    # parse_PaperReferences_test_version.py differ only in the essentials
    def test_parse_PaperReferences_diff(self):
        with open('parse_PaperReferences.py', 'r') as main_script:
            main_script_text = main_script.readlines()
        with open('parse_PaperReferences_test_version.py', 'r') as test_script:
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
        expected_lines.append("@@ -37,14 +37,14 @@\n")
        expected_lines.append(" # For the 2021-02-05 version of MAG, there are 1744495483 entries in\n")
        expected_lines.append(" # PaperReferences.txt. Initialize the progress bar, which is approximate if a\n")
        expected_lines.append(" # different version of MAG is used.\n")
        expected_lines.append("-num_pr = 1744495483\n")
        expected_lines.append("+num_pr = 10\n")
        expected_lines.append(" prog_bar = progressbar.ProgressBar(max_value=num_pr)\n")
        expected_lines.append(" \n")
        expected_lines.append(" start_time = datetime.now()\n")
        expected_lines.append(" print(\"Iterating over PaperReferences.txt\")\n")
        expected_lines.append(" with open(pr_file) as infile:\n")
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

    # Check that calling parse_PaperReferences_test_version.py created the
    # correct results
    def test_parse_PaperReferences(self):
        path_dict_file = "path_dict.yaml"
        with open(path_dict_file, 'r') as f:
            path_dict = yaml.load(f)

        results_dir = path_dict["results_dir"]
        db_dir = os.path.join(results_dir, "paper_hash")
        paper_data = lmdb_to_dict(db_dir)

        self.assertEqual(
            paper_data,
            {b'01': b'1972-2-0',
             b'02': b'1974-2-0',
             b'03': b'1976-0-0',
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

if __name__ == '__main__':
    unittest.main()
