import unittest
from univprod import *
import os
import difflib
import yaml
import math

class TestParsePapers(unittest.TestCase):
    # Check that the two scripts parse_PaperAuthorAffiliations.py and
    # parse_PaperAuthorAffiliations_test_version.py differ only in the
    # essentials
    def test_parse_PaperAuthorAffiliations_diff(self):
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

    # Check that calling parse_PaperAuthorAffiliations_test_version.py created
    # the correct results
    def test_parse_PaperAuthorAffiliations(self):
        path_dict_file = "path_dict.yaml"
        with open(path_dict_file, 'r') as f:
            path_dict = yaml.load(f)

        results_dir = path_dict["results_dir"]

        # The productions
        years = json.load(open(os.path.join(results_dir, "years.json")))
        years = [int(y) for y in years]
        self.assertEqual(
            years,
            list(np.arange(1800, 2020 + 1))
        )

        all_affil = json.load(open(os.path.join(results_dir, "all_affil.json")))
        # all_affil is not sorted in the parsing script. Casting from a set to
        # a list in the script yields an ordering of the affiliations that
        # happens to be the following.
        self.assertEqual(
            all_affil,
            ['', '1', '2', '3', '4', '5', '6']
        )

        Ny = len(years)
        Na = len(all_affil)
        pap_prod_array = load_prod_array(
            os.path.join(results_dir, "pap_prod_array.npz")
        )
        cit_prod_array = load_prod_array(
            os.path.join(results_dir, "cit_prod_array.npz")
        )

        # Directly build pap_prod_array and cit_prod_array to check that the
        # loaded versions created by parse_PaperAuthorAffiliations.py are
        # correct
        num_fos = pap_prod_array.shape[2] - 1
        pap_prod_array_direct = np.zeros((Ny, Na, 1 + num_fos))
        cit_prod_array_direct = np.zeros((Ny, Na, 1 + num_fos))

        # Paper 1
        # -------------------
        # paper_id      b"01"
        #     year       1972
        # num_cite          2
        # fos_list    1 1 1 0
        #
        # Affil: "1"      3/4
        #        "2"      1/4
        pap_prod_array_direct[years.index(1972),  all_affil.index("1"), :] = [0, 1/4, 1/4, 1/4, 0]
        cit_prod_array_direct[years.index(1972),  all_affil.index("1"), :] = [0, 2/4, 2/4, 2/4, 0]
        pap_prod_array_direct[years.index(1972),  all_affil.index("2"), :] = [0, 1/12, 1/12, 1/12, 0]
        cit_prod_array_direct[years.index(1972),  all_affil.index("2"), :] = [0, 1/6, 1/6, 1/6, 0]

        # Paper 2
        # -------------------
        # paper_id      b"02"
        #     year       1974
        # num_cite          2
        # fos_list    1 0 1 0
        #
        # Affil: ""         1
        pap_prod_array_direct[years.index(1974),  all_affil.index(""), :] = [0, 1/2, 0, 1/2, 0]
        cit_prod_array_direct[years.index(1974),  all_affil.index(""), :] = [0, 1, 0, 1, 0]

        # Paper 4
        # -------------------
        # paper_id      b"04"
        #     year       1978
        # num_cite          1
        # fos_list    0 0 0 0
        #
        # Affil: "3"        1
        pap_prod_array_direct[years.index(1978),  all_affil.index("3"), :] = [1, 0, 0, 0, 0]
        cit_prod_array_direct[years.index(1978),  all_affil.index("3"), :] = [1, 0, 0, 0, 0]

        # Paper 11
        # -------------------
        # paper_id      b"11"
        #     year       1990
        # num_cite          1
        # fos_list    0 0 0 0
        #
        # Affil: "4"        1
        pap_prod_array_direct[years.index(1990),  all_affil.index("4"), :] = [1, 0, 0, 0, 0]
        cit_prod_array_direct[years.index(1990),  all_affil.index("4"), :] = [1, 0, 0, 0, 0]

        # Paper 12
        # -------------------
        # paper_id      b"12"
        #     year       1991
        # num_cite          0
        # fos_list    0 0 0 0
        #
        # Affil: "6"        1
        pap_prod_array_direct[years.index(1991),  all_affil.index("6"), :] = [1, 0, 0, 0, 0]
        cit_prod_array_direct[years.index(1991),  all_affil.index("6"), :] = [0, 0, 0, 0, 0]

        # Paper 08
        # -------------------
        # paper_id      b"08"
        #     year       1990
        # num_cite          2
        # fos_list    0 0 0 0
        #
        # Affil: "4"        1/2
        # Affil: "5"        1/2
        #
        # [Affil "4" has already been added to above for 1990.]
        pap_prod_array_direct[years.index(1990),  all_affil.index("4"), :] = pap_prod_array_direct[years.index(1990),  all_affil.index("4"), :] + [1/2, 0, 0, 0, 0]
        cit_prod_array_direct[years.index(1990),  all_affil.index("4"), :] = cit_prod_array_direct[years.index(1990),  all_affil.index("4"), :] + [1, 0, 0, 0, 0]
        pap_prod_array_direct[years.index(1990),  all_affil.index("5"), :] = [1/2, 0, 0, 0, 0]
        cit_prod_array_direct[years.index(1990),  all_affil.index("5"), :] = [1, 0, 0, 0, 0]

        for n0 in range(0, pap_prod_array.shape[0]):
            for n1 in range(0, pap_prod_array.shape[1]):
                for n2 in range(0, pap_prod_array.shape[2]):
                    v1 = pap_prod_array[n0, n1, n2]
                    v2 = pap_prod_array_direct[n0, n1, n2]
                    self.assertTrue(math.isclose(v1, v2, abs_tol=1e-20))
                    v1 = cit_prod_array[n0, n1, n2]
                    v2 = cit_prod_array_direct[n0, n1, n2]
                    self.assertTrue(math.isclose(v1, v2, abs_tol=1e-20))

if __name__ == '__main__':
    unittest.main()
