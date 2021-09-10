import unittest
from univprod import *
import os
import difflib
import yaml

class TestShimComp(unittest.TestCase):
    def test_fos_byte_string2list(self):
        self.assertEqual(fos_byte_string2list(b"6",5),
                         [False,True,True,False,False])
        self.assertEqual(fos_byte_string2list(b"7",5),
                         [True,True,True,False,False])
        self.assertEqual(fos_byte_string2list(b"6",7),
                         [False,True,True,False,False,False,False])
        self.assertEqual(fos_byte_string2list(b"7",7),
                         [True,True,True,False,False,False,False])

    def test_list2fos_byte_string(self):
        self.assertEqual(list2fos_byte_string(
            [False, True, True, False, False]),
            b"6")
        self.assertEqual(list2fos_byte_string(
            [True, True, True, False, False]),
            b"7")
        self.assertEqual(list2fos_byte_string(
            [False, True, True, False, False,False,False]),
            b"6")
        self.assertEqual(list2fos_byte_string(
            [True, True, True, False, False,False,False]),
            b"7")

    def test_update_fos_byte_string(self):
        self.assertEqual(update_fos_byte_string(b"129",
                                                b"4",
                                                [b"129",b"130",b"131"]),
                         b"5")
        self.assertEqual(update_fos_byte_string(b"130",
                                                b"4",
                                                [b"129",b"130",b"131"]),
                         b"6")
        self.assertEqual(update_fos_byte_string(b"132",
                                                b"3",
                                                [b"129",b"130",b"131",b"132"]),
                         b"11")
        self.assertEqual(update_fos_byte_string(b"131",
                                                b"3",
                                                [b"129",b"130",b"131",b"132"]),
                         b"7")

        # Also check the exceptions
        self.assertRaises(ValueError,
                          update_fos_byte_string,
                          b"128",
                          b"4",
                          [b"129",b"130",b"131"])

        self.assertRaises(ValueError,
                          update_fos_byte_string,
                          b"131",
                          b"4",
                          [b"129",b"130",b"131"])

    def test_affildata_byte2lists(self):
        self.assertEqual(affildata_byte2lists(b""),
                         (list(),list()))
        self.assertEqual(affildata_byte2lists(b"1-A 1-B 2-B 3-C"),
                         ([b"1", b"1", b"2", b"3"],[b"A", b"B", b"B", b"C"]))

    def test_affildata_byte2lists(self):
        self.assertEqual(affildata_lists2byte(list(),list),
                         b"")
        self.assertEqual(affildata_lists2byte([b"1", b"1", b"2", b"3"],
                                              [b"A", b"B", b"B", b"C"]),
                         b"1-A 1-B 2-B 3-C")

    def test_get_fractional_contributions(self):
        self.assertEqual(
            get_fractional_contributions(list(),
                                         list(),
                                         [b"A", b"AA", b"B", b"C", b"CC"]),
            (list(),list())
        )

        self.assertEqual(
            get_fractional_contributions([b"1", b"1", b"2", b"3"],
                                         [b"A", b"B", b"B", b"C"],
                                         [b"A", b"AA", b"B", b"C", b"CC"]),
            ([0, 2, 3], [1/6, 3/6, 2/6])
        )

    # Test the two production array input/output functions together
    # (save_prod_array and load_prod_array)
    def test_prod_array_io(self):
        Ny = 10
        Na = 11
        num_fos = 23

        N1 = Ny
        N2 = Na
        N3 = 1 + num_fos
        prod_array = np.reshape(np.arange(0, N1*N2*N3)+0.1, [N1, N2, N3])

        file_path = "temp_file_for_test.npz"
        save_prod_array(prod_array, file_path)
        prod_array2 = load_prod_array(file_path)
        self.assertTrue(np.all(prod_array == prod_array2))

if __name__ == '__main__':
    unittest.main()
