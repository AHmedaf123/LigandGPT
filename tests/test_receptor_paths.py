"""receptor_paths: lmdb protein_filename (*_pocket10.pdb, absent) -> full <id>_rec.pdb and its prepared PDBQT."""
import json
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from data_utils import receptor_paths  # noqa: E402

DATA = "/home/coder/project/data"


class ReceptorPaths(unittest.TestCase):
    def test_pocket10_name_maps_to_full_receptor(self):
        pdb, pdbqt = receptor_paths("BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb", "/d")
        self.assertEqual(pdb, "/d/test_set/BSD_ASPTE_1_130_0/2z3h_A_rec.pdb")
        self.assertEqual(pdbqt, "/d/receptors_pdbqt/BSD_ASPTE_1_130_0/2z3h_A_rec.pdbqt")

    @unittest.skipUnless(os.path.exists(DATA + "/target_map.json"), "prepared data not available")
    def test_every_test_target_resolves_to_existing_files(self):
        tmap = json.load(open(DATA + "/target_map.json"))
        self.assertEqual(len(tmap), 100)
        for t, m in tmap.items():
            pdb, pdbqt = receptor_paths(m["protein_filename"], DATA)
            self.assertTrue(os.path.exists(pdb), (t, pdb))
            self.assertTrue(os.path.exists(pdbqt) and os.path.getsize(pdbqt) > 1000, (t, pdbqt))


if __name__ == "__main__":
    unittest.main()
