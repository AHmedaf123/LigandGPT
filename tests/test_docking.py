"""End-to-end docking on targets 0 and 3: receptor PDBQT (Meeko) + ligand PDBQT (Meeko) + QuickVina 2.
Run: .venv/bin/python -m unittest discover -s tests -p 'test_docking.py' -v   (about 10 s)"""
import json
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import scoring_function as sf  # noqa: E402
from data_utils import receptor_paths  # noqa: E402

DATA = os.path.join(REPO, "data")
QVINA = os.path.join(REPO, "docking", "qvina02")
READY = os.path.exists(QVINA) and os.path.exists(os.path.join(DATA, "target_map.json"))


@unittest.skipUnless(READY, "qvina02 or prepared receptors not available")
class DockTargets0And3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmap = json.load(open(os.path.join(DATA, "target_map.json")))

    def dock(self, target, smiles):
        m = self.tmap[str(target)]
        _, pdbqt = receptor_paths(m["protein_filename"], DATA)
        with tempfile.TemporaryDirectory() as td:
            return sf.perform_qvina_docking(smiles, pdbqt, m["center"], td)

    def test_target_0_scores_a_drug_like_molecule(self):
        score = self.dock(0, "CC(=O)Nc1ccc(O)cc1")
        self.assertIsNotNone(score)
        self.assertTrue(-12.0 <= score <= -3.0, score)

    def test_target_3_scores_a_drug_like_molecule(self):
        score = self.dock(3, "CC(=O)Nc1ccc(O)cc1")
        self.assertIsNotNone(score)
        self.assertTrue(-12.0 <= score <= -3.0, score)

    def test_invalid_molecule_gives_none_not_an_exception(self):
        self.assertIsNone(self.dock(3, "Oc1cccc(c1)OC1OC(CO)OC(O)OP(O)(O)(O)OP(O)OP(O)=CCC1"))


if __name__ == "__main__":
    unittest.main()
