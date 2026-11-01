"""Ligand SDF -> PDBQT with Meeko (replaces the obabel call). Run: .venv/bin/python -m unittest discover -s tests -p 'test_ligand_pdbqt.py' -v"""
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import scoring_function as sf  # noqa: E402


def to_pdbqt(smiles):
    mol = sf.prepare_ligand_molecule(smiles)
    if mol is None:
        return None, "embed failed"
    with tempfile.TemporaryDirectory() as td:
        sdf, out = os.path.join(td, "l.sdf"), os.path.join(td, "l.pdbqt")
        assert sf.write_molecule_to_sdf(mol, sdf)
        ok = sf.convert_sdf_to_pdbqt(sdf, out)
        return (open(out).read() if ok and os.path.exists(out) else None), ok


class LigandPdbqt(unittest.TestCase):
    def test_small_molecule_has_torsion_tree_and_autodock_types(self):
        text, ok = to_pdbqt("CC(=O)Nc1ccc(O)cc1")
        self.assertTrue(ok)
        lines = text.splitlines()
        self.assertIn("ROOT", lines)
        self.assertTrue(any(l.startswith("TORSDOF") for l in lines))
        types = {l.split()[-1] for l in lines if l.startswith("ATOM")}
        self.assertTrue({"C", "A", "OA", "N", "HD"} <= types, types)        # aromatic carbon is A, polar hydrogens are HD

    def test_macrocycle_does_not_write_glue_pseudo_atoms(self):
        text, ok = to_pdbqt("C1CCCCCCCCCCC1")
        self.assertTrue(ok)
        types = {l.split()[-1] for l in text.splitlines() if l.startswith("ATOM")}
        self.assertFalse({"G0", "CG0"} & types, types)                       # QVina cannot parse CG0/G0

    def test_hypervalent_junk_fails_cleanly(self):
        text, ok = to_pdbqt("Oc1cccc(c1)OC1OC(CO)OC(O)OP(O)(O)(O)OP(O)OP(O)=CCC1")
        if text is None and ok == "embed failed":
            self.skipTest("RDKit could not embed this molecule")
        self.assertFalse(ok)                                                  # NaN charges: reported as failure, not an exception


if __name__ == "__main__":
    unittest.main()
