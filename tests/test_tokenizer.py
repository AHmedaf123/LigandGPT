"""Tests for the coordinate-bearing ProtLigTokenizer (the 125fedc format, restored after a5fa5c3 removed the coordinate tokens).

Run:  .venv/bin/python -m unittest tests.test_tokenizer -v
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tokenizer import ProtLigTokenizer  # noqa: E402
from vocabulary import read_vocabulary  # noqa: E402

DATA = "/home/coder/project/data"
LMDB = os.path.join(DATA, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb")
SPLIT = os.path.join(DATA, "crossdocked_pocket10_pose_split.pt")

PROT_ATOMS = ["N", "CA", "C", "O", "CB", "H"]          # 'H' is dropped by the tokenizer
PROT_XYZ = np.arange(18, dtype=np.float64).reshape(6, 3) - 7.0
LIG_SMILES = "CC(=O)N"                                   # aliphatic C, O, N must survive a round trip
LIG_ATOMS = ["C", "C", "O", "N"]
LIG_XYZ = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.2, 0.0], [2.2, -1.1, 0.4]])


class TokenizerFormat(unittest.TestCase):
    def setUp(self):
        self.tk = ProtLigTokenizer()
        self.tokens, self.numbers = self.tk.tokenize(
            protein=PROT_ATOMS, prot_coords=PROT_XYZ,
            ligand_SMILES=LIG_SMILES, ligand_atoms=LIG_ATOMS, lig_coords=LIG_XYZ)

    def test_tokens_and_numbers_are_aligned(self):
        self.assertEqual(len(self.tokens), len(self.numbers))

    def test_protein_block_has_one_xyz_triplet_per_heavy_atom(self):
        t = self.tokens
        a, b = t.index("<PROT_COORDS_START>"), t.index("<PROT_COORDS_END>")
        self.assertEqual(t[a + 1:b], ["[x]", "[y]", "[z]"] * 5)           # 5 heavy atoms, H removed
        self.assertEqual(t[t.index("<PROT_START>") + 1:t.index("<PROT_END>")], ["N", "CA", "C", "O", "C"])

    def test_ligand_smiles_block_has_no_coordinate_tokens(self):
        t = self.tokens
        smi = t[t.index("<LIG_START>") + 1:t.index("<LIG_END>")]
        self.assertEqual(smi, ["C", "C", "(", "=", "O", ")", "N"])
        self.assertFalse({"[x]", "[y]", "[z]"} & set(smi))

    def test_ligand_coordinate_block_follows_the_smiles_block(self):
        t = self.tokens
        self.assertEqual(t[t.index("<LIG_END>") + 1], "<LIG_COORDS_START>")
        a, b = t.index("<LIG_COORDS_START>"), t.index("<LIG_COORDS_END>")
        self.assertEqual(t[a + 1:b], ["[x]", "[y]", "[z]"] * 4)

    def test_coordinate_numbers_are_scaled_by_coord_factor(self):
        t, n = self.tokens, self.numbers
        a = t.index("<PROT_COORDS_START>")
        got = np.array(n[a + 1:a + 1 + 15]).reshape(5, 3)
        keep = [i for i, name in enumerate(PROT_ATOMS) if not name.startswith("H")]
        np.testing.assert_allclose(got, PROT_XYZ[keep] / self.tk.coord_factor)
        non_coord = [x for tok, x in zip(t, n) if tok not in ("[x]", "[y]", "[z]")]
        self.assertTrue(all(x == 1 for x in non_coord))                     # every non-coordinate token carries number 1

    def test_untokenize_round_trip_keeps_aliphatic_atoms_and_coordinates(self):
        out = self.tk.untokenize(self.tokens, self.numbers)
        self.assertEqual(out["LigSmiles"], LIG_SMILES)                      # a5fa5c3 dropped C, N, O here
        np.testing.assert_allclose(out["LigCoords"], LIG_XYZ, atol=1e-6)
        keep = [i for i, name in enumerate(PROT_ATOMS) if not name.startswith("H")]
        np.testing.assert_allclose(out["ProtCoords"], PROT_XYZ[keep], atol=1e-6)

    def test_every_token_is_in_the_vocabulary(self):
        voc = read_vocabulary(os.path.join(REPO, "ProtLigVoc.txt"))
        self.assertTrue(all(tok in voc for tok in self.tokens))


@unittest.skipUnless(os.path.exists(LMDB) and os.path.exists(SPLIT), "CrossDocked lmdb not available")
class RealPocketPromptLengths(unittest.TestCase):
    """Prompt = pocket tokens + coordinates + <LIG_START>, as built in generation_finetuning.run_datapoint_generation."""

    @classmethod
    def setUpClass(cls):
        import pickle
        import lmdb
        import torch
        split = torch.load(SPLIT, weights_only=False)["test"]
        env = lmdb.open(LMDB, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
        cls.txn = env.begin()
        cls.keys = list(cls.txn.cursor().iternext(values=False))
        cls.split = split
        cls.pickle = pickle

    def prompt(self, target):
        dp = self.pickle.loads(self.txn.get(self.keys[self.split[target]]))
        coords = dp["protein_pos"].numpy() - dp["ligand_center_of_mass"].numpy()
        tokens, numbers = ProtLigTokenizer().tokenize(protein=dp["protein_atom_name"], prot_coords=coords)
        tokens.append("<LIG_START>")
        numbers.append(1)
        return tokens, numbers

    def test_target_0_prompt_length(self):
        tokens, numbers = self.prompt(0)
        self.assertEqual((len(tokens), len(numbers)), (1641, 1641))

    def test_target_3_prompt_length(self):
        tokens, numbers = self.prompt(3)
        self.assertEqual((len(tokens), len(numbers)), (1025, 1025))


if __name__ == "__main__":
    unittest.main()
