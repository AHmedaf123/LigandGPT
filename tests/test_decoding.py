"""Property tests for the SMILES state machine in decoding.py (no model needed).

Run:  .venv/bin/python -m unittest tests.test_decoding -v
A random walk over the allowed tokens must always give a SMILES that RDKit can parse (sanitize=False), with balanced
branches and every ring digit opened and closed exactly once. With valence=True most walks must also sanitize.
"""
import os
import random
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rdkit import Chem, RDLogger  # noqa: E402

import decoding as dec  # noqa: E402
from vocabulary import read_vocabulary  # noqa: E402

RDLogger.DisableLog("rdApp.*")
VOC = read_vocabulary(os.path.join(REPO, "ProtLigVoc.txt"))
CLS = dec.token_classes(VOC)


ALIPHATIC = ["C", "N", "O", "F", "Cl", "Br", "I"]        # exact valences, so "valence respected" means "RDKit sanitizes"


def walk(rng, valence, max_steps=120, min_atoms=4, atoms=None, p_end=0.15):
    st = dec.SmilesState(atoms or CLS["atoms"], CLS["digits"], valence=valence, min_atoms=min_atoms)
    toks = []
    for step in range(max_steps):
        allowed = st.allowed(max_steps - step)
        if not allowed:
            return toks, False, st
        t = dec.END if (dec.END in allowed and rng.random() < p_end) else rng.choice([a for a in allowed if a != dec.END] or allowed)
        if t == dec.END:
            return toks, True, st
        st.advance(t)
        toks.append(t)
    return toks, False, st


class StateMachine(unittest.TestCase):
    def run_walks(self, valence, n=400, seed=0, atoms=None):
        rng = random.Random(seed)
        return [walk(rng, valence, atoms=atoms) for _ in range(n)]

    def test_token_classes_exclude_coordinate_and_protein_tokens(self):
        forbidden = set(dec.NON_SMILES) | set(dec.NON_ORGANIC)
        self.assertFalse(forbidden & set(CLS["atoms"]))
        for t in ("C", "N", "O", "c", "n", "Cl", "Br"):
            self.assertIn(t, CLS["atoms"])                                  # aliphatic C, N, O are legal ligand atoms

    def test_never_emits_a_forbidden_token(self):
        for toks, _, _ in self.run_walks(False, n=200):
            self.assertFalse(set(toks) & (set(dec.NON_SMILES) | set(dec.NON_ORGANIC)))

    def test_walks_terminate_and_are_syntactically_valid(self):
        for valence in (False, True):
            walks = self.run_walks(valence)
            finished = [w for w in walks if w[1]]
            self.assertGreater(len(finished), 0.85 * len(walks), f"valence={valence}: too many walks failed to terminate")
            for toks, _, st in finished:
                smi = "".join(toks)
                self.assertIsNotNone(Chem.MolFromSmiles(smi, sanitize=False), smi)   # syntax only
                self.assertEqual(smi.count("("), smi.count(")"), smi)
                self.assertFalse(st.open_rings, smi)
                self.assertGreaterEqual(len(st.syms), 4, smi)

    def test_no_bond_next_to_bond_and_no_empty_branch(self):
        for toks, _, _ in self.run_walks(False, n=300):
            for a, b in zip(toks, toks[1:]):
                self.assertFalse(a in dec.BOND_ORDER and b in dec.BOND_ORDER, "".join(toks))
                self.assertFalse(a == "(" and b == ")", "".join(toks))
                self.assertFalse(a in dec.BOND_ORDER and b in (")", "("), "".join(toks))

    def test_valence_mode_sanitizes_every_aliphatic_walk(self):
        finished = [w for w in self.run_walks(True, n=1000, seed=1, atoms=ALIPHATIC) if w[1]]
        self.assertGreater(len(finished), 300)
        bad = [("".join(t)) for t, _, _ in finished if Chem.MolFromSmiles("".join(t)) is None]
        self.assertEqual(bad, [], f"{len(bad)}/{len(finished)} failed to sanitize, e.g. {bad[:5]}")

    def test_valence_mode_never_strands_a_sample(self):
        for valence in (True,):
            for atoms in (ALIPHATIC, None):
                for _, _, st in self.run_walks(valence, n=600, seed=3, atoms=atoms):
                    self.assertFalse(st.stuck)

    def test_syntax_mode_is_not_enough_for_valence(self):
        def rate(valence):
            f = [w for w in self.run_walks(valence, n=800, seed=2, atoms=ALIPHATIC) if w[1]]
            return sum(Chem.MolFromSmiles("".join(t)) is not None for t, _, _ in f) / len(f)
        self.assertGreater(rate(True), rate(False) + 0.5)

    def test_halogen_cannot_follow_a_double_bond_in_valence_mode(self):
        st = dec.SmilesState(CLS["atoms"], CLS["digits"], valence=True)
        st.advance("C")
        st.advance("=")
        self.assertFalse({"F", "Cl", "Br", "I"} & set(st.allowed()))


if __name__ == "__main__":
    unittest.main()
