"""SA oracle data file: TDC's fpscores.pkl must come from <data>/oracle/ and must never be written to the cwd or the repo."""
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPSCORES = os.path.join(REPO, "data", "oracle", "fpscores.pkl")


@unittest.skipUnless(os.path.exists(FPSCORES), "data/oracle/fpscores.pkl not present (would trigger a download)")
class SaOracle(unittest.TestCase):
    def run_in(self, cwd):
        code = ("import sys, os; sys.path.insert(0, %r)\n"
                "import scoring_function as sf\n"
                "before = os.getcwd()\n"
                "o = sf.create_synthetic_accessibility_oracle()\n"
                "assert os.getcwd() == before, 'cwd changed'\n"
                "print('SA', o('CC(=O)Nc1ccc(O)cc1'), o('CCO'))\n") % REPO
        return subprocess.run([sys.executable, "-I", "-c", code], cwd=cwd, capture_output=True, text=True, timeout=300)

    def test_loads_from_data_folder_and_leaves_no_files_behind(self):
        with tempfile.TemporaryDirectory() as td:
            r = self.run_in(td)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            self.assertIn("SA ", r.stdout)
            values = [float(x) for x in r.stdout.split("SA ")[1].split()]
            self.assertTrue(all(1.0 <= v <= 10.0 for v in values), values)
            self.assertEqual(os.listdir(td), [])                       # nothing written to the working directory
        self.assertFalse(os.path.exists(os.path.join(REPO, "oracle")))  # nor to the repo root


if __name__ == "__main__":
    unittest.main()
