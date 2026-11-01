"""Loss-side log-probabilities must be deterministic: dropout disabled for the agent, the prior and any stored behavior policy.
Run: .venv/bin/python -m unittest discover -s tests -p 'test_policy_loss.py' -v   (needs CUDA; uses a tiny random model)"""
import os
import pickle
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402

CUDA = torch.cuda.is_available()
if CUDA:
    import generation_finetuning as gf  # noqa: E402
    from model import LigandGPT  # noqa: E402


def tiny(seed):
    torch.manual_seed(seed)
    return LigandGPT(vocab_size=30, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.5, context_length=64).cuda()


@unittest.skipUnless(CUDA, "CUDA required")
class DropoutDisabledInLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.seqs = torch.randint(1, 30, (4, 24)).cuda()
        self.nums = torch.ones(4, 24).cuda()

    def test_log_probs_do_not_depend_on_the_callers_mode_or_rng(self):
        m = tiny(1)
        m.train()
        torch.manual_seed(10)
        a, _ = gf.sequence_log_probs(m, self.seqs, self.nums)
        torch.manual_seed(11)
        b, _ = gf.sequence_log_probs(m, self.seqs, self.nums)
        m.eval()
        c, _ = gf.sequence_log_probs(m, self.seqs, self.nums)
        self.assertTrue(torch.equal(a, b) and torch.equal(a, c))                 # exactly reproducible, mode-independent

    def test_callers_mode_is_restored(self):
        m = tiny(1)
        for mode in (True, False):
            m.train(mode)
            gf.sequence_log_probs(m, self.seqs, self.nums)
            self.assertEqual(m.training, mode)

    def test_train_mode_would_have_been_random(self):
        m = tiny(1)
        m.train()
        torch.manual_seed(10)
        x1 = m(self.seqs[:, :-1], self.nums[:, :-1])[0]
        x2 = m(self.seqs[:, :-1], self.nums[:, :-1])[0]
        self.assertFalse(torch.equal(x1, x2))                                    # guards the test itself: dropout is active in train mode

    def test_loss_is_deterministic_and_has_gradients_for_an_agent_left_in_train_mode(self):
        agent, prior = tiny(2), tiny(3)
        prior.eval()
        for p in prior.parameters():
            p.requires_grad_(False)
        agent.train()
        adv = torch.tensor([1.0, -0.5, 0.3, 0.0]).cuda()
        outs = []
        for seed in (5, 6):
            torch.manual_seed(seed)
            loss, policy, _, _, entropy, kl = gf.compute_reinforce_loss(agent, prior, self.seqs, self.nums, adv, 0.1)
            outs.append((loss.item(), policy.item(), entropy.item(), kl))
        self.assertEqual(outs[0], outs[1])
        agent.zero_grad()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.parameters()))


@unittest.skipUnless(CUDA, "CUDA required")
class AttentionPathIsPinned(unittest.TestCase):
    def test_fused_fast_path_is_off(self):
        # With it on, the frozen prior (no grad) and the agent (grad) used different kernels: logits differed by up to 0.47
        self.assertFalse(torch.backends.mha.get_fastpath_enabled())

    def test_same_weights_give_same_log_probs_for_agent_and_frozen_prior(self):
        agent, prior = tiny(7), tiny(7)
        prior.eval()
        for p in prior.parameters():
            p.requires_grad_(False)
        agent.train()
        seqs, nums = torch.randint(1, 30, (4, 40)).cuda(), torch.ones(4, 40).cuda()
        loss, _, _, _, _, kl = gf.compute_reinforce_loss(agent, prior, seqs, nums, torch.ones(4).cuda(), 0.1)
        self.assertLess(abs(kl), 1e-4)


DATA = os.path.join(REPO, "data")
CKPT = os.path.join(REPO, "final.pt")
LMDB = os.path.join(DATA, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb")


@unittest.skipUnless(CUDA and os.path.exists(CKPT) and os.path.exists(LMDB), "needs CUDA, final.pt and the CrossDocked lmdb")
class StepZeroSanity(unittest.TestCase):
    """Agent and prior start with identical weights, so at step 0 the log-ratio and kl_to_prior must be ~0 (target 0, real pocket prompt)."""

    def test_kl_to_prior_and_log_ratio_are_zero_at_step_zero(self):
        from tokenizer import ProtLigTokenizer
        from vocabulary import read_vocabulary
        voc, tk = read_vocabulary(os.path.join(REPO, "ProtLigVoc.txt")), ProtLigTokenizer()
        split = torch.load(os.path.join(DATA, "crossdocked_pocket10_pose_split.pt"), weights_only=False)["test"]
        txn, keys = gf.get_lmdb_data(LMDB)
        dp = pickle.loads(txn.get(keys[split[0]]))
        toks, nums = tk.tokenize(protein=dp["protein_atom_name"], prot_coords=dp["protein_pos"].numpy() - dp["ligand_center_of_mass"].numpy())
        toks.append("<LIG_START>")
        nums.append(1)
        enc = torch.tensor(voc.encode(toks), dtype=torch.long).unsqueeze(0).expand(4, -1).cuda()
        num = torch.tensor(nums, dtype=torch.float).unsqueeze(0).expand(4, -1).cuda()
        make = lambda: LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()  # noqa: E731
        agent, prior = make(), make()
        agent.load_state_dict(torch.load(CKPT))
        prior.load_state_dict(torch.load(CKPT))
        prior.eval()
        for p in prior.parameters():
            p.requires_grad_(False)
        torch.manual_seed(0)
        seqs, numbers, _, _ = gf.generate_smiles_batch(agent, enc, num, 0, 700, False)      # leaves the agent in train mode, as in the RL loop
        with torch.no_grad():
            lp_prior, _ = gf.sequence_log_probs(prior, seqs, numbers)
        lp_agent, _ = gf.sequence_log_probs(agent, seqs, numbers)
        log_ratio = (lp_agent.detach() - lp_prior).sum(dim=1)
        self.assertLess(float(log_ratio.abs().max()), 1e-3, log_ratio)
        _, _, _, _, _, kl = gf.compute_reinforce_loss(agent, prior, seqs, numbers, torch.ones(4).cuda(), 0.1)
        self.assertLess(abs(kl), 1e-3, kl)


if __name__ == "__main__":
    unittest.main()
