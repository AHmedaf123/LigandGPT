"""Loss-side log-probabilities must be deterministic: dropout disabled for the agent, the prior and any stored behavior policy.
Run: .venv/bin/python -m unittest discover -s tests -p 'test_policy_loss.py' -v   (needs CUDA; uses a tiny random model)"""
import os
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


if __name__ == "__main__":
    unittest.main()
