"""RL core checks: loss-side log-probs are deterministic, the attention path is pinned, the update behaves at step 0, the stage function is the
only source of stages. Run: .venv/bin/python -m unittest discover -s tests -p 'test_policy_loss.py' -v   (needs CUDA; tiny random models)"""
import argparse
import os
import pickle
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402

import mo_mpo  # noqa: E402

CUDA = torch.cuda.is_available()
if CUDA:
    import decoding as dec  # noqa: E402
    import generation_finetuning as gf  # noqa: E402
    from model import LigandGPT  # noqa: E402


def tiny(seed):
    torch.manual_seed(seed)
    return LigandGPT(vocab_size=30, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.5, context_length=128).cuda()


class Stage(unittest.TestCase):
    def test_equal_thirds(self):
        self.assertEqual([mo_mpo.resolve_stage(s, 100) for s in (0, 32, 33, 65, 66, 99)], [1, 1, 2, 2, 3, 3])
        self.assertEqual([mo_mpo.resolve_stage(s, 700) for s in (0, 232, 233, 465, 466, 699)], [1, 1, 2, 2, 3, 3])

    def test_no_second_source_of_stages(self):
        src = open(os.path.join(REPO, "mo_mpo.py")).read() + open(os.path.join(REPO, "generation_finetuning.py")).read()
        for forbidden in ("current_stage", "self.curriculum_stage", "step < 200", "step < 400", "current_step < 200"):
            self.assertNotIn(forbidden, src)


@unittest.skipUnless(CUDA, "CUDA required")
class LossSideLogProbs(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.seqs, self.nums = torch.randint(1, 30, (4, 24)).cuda(), torch.ones(4, 24).cuda()

    def test_deterministic_and_mode_restored(self):
        m = tiny(1)
        m.train()
        torch.manual_seed(10)
        a, _ = gf.sequence_log_probs(m, self.seqs, self.nums)
        torch.manual_seed(11)
        b, _ = gf.sequence_log_probs(m, self.seqs, self.nums)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(m.training)
        self.assertFalse(torch.equal(m(self.seqs[:, :-1], self.nums[:, :-1])[0], m(self.seqs[:, :-1], self.nums[:, :-1])[0]))   # dropout really is on in train mode

    def test_fast_path_off_and_same_weights_same_log_probs(self):
        self.assertFalse(torch.backends.mha.get_fastpath_enabled())
        agent, prior = tiny(7), tiny(7)
        prior.eval()
        for p in prior.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            lp_prior, _ = gf.sequence_log_probs(prior, self.seqs, self.nums)
        lp_agent, _ = gf.sequence_log_probs(agent, self.seqs, self.nums)
        self.assertLess(float((lp_agent.detach() - lp_prior).abs().max()), 1e-4)

    def test_log_probs_are_renormalised_over_the_grammar_mask(self):
        """The loss must see the distribution the sampler sampled from; otherwise probability can leak onto tokens the sampler never allows."""
        m = tiny(1)
        allowed = torch.zeros(4, 23, 30, dtype=torch.bool).cuda()
        allowed[..., :7] = True
        allowed[torch.arange(4), :, self.seqs[:, 1:].min().item()] = True
        allowed.scatter_(2, self.seqs[:, 1:].unsqueeze(-1), True)               # the sampled token is always allowed
        lp, full = gf.sequence_log_probs(m, self.seqs, self.nums, allowed)
        p = full.exp()
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5))     # sums to 1 over the allowed set
        self.assertEqual(float(p.masked_fill(allowed, 0).sum()), 0.0)                         # and nothing outside it
        self.assertTrue(bool(torch.isfinite(lp).all()))

    def test_mask_is_generated_tokens_only(self):
        prompt = torch.tensor([[5, 6, 7, 8]]).cuda()
        pn = torch.tensor([[1.0, 0.25, -0.5, 1.0]]).cuda()
        ids, nums, mask, allowed = gf.pack(prompt, pn, [[9, 10, 11], [12]])
        self.assertEqual(nums[0].tolist(), [1.0, 0.25, -0.5, 1.0, 1.0, 1.0, 1.0])        # real prompt numbers (pocket coordinates), 1 for generated tokens
        self.assertIsNone(allowed)
        self.assertEqual(mask.sum(1).tolist(), [3.0, 1.0])
        self.assertEqual(mask[0].tolist(), [0, 0, 0, 1, 1, 1])           # logits at positions 3,4,5 predict the three generated tokens
        self.assertTrue(torch.equal(ids[1, 4:], torch.tensor([12, 0, 0]).cuda()))


@unittest.skipUnless(CUDA, "CUDA required")
class PolicyUpdate(unittest.TestCase):
    def test_step_zero_agent_equals_prior(self):
        """Identical weights, behaviour logp stored from the same agent: kl_to_prior and the log-ratio are ~0, fresh samples are in band."""
        agent, prior = tiny(2), tiny(2)
        prior.eval()
        for p in prior.parameters():
            p.requires_grad_(False)
        prompt = torch.randint(1, 30, (1, 20)).cuda()
        gens = [torch.randint(1, 30, (n,)).tolist() for n in (5, 9, 7, 12, 6, 8)]
        pn = torch.rand(1, 20).cuda() + 0.5
        behaviour = gf.behaviour_logp(agent, prompt, pn, gens)
        opt = torch.optim.SGD(agent.parameters(), lr=0.0)
        cfg = argparse.Namespace(micro_batch=4, pos_adv_weight=1.0, log_ratio_clamp=0.7, kl_coef=1.0, ent_coef=0.005, grad_clip=5.0)
        adv = torch.tensor([1.0, -1.0, 0.5, -0.2, 0.0, 2.0]).cuda()
        st = gf.policy_update(agent, prior, opt, prompt, pn, gens, None, behaviour, adv, len(gens), cfg)
        self.assertFalse(st["nan"])
        self.assertLess(abs(st["kl_to_prior"]), 1e-3)
        self.assertLess(abs(st["log_ratio_mean"]), 1e-3)
        self.assertEqual(st["percent_in_band"], 100.0)
        self.assertGreater(st["grad_norm"], 0.0)

    def test_pos_adv_weight_changes_the_gradient(self):
        agent, prior = tiny(3), tiny(4)
        prior.eval()
        prompt = torch.randint(1, 30, (1, 20)).cuda()
        gens = [torch.randint(1, 30, (n,)).tolist() for n in (5, 9, 7, 6)]
        pn = torch.rand(1, 20).cuda() + 0.5
        behaviour = gf.behaviour_logp(agent, prompt, pn, gens)
        adv = torch.tensor([1.0, -1.0, 1.0, -1.0]).cuda()
        norms = []
        for w in (1.0, 5.0):
            opt = torch.optim.SGD(agent.parameters(), lr=0.0)
            cfg = argparse.Namespace(micro_batch=2, pos_adv_weight=w, log_ratio_clamp=0.7, kl_coef=0.0, ent_coef=0.0, grad_clip=1e9)
            norms.append(gf.policy_update(agent, prior, opt, prompt, pn, gens, None, behaviour, adv, 4, cfg)["grad_norm"])
        self.assertNotAlmostEqual(norms[0], norms[1], places=3)


DATA = os.path.join(REPO, "data")
CKPT = os.path.join(REPO, "final.pt")
LMDB = os.path.join(DATA, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb")


@unittest.skipUnless(CUDA and os.path.exists(CKPT) and os.path.exists(LMDB), "needs CUDA, final.pt and the CrossDocked lmdb")
class StepZeroRealCheckpoint(unittest.TestCase):
    def test_log_ratio_of_sampled_sequences_is_zero_for_identical_weights(self):
        from tokenizer import ProtLigTokenizer
        from vocabulary import read_vocabulary
        voc, tk = read_vocabulary(os.path.join(REPO, "ProtLigVoc.txt")), ProtLigTokenizer()
        split = torch.load(os.path.join(DATA, "crossdocked_pocket10_pose_split.pt"), weights_only=False)["test"]
        txn, keys = gf.get_lmdb_data(LMDB)
        dp = pickle.loads(txn.get(keys[split[0]]))
        toks, nums = tk.tokenize(protein=dp["protein_atom_name"], prot_coords=dp["protein_pos"].numpy() - dp["ligand_center_of_mass"].numpy())
        toks.append("<LIG_START>")
        nums.append(1)
        prompt = torch.tensor(voc.encode(toks), dtype=torch.long).unsqueeze(0).cuda()
        pn = torch.tensor(nums, dtype=torch.float).unsqueeze(0).cuda()
        make = lambda: LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()  # noqa: E731
        agent, prior = make(), make()
        agent.load_state_dict(torch.load(CKPT))
        prior.load_state_dict(torch.load(CKPT))
        prior.eval()
        for p in prior.parameters():
            p.requires_grad_(False)
        torch.manual_seed(0)
        rows, _, step_masks = dec.sample_smiles(agent, voc, prompt, pn, 4, structured=True, valence=True, top_k=10, max_new_tokens=60, return_masks=True)
        end = voc[dec.END]
        gens = [r[:r.index(end) + 1] if end in r else r for r in rows]
        masks = [step_masks[i, :len(g)] for i, g in enumerate(gens)]
        b = gf.behaviour_logp(agent, prompt, pn, gens, masks)
        ids, n_, mask, allowed = gf.pack(prompt, pn, gens, masks)
        with torch.no_grad():
            lp, _ = gf.sequence_log_probs(prior, ids, n_, allowed)
        self.assertLess(float((b - (lp * mask).sum(1)).abs().max()), 1e-3)

    def test_training_forward_matches_the_sampler_with_the_real_pocket_numbers(self):
        """Regression: pack() once set every number to 1, silently deleting the pocket coordinates from the loss-side forward pass."""
        from tokenizer import ProtLigTokenizer
        from vocabulary import read_vocabulary
        voc, tk = read_vocabulary(os.path.join(REPO, "ProtLigVoc.txt")), ProtLigTokenizer()
        split = torch.load(os.path.join(DATA, "crossdocked_pocket10_pose_split.pt"), weights_only=False)["test"]
        txn, keys = gf.get_lmdb_data(LMDB)
        dp = pickle.loads(txn.get(keys[split[0]]))
        toks, nums = tk.tokenize(protein=dp["protein_atom_name"], prot_coords=dp["protein_pos"].numpy() - dp["ligand_center_of_mass"].numpy())
        toks.append("<LIG_START>")
        nums.append(1)
        prompt = torch.tensor(voc.encode(toks), dtype=torch.long).unsqueeze(0).cuda()
        pn = torch.tensor(nums, dtype=torch.float).unsqueeze(0).cuda()
        self.assertGreater(float((pn != 1).sum()), 100)                                  # the prompt really carries coordinates
        agent = LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()
        agent.load_state_dict(torch.load(CKPT))
        agent.eval()
        torch.manual_seed(0)
        rows, _, step_masks = dec.sample_smiles(agent, voc, prompt, pn, 4, structured=True, valence=True, top_k=10, max_new_tokens=40, return_masks=True)
        end = voc[dec.END]
        gens = [r[:r.index(end) + 1] if end in r else r for r in rows]
        ids, n_, mask, _ = gf.pack(prompt, pn, gens, [step_masks[i, :len(g)] for i, g in enumerate(gens)])
        L, T = prompt.shape[1], max(len(g) for g in gens)
        with torch.no_grad():
            ref = agent(ids[:, :-1], n_[:, :-1])[0][:, L - 1:L - 1 + T]
            kvp, lg, _ = dec.prefill(agent, prompt, pn)
            kvn, got = [None] * 12, [lg.expand(4, -1)]
            for t in range(T - 1):
                lg, _ = dec.decode_step(agent, ids[:, L + t:L + t + 1], torch.ones(4, 1).cuda(), L + t, kvp, kvn)
                got.append(lg)
        m = mask[:, L - 1:L - 1 + T] > 0
        self.assertLess(float((torch.stack(got, 1) - ref).abs()[m].max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
