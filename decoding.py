"""Structured SMILES decoding for LigandGPT.

The ligand part of a sequence is  <LIG_START> SMILES tokens <LIG_END> [<LIG_COORDS_START> (x y z)*n <LIG_COORDS_END>].
Inside the SMILES block there are only atom, bond, branch and ring-digit tokens. The x/y/z tokens belong to the
separate coordinate blocks. This module generates the SMILES block only, using a per-sample state machine that
turns the SMILES grammar into a logit mask at every step. It adds NO bias towards particular atoms: it only removes
tokens that cannot legally come next. The sampling temperature is untouched.

Two levels:
  valence=False  token-type rules + SMILES syntax (balanced branches, ring digits opened and closed exactly once,
                 no bond next to a bond, no empty branch, no ring closure onto a neighbour or onto itself)
  valence=True   additionally tracks bond orders so that no organic atom exceeds its maximum valence

Nothing here is imported by the RL loop. predict_smiles in training_utils.py is unchanged.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

# tokens that never belong inside the SMILES block
NON_SMILES = ("<PAD>", "<PROT_START>", "<PROT_END>", "<PROT_COORDS_START>", "<PROT_COORDS_END>", "<LIG_START>",
              "<LIG_COORDS_START>", "<LIG_COORDS_END>", "[x]", "[y]", "[z]", "CA", ".", "/", "%10", "%11")
# isotopes, metals and other non-organic tokens (the same list training_utils.isotope_tokens masks)
NON_ORGANIC = ("[17O]", "[18F]", "[2H]", "[As]", "[Fe]", "[Ca+2]", "[K]", "[K+]", "[Li+]", "[Mg+2]", "[Na]", "[Na+]",
               "[Sn]", "[V]")
BOND_ORDER = {"-": 1, "=": 2, "#": 3}
RING_DIGITS = tuple("123456789")
END = "<LIG_END>"
AROMATIC = {"c", "n", "o", "s", "p"}
MAX_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 6, "P": 5, "B": 3, "F": 1, "Cl": 1, "Br": 1, "I": 1,
               "c": 4, "n": 3, "o": 2, "s": 2, "p": 3}   # aromatic ring bonds count 1 each, so c(=O) uses 4


def token_classes(vocab) -> Dict[str, List[str]]:
    """Split the vocabulary into the token classes the grammar needs."""
    atoms, other = [], []
    for t in vocab.tokens():
        if t in NON_SMILES or t in NON_ORGANIC or t in BOND_ORDER or t in RING_DIGITS or t in ("(", ")", END):
            continue
        atoms.append(t)
    return {"atoms": sorted(atoms), "bonds": list(BOND_ORDER), "digits": [d for d in RING_DIGITS if d in vocab]}


@dataclass
class SmilesState:
    """Grammar state of one sample. advance() consumes a token, allowed() lists the legal next tokens."""
    atoms_all: List[str]
    digits_all: List[str]
    valence: bool = False
    min_atoms: int = 4
    syms: List[str] = field(default_factory=list)
    used: List[int] = field(default_factory=list)            # valence used so far (valence mode)
    nbrs: List[Set[int]] = field(default_factory=list)
    cur: Optional[int] = None
    stack: List[Optional[int]] = field(default_factory=list)
    pending: Optional[int] = None                            # bond order of a bond token not yet followed by an atom
    last: str = "start"                                      # start | atom | bond | open | close | ring
    bond_ctx: str = "start"                                  # what preceded the pending bond token
    open_rings: Dict[str, tuple] = field(default_factory=dict)
    done: bool = False
    stuck: bool = False

    # ---- helpers -----------------------------------------------------------------------------------------------
    def _maxval(self, i: int) -> int:
        return MAX_VALENCE.get(self.syms[i], 99)             # bracket atoms: unconstrained

    def _room(self, i: int) -> int:
        return self._maxval(i) - self.used[i]

    def _atom_cap(self, sym: str) -> int:
        return MAX_VALENCE.get(sym, 99)

    def _room_ok(self, i: int, need: int) -> bool:
        return (not self.valence) or self._maxval(i) == 99 or self._room(i) >= need

    def _can_close(self, d: str) -> bool:
        a, order = self.open_rings[d]
        c = self.cur
        if c is None or c == a or a in self.nbrs[c]:
            return False                                     # ring closure onto itself or onto a bonded neighbour
        if self.pending is not None and self.pending != order:
            return False                                     # a ring bond keeps the order it was opened with
        need = order
        if not self._room_ok(c, need):
            return False
        left = (self._room(c) - need) if (self.valence and self._maxval(c) != 99) else 99
        return not self._strands(left, len(self.syms), len(self.open_rings) - 1, len(self.stack))

    def _can_open(self) -> bool:
        if self.cur is None or self.last in ("open", "close", "start"):
            return False
        return self._room_ok(self.cur, (self.pending or 1) + 1)   # one slot for the ring bond, one to keep the chain going

    def _atoms_ok(self) -> List[str]:
        if not self.valence or self.cur is None:
            return list(self.atoms_all)
        order = self.pending or 1
        if not self._room_ok(self.cur, order + (1 if self.last == "open" else 0)):   # inside "(" the parent must keep a slot to continue
            return []
        return [a for a in self.atoms_all if self._atom_cap(a) >= order
                and not self._strands(self._atom_cap(a) - order, len(self.syms) + 1, len(self.open_rings), len(self.stack))]

    def _strands(self, room_left: int, n_atoms_after: int, rings_after: int, stack_after: int) -> bool:
        """True if the current atom would have no free valence while the sample still needs more tokens
        (open branch handled by ')', otherwise open rings or too few atoms leave nowhere to go)."""
        if not self.valence or room_left > 0:
            return False
        return not (stack_after > 0 or (rings_after == 0 and n_atoms_after >= self.min_atoms))

    def _bonds_ok(self) -> List[str]:
        extra = 1 if self.last == "open" else 0
        return [b for b in BOND_ORDER if self._room_ok(self.cur, BOND_ORDER[b] + extra)]

    def _ring_digits_ok(self, opening_allowed: bool) -> List[str]:
        out = []
        for d in self.digits_all:
            if d in self.open_rings:
                if self._can_close(d):
                    out.append(d)
            elif opening_allowed and self._can_open():
                out.append(d)
        return out

    # ---- grammar -----------------------------------------------------------------------------------------------
    def allowed(self, steps_left: int = 10 ** 9) -> List[str]:
        if self.done:
            return []
        last = self.last
        closing_only = steps_left <= len(self.open_rings) + len(self.stack) + 2    # budget nearly gone: only close things
        can_end = (not self.open_rings) and (not self.stack) and self.pending is None \
            and len(self.syms) >= self.min_atoms and last in ("atom", "ring", "close")
        out: List[str] = []
        if last == "start":
            return self._atoms_ok()
        if last in ("atom", "ring", "close"):
            out += self._atoms_ok() + self._bonds_ok()
            if self._can_open_branch():
                out.append("(")
            if self.stack and self._can_return():
                out.append(")")
            if last in ("atom", "ring"):                     # ring digits attach to the atom just written, never after ')'
                out += self._ring_digits_ok(True)
            if can_end:
                out.append(END)
        elif last == "bond":                                 # a ring-bond digit may only follow a bond written right after an atom
            out += self._atoms_ok() + (self._ring_digits_ok(True) if self.bond_ctx in ("atom", "ring") else [])
        elif last == "open":
            out += self._atoms_ok() + self._bonds_ok()       # a branch starts with an atom or a bond, never ')' or a digit
        if closing_only:                                     # budget nearly gone: close what is open, else add a spacer atom
            closers = [t for t in out if t in self.open_rings or t in (")", END)]
            out = closers or [t for t in out if t in self.atoms_all] or out
        if not out:
            self.stuck = True
        return out

    def _can_return(self) -> bool:
        p = self.stack[-1]
        if p is None or not self.valence or self._maxval(p) == 99:
            return True
        return not self._strands(self._room(p), len(self.syms), len(self.open_rings), len(self.stack) - 1)

    def _can_open_branch(self) -> bool:
        return self.cur is not None and self.last in ("atom", "ring", "close") and self._room_ok(self.cur, 2)

    def _connect(self, a: int, b: int, order: int) -> None:
        self.nbrs[a].add(b)
        self.nbrs[b].add(a)
        if self.valence:
            self.used[a] += order
            self.used[b] += order

    def advance(self, tok: str) -> None:
        if tok == END:
            self.done = True
            return
        if tok in BOND_ORDER:
            self.pending = BOND_ORDER[tok]
            self.bond_ctx = self.last
            self.last = "bond"
        elif tok == "(":
            self.stack.append(self.cur)
            self.last = "open"
        elif tok == ")":
            self.cur = self.stack.pop()
            self.last = "close"
        elif tok in RING_DIGITS:
            if tok in self.open_rings:
                a, order = self.open_rings.pop(tok)
                if self.valence:
                    self.used[self.cur] += order
                self.nbrs[a].add(self.cur)
                self.nbrs[self.cur].add(a)
            else:
                order = self.pending or 1
                self.open_rings[tok] = (self.cur, order)
                if self.valence:
                    self.used[self.cur] += order
            self.pending = None
            self.last = "ring"
        else:                                                # an atom token
            idx = len(self.syms)
            self.syms.append(tok)
            self.used.append(0)
            self.nbrs.append(set())
            if self.cur is not None:
                self._connect(self.cur, idx, self.pending or 1)
            self.cur = idx
            self.pending = None
            self.last = "atom"


def build_mask(states: List[Optional[SmilesState]], vocab, steps_left: int, device) -> torch.Tensor:
    """Boolean [B, V] mask of legal next tokens. Finished samples (state None) allow only <PAD> (id 0)."""
    V = len(vocab)
    mask = torch.zeros(len(states), V, dtype=torch.bool)
    for i, st in enumerate(states):
        if st is None or st.done:
            mask[i, 0] = True
            continue
        ids = [vocab[t] for t in st.allowed(steps_left)]
        if not ids:
            mask[i, vocab[END]] = True                       # nothing legal: stop, the sample is scored as it stands
        else:
            mask[i, ids] = True
    return mask.to(device)


@torch.no_grad()
def sample_smiles(model, vocab, prompt_ids: torch.Tensor, prompt_nums: torch.Tensor, n: int, *, structured: bool,
                  temperature: float = 1.0, top_k: int = 10, max_new_tokens: int = 200, valence: bool = False,
                  min_atoms: int = 4, generator: Optional[torch.Generator] = None):
    """Sample n ligands for one pocket prompt.

    prompt_ids / prompt_nums: [1, L] pocket tokens (with coordinates) ending in <LIG_START>.
    structured=False reproduces the original 3DMolFormer sampler (no masks, temperature, top-k).
    Returns (generated token ids [n, T] as a list of lists, finished flags). Generation stops per sample at <LIG_END>.
    """
    device = prompt_ids.device
    ids = prompt_ids.expand(n, -1).clone()
    nums = prompt_nums.expand(n, -1).clone()
    cls = token_classes(vocab)
    states = [SmilesState(cls["atoms"], cls["digits"], valence=valence, min_atoms=min_atoms) if structured else None
              for _ in range(n)]
    end_id = vocab[END]
    done = torch.zeros(n, dtype=torch.bool, device=device)
    coord_ids = torch.tensor([vocab["[x]"], vocab["[y]"], vocab["[z]"]], device=device)
    model.eval()
    for step in range(max_new_tokens):
        logits, num_pred = model(ids, nums)
        logits = logits[:, -1, :].float()
        if structured:
            mask = build_mask([None if done[i] else states[i] for i in range(n)], vocab, max_new_tokens - step, device)
            logits = logits.masked_fill(~mask, float("-inf"))
        logits = logits / temperature
        if top_k and top_k > 0:
            v, ix = torch.topk(logits, min(top_k, logits.shape[-1]))
            probs = torch.softmax(v, dim=-1)
            nxt = ix.gather(-1, torch.multinomial(probs, 1, generator=generator)).squeeze(-1)
        else:
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(-1)
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)
        new_num = torch.ones(n, device=device)
        is_coord = torch.isin(nxt, coord_ids)
        new_num = torch.where(is_coord, num_pred[:, -1, 0], new_num)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        nums = torch.cat([nums, new_num[:, None]], dim=1)
        if structured:
            for i in range(n):
                if not done[i]:
                    states[i].advance(vocab[int(nxt[i])])
        done = done | (nxt == end_id)
        if bool(done.all()):
            break
    return ids[:, prompt_ids.shape[1]:].tolist(), done.tolist()
