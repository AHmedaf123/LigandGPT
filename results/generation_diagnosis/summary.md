# Free-generation diagnosis

strict = terminated + RDKit parse + sanitize + >= 4 heavy atoms (raw SMILES, no SELFIES repair); pipeline = SELFIES round trip as in generation_finetuning.validate_and_standardize_smiles.

| setting | T0 strict | T0 pipeline | T3 strict | T3 pipeline | both strict | unique valid | mean len |
|---|---|---|---|---|---|---|---|
| plain_k10 | 0/64 | 2/64 | 3/64 | 3/64 | 0.02 | 3 | 32 |
| plain_full | 3/64 | 3/64 | 5/64 | 5/64 | 0.06 | 8 | 33 |
| current_k10 | 8/64 | 8/64 | 1/64 | 1/64 | 0.07 | 9 | 28 |
| current_rl_step0 | 9/64 | 10/64 | 0/64 | 0/64 | 0.07 | 8 | 26 |
| struct_syntax_k10 | 8/64 | 11/64 | 6/64 | 6/64 | 0.11 | 14 | 65 |
| struct_syntax_full | 7/64 | 7/64 | 6/64 | 8/64 | 0.10 | 13 | 66 |
| struct_valence_k10 | 25/64 | 27/64 | 27/64 | 30/64 | 0.41 | 52 | 64 |
| struct_valence_full | 17/64 | 19/64 | 26/64 | 30/64 | 0.34 | 43 | 58 |

## Failure breakdown (count of samples, both targets pooled)

| category | plain_k10 | plain_full | current_k10 | current_rl_step0 | struct_syntax_k10 | struct_syntax_full | struct_valence_k10 | struct_valence_full |
|---|---|---|---|---|---|---|---|---|
| no_termination | 0 | 0 | 0 | 0 | 5 | 5 | 2 | 0 |
| rdkit:kekulize(aromatic ring invalid) | 10 | 9 | 28 | 46 | 18 | 24 | 72 | 83 |
| rdkit:valence | 11 | 11 | 25 | 18 | 91 | 86 | 2 | 2 |
| syntax:other_syntax | 3 | 3 | 3 | 2 | 0 | 0 | 0 | 0 |
| syntax:unclosed_branch | 16 | 26 | 3 | 4 | 0 | 0 | 0 | 0 |
| syntax:unclosed_ring_digit | 50 | 47 | 60 | 49 | 0 | 0 | 0 | 0 |
| syntax:unmatched_close_paren | 35 | 24 | 0 | 0 | 0 | 0 | 0 | 0 |
| valid | 3 | 8 | 9 | 9 | 14 | 13 | 52 | 43 |

## Atom token usage (share of atom tokens)

| setting | aliphatic C/N/O | aromatic c/n/o | halogens | bracket atoms |
|---|---|---|---|---|
| plain_k10 | 0.62 | 0.34 | 0.01 | 0.00 |
| plain_full | 0.59 | 0.37 | 0.02 | 0.00 |
| current_k10 | 0.00 | 0.97 | 0.03 | 0.00 |
| current_rl_step0 | 0.00 | 0.97 | 0.03 | 0.00 |
| struct_syntax_k10 | 0.70 | 0.23 | 0.03 | 0.00 |
| struct_syntax_full | 0.72 | 0.21 | 0.03 | 0.00 |
| struct_valence_k10 | 0.72 | 0.22 | 0.01 | 0.00 |
| struct_valence_full | 0.72 | 0.24 | 0.01 | 0.00 |
