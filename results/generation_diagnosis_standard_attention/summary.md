# Free-generation diagnosis

strict = terminated + RDKit parse + sanitize + >= 4 heavy atoms (raw SMILES, no SELFIES repair); pipeline = SELFIES round trip as in generation_finetuning.validate_and_standardize_smiles.

| setting | T0 strict | T0 pipeline | T3 strict | T3 pipeline | both strict | unique valid | mean len |
|---|---|---|---|---|---|---|---|
| plain_k10 | 0/64 | 2/64 | 3/64 | 3/64 | 0.02 | 3 | 32 |
| plain_full | 3/64 | 3/64 | 5/64 | 5/64 | 0.06 | 8 | 33 |
| current_k10 | 8/64 | 8/64 | 1/64 | 1/64 | 0.07 | 9 | 28 |
| current_rl_step0 | 7/64 | 8/64 | 0/64 | 0/64 | 0.05 | 6 | 26 |
| struct_syntax_k10 | 9/64 | 12/64 | 5/64 | 5/64 | 0.11 | 14 | 65 |
| struct_syntax_full | 7/64 | 7/64 | 6/64 | 8/64 | 0.10 | 13 | 64 |
| struct_valence_k10 | 25/64 | 27/64 | 27/64 | 31/64 | 0.41 | 52 | 65 |
| struct_valence_full | 17/64 | 18/64 | 26/64 | 30/64 | 0.34 | 43 | 57 |

## Failure breakdown (count of samples, both targets pooled)

| category | plain_k10 | plain_full | current_k10 | current_rl_step0 | struct_syntax_k10 | struct_syntax_full | struct_valence_k10 | struct_valence_full |
|---|---|---|---|---|---|---|---|---|
| no_termination | 0 | 0 | 0 | 0 | 1 | 5 | 3 | 0 |
| rdkit:kekulize(aromatic ring invalid) | 9 | 9 | 30 | 46 | 16 | 25 | 70 | 84 |
| rdkit:valence | 11 | 11 | 22 | 19 | 97 | 85 | 3 | 1 |
| syntax:other_syntax | 4 | 3 | 3 | 2 | 0 | 0 | 0 | 0 |
| syntax:unclosed_branch | 15 | 26 | 2 | 4 | 0 | 0 | 0 | 0 |
| syntax:unclosed_ring_digit | 50 | 48 | 62 | 50 | 0 | 0 | 0 | 0 |
| syntax:unmatched_close_paren | 36 | 23 | 0 | 0 | 0 | 0 | 0 | 0 |
| valid | 3 | 8 | 9 | 7 | 14 | 13 | 52 | 43 |

## Atom token usage (share of atom tokens)

| setting | aliphatic C/N/O | aromatic c/n/o | halogens | bracket atoms |
|---|---|---|---|---|
| plain_k10 | 0.63 | 0.33 | 0.01 | 0.00 |
| plain_full | 0.59 | 0.37 | 0.02 | 0.00 |
| current_k10 | 0.00 | 0.97 | 0.03 | 0.00 |
| current_rl_step0 | 0.00 | 0.97 | 0.03 | 0.00 |
| struct_syntax_k10 | 0.71 | 0.22 | 0.03 | 0.00 |
| struct_syntax_full | 0.72 | 0.22 | 0.03 | 0.00 |
| struct_valence_k10 | 0.73 | 0.22 | 0.01 | 0.00 |
| struct_valence_full | 0.71 | 0.24 | 0.01 | 0.00 |
