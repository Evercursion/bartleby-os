# Third-Party Benchmark Dataset Licenses & Attributions (`data/raw/`)

All `bartleby-os` source code (`bartleby/`, `tradeoff_jev/`), Lean 4 formal proofs (`lean/`), first-party synthetic decision corpora (`data/corpus_v1/`, `data/corpus_v2/`, `data/frozen_vault_eval/`), and trained `bartleby-1.2` checkpoints (`artifacts/bartleby-1.2/`) are licensed under the **Apache License 2.0**.

The files under `data/raw/` are cached snapshots of external third-party benchmarks used exclusively to verify zero-shot evaluation reproducibility (`scripts/eval_canonical_security_benchmarks.py`, `scripts/eval_canonical_benchmarks.py`, `scripts/eval_zero_bleeding_audit.py`). Each third-party dataset remains governed by its upstream license below:

| Directory / File | Upstream Dataset & Authors | Upstream License | Usage in `bartleby-os` |
| :--- | :--- | :--- | :--- |
| `data/raw/typed_decisions/` | **`LocalLLaMA/typed-decisions`**<br>Liu et al. (2026), *Decision 1.0* | **Apache License 2.0** | Training `jev_1_14` baseline & held-out SOC disposition evaluation |
| `data/raw/canonical_security/mmlu_computer_security_test.json` | **`cais/mmlu` (`computer_security`)**<br>Hendrycks et al. (2021), *Measuring Massive Multitask Language Understanding* (ICLR 2021) | **MIT License** | Zero-shot external evaluation ($N=100$) |
| `data/raw/moralchoice/` | **`MoralChoice`**<br>Scherrer et al. (2023), *Evaluating the Moral Beliefs Encoded in LLMs* (NeurIPS 2023) | **CC BY 4.0** | Zero-overlap OOD moral dilemma evaluation ($N=1,352$) |
| `data/raw/canonical_security/secqa_v1_test.csv`<br>`data/raw/canonical_security/secqa_v2_test.csv` | **`zefang-liu/secqa` (`v1` & `v2`)**<br>Liu et al. (2023), *SecQA: A Concise Question-Answering Dataset for Evaluating LLMs in Computer Security* | **CC BY-NC-SA 4.0** *(Non-Commercial Evaluation Only)* | Zero-shot external evaluation ($N=110$ `v1`, $N=100$ `v2`) |
| `data/raw/canonical_security/cti_mcq.tsv` | **`AI4Sec/CTIBench` (`cti-mcq`)**<br>Alam et al. (2024), *CTIBench: A Benchmark for Evaluating LLMs in Cyber Threat Intelligence* (NeurIPS 2024) | **CC BY-NC-SA 4.0** *(Non-Commercial Evaluation Only)* | Zero-shot external evaluation ($N=200$) |

## Commercial Redistribution Note

Because `SecQA` and `CTIBench` carry **CC BY-NC-SA 4.0** (non-commercial) licenses from their upstream authors, commercial redistributions of `bartleby-os` should omit `data/raw/canonical_security/secqa_*.csv` and `data/raw/canonical_security/cti_mcq.tsv` (or the entire `data/raw/canonical_security/` directory). Removing `data/raw/canonical_security/` has **zero impact** on `bartleby-1.2` model loading, `.decide()` inference, CLI operation, `corpus_v2` training, Frozen Blind Vault evaluation, or the `pytest` test suite. When `data/raw/canonical_security/` is absent, `scripts/eval_canonical_security_benchmarks.py` automatically downloads the evaluation splits from HuggingFace Hub at runtime.
