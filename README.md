# Knowledge-Editing-Experiments

Experiments on **sequential (lifelong) knowledge editing** for large language
models: changing specific facts stored in a pretrained LLM's weights, edit
after edit, without retraining it and without damaging what it already knows.

The project is organized in three stages:

1. **Reproduce AlphaEdit** (ICLR 2025) and the classic locate-then-edit baselines. *(current)*
2. **Implement StableEdit** from *More Edits, More Stable: Understanding the
   Lifelong Normalization in Sequential Model Editing* (ICML 2026) as a second,
   stronger baseline for comparison. *(next)*
3. **Design our own editing technique** that outperforms both on edit success,
   knowledge preservation, and long-horizon stability. *(planned)*

---

## Background

A knowledge edit rewrites a fact such as:

```text
The mother tongue of Danielle Darrieux is French   →   ... is English
```

A good editor must satisfy three goals at once:

| Metric | Question it answers |
|---|---|
| **Efficacy** | Does the model now produce the new answer? |
| **Generalization** | Does the edit hold under paraphrased prompts? |
| **Specificity / Locality** | Are unrelated (neighborhood) facts left intact? |

In the **sequential** setting edits arrive continuously (thousands, ideally
millions). Most editors then degrade: weight-update norms grow, earlier edits
are forgotten, and general ability (e.g., GLUE) collapses. This degradation is
the core problem this project studies.

## Methods

### AlphaEdit (implemented)

AlphaEdit is a MEMIT-style locate-then-edit method. For each batch of edits it:

1. computes the key representation `k` of the subject at selected MLP layers,
2. optimizes a target value `z` that produces the new answer,
3. solves a least-squares weight update for the MLP down-projection, and
4. projects that update onto the **null space** of preserved-knowledge keys,
   so preserved outputs are (approximately) unchanged:

```text
W_new = W_old + Δ · P        P = U_small U_smallᵀ   (from SVD of E[k kᵀ] on Wikipedia)
```

See [AlphaEdit-Explain.md](AlphaEdit-Explain.md) for a detailed walkthrough of
the data, the null-space statistics, and the evaluation pipeline, and
[AlphaEdit-Original-README.md](AlphaEdit-Original-README.md) for the upstream
instructions.

### StableEdit — Lifelong Normalization (next step)

*More Edits, More Stable* (Ma et al., ICML 2026,
[arXiv:2605.11836](https://arxiv.org/abs/2605.11836),
[code](https://github.com/MINE-USTC/StableEdit)) shows that several recent
lifelong editors share a **Lifelong Normalization (LN)** module, which
normalizes value gradients using running statistics accumulated across edits.
Its main points:

- Removing LN makes editing performance collapse immediately.
- Editing has a *positive cumulative effect*: running the full edit sequence
  beats starting midway, so earlier edits help later ones.
- In theory, LN combined with ridge-regularized regression gives updates that
  are **asymptotically orthogonal** and **norm-bounded**, which limits
  forgetting and model collapse.
- Based on this analysis they propose **StableEdit**, which adds an explicit
  warm-up stage and full whitening, and they report strong results at up to
  2M edits.

Planned work:

- [ ] Port StableEdit into `src/baselines/stable_edit/`, keeping the same data
      loaders and evaluation so results are directly comparable.
- [ ] Run AlphaEdit, MEMIT, and StableEdit under identical sequential settings
      (same model, edit order, batch size, and total edits).
- [ ] Track efficacy, generalization, and specificity across the edit sequence,
      plus fluency, GLUE downstream scores, and weight-update norms.
- [ ] Ablate StableEdit's components (LN on/off, warm-up, whitening).

### Our method (planned)

After the comparison, we will develop our own technique in `src/ours/`. The
goal is to combine the strengths of both families: AlphaEdit's null-space
protection of preserved knowledge and StableEdit's stability from Lifelong
Normalization. We then aim to beat both on long edit sequences. The design
will follow from the failure modes we observe in stage 2. Results will be
written to `results/Ours/`.

---

## Repository layout

```text
AlphaEdit/        AlphaEdit update algorithm (compute_ks, compute_z, main)
rome/             ROME + covariance / layer-stats utilities
memit/            MEMIT baseline
nse/              NSE (neuron-level sequential editing) baseline
baselines/        FT and MEND
dsets/            CounterFact, MultiCounterFact, ZsRE, MQuAKE loaders
experiments/      evaluate.py, summarize.py, causal tracing, sweeps
glue_eval/        downstream general-ability evaluation
hparams/          per-algorithm, per-model hyperparameters
util/             model hooks and shared helpers
src/
  baselines/      cleaned-up baseline runners (alpha_edit, memit, rome, ...)
  ours/           our method (planned)
scripts/          dataset preparation and experiment launch scripts
data/             datasets and cached covariance stats (large files git-ignored)
results/          run outputs: results/<method>/run_<id>/
globals.yml       paths for results, data, stats, hparams
```

## Setup

Requires a CUDA GPU (the original AlphaEdit recommends ≥ 48 GB for Llama3-8B;
GPT-2 XL runs on less).

```bash
pip install -r requirements.txt
python scripts/prepare_datasets.py      # download CounterFact / ZsRE / MQuAKE
```

Covariance statistics (`data/stats/`) are computed from Wikipedia on first run
and cached. They are large (~156 MB per layer for GPT-2 XL), so they are not
tracked in git.

## Running experiments

AlphaEdit on GPT-2 XL with MultiCounterFact, 2,000 edits applied sequentially
in batches of 100:

```bash
python3 -m experiments.evaluate \
  --alg_name=AlphaEdit \
  --model_name=gpt2-xl \
  --hparams_fname=gpt2-xl.json \
  --ds_name=mcf \
  --dataset_size_limit=2000 \
  --num_edits=100 \
  --downstream_eval_steps=5
```

Summarize results:

```bash
python -m experiments.summarize --dir_name=AlphaEdit --runs=run_000
```

The custom runner for CounterFact / ZsRE / MQuAKE is invoked by:

```bash
bash scripts/run_experiment_alphaedit.sh
```

Supported models in `hparams/`: GPT-2 XL, GPT-J 6B, Llama3-8B, Phi-1.5. The
`gemma-4-E2B` configuration is still being adapted; its module names and
layers need to be matched to the Gemma architecture.

## Comparison plan

| Method | Type | Status |
|---|---|---|
| FT / ROME / MEMIT / NSE | baselines | available |
| AlphaEdit | null-space projection | reproduced |
| StableEdit | lifelong normalization | next |
| **Ours** | — | planned |

All methods will share the same model, datasets, edit order, and evaluation
scripts so that differences reflect the method only.

## References

- Fang et al. *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR 2025.
- Ma et al. *More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing.* ICML 2026. [arXiv:2605.11836](https://arxiv.org/abs/2605.11836)
- Meng et al. *Locating and Editing Factual Associations in GPT (ROME).* NeurIPS 2022.
- Meng et al. *Mass-Editing Memory in a Transformer (MEMIT).* ICLR 2023.
- Jiang et al. *Neuron-Level Sequential Editing for Large Language Models (NSE).* 2024.

## Acknowledgment

Built on the original [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit)
codebase, which in turn builds on ROME and MEMIT. See
`AlphaEdit-Original-LICENSE`.
