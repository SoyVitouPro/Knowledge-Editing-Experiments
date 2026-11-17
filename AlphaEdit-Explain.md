# Original AlphaEdit Implementation

The original AlphaEdit source is kept at the repository root. The temporary
local datasets and `Original_Implementation/` directory were removed.

```text
AlphaEdit/      AlphaEdit update algorithm
baselines/      FT and MEND implementations
dsets/          CounterFact, ZsRE, and MQuAKE loaders
experiments/    evaluation and result summarization
hparams/        algorithm/model hyperparameters
memit/          MEMIT baseline
nse/            NSE baseline
rome/           ROME and covariance utilities
util/           model hooks and shared utilities
```

## What AlphaEdit does

AlphaEdit edits the weights of an already-trained language model. It does not
train a new language model. For each factual rewrite it:

1. Finds the hidden key representation (`k`) that retrieves the fact.
2. Optimizes a target representation (`z`) for the new answer.
3. Computes an update for selected MLP layers.
4. Projects the update into a null space to reduce disruption to preserved
   knowledge.

Conceptually:

```text
W_new = W_old + P * update
```

where `P` is the null-space projection matrix.

## Data required by the original code

The original repository does not bundle the edit datasets. `globals.yml`
defines the dataset root:

```yaml
DATA_DIR: "data"
```

The original loaders expect these exact files:

| Evaluator name | Expected file | Behavior |
|---|---|---|
| `cf` | `data/counterfact.json` | Downloads from the original MEMIT data URL if absent |
| `mcf` | `data/multi_counterfact.json` | Downloads from the original MEMIT data URL if absent |
| `zsre` | `data/zsre_mend_eval.json` | Downloads from the original MEMIT data URL if absent |
| `mquake` | `data/MQuAKE-CF-3k-v2.json` | Must already exist; the loader reads it directly |

These are edit and evaluation cases, not training data for the LLM.
`--dataset_size_limit` controls how many cases are loaded. The original
quick-start uses 2,000 MultiCounterFact cases and applies 100 edits at a time.

### CounterFact and MultiCounterFact

Their records contain a rewrite such as:

```json
{
  "requested_rewrite": {
    "prompt": "The mother tongue of {} is",
    "subject": "Danielle Darrieux",
    "target_true": {"str": "French"},
    "target_new": {"str": "English"}
  }
}
```

This changes:

```text
The mother tongue of Danielle Darrieux is French
```

to:

```text
The mother tongue of Danielle Darrieux is English
```

`counterfact.json` contains individual edits. `multi_counterfact.json`
contains the batched MultiCounterFact version used by the original quick
start.

### ZsRE

The original loader expects the MEND evaluation file
`zsre_mend_eval.json`. It reads fields such as `src`, `subject`, `answers`,
`rephrase`, `loc`, and `loc_ans`, then converts them into the common
`requested_rewrite` format. It also creates paraphrase and neighborhood
prompts for evaluation.

This is not the same as a generic Hugging Face ZsRE export; the original
loader expects the specific MEND evaluation JSON schema.

### MQuAKE

The original loader expects `MQuAKE-CF-3k-v2.json`. Each record contains:

- `requested_rewrite`: the fact to edit
- `questions`: multi-hop questions
- `answer`: the answer before the edit
- `new_answer`: the answer expected after the edit

The loader uses `questions` as evaluation prompts and retains both answers for
multi-hop evaluation.

## Null-space statistics

Null-space statistics are computed from ordinary Wikipedia text, separately
from the edit cases. The original statistics loader defaults to:

```text
Dataset: wikipedia
Configuration: 20220301.en
Samples: 100,000
Precision: float32
```

The configuration and sample count come from the selected hyperparameter JSON.
The original `Llama3-8B.json`, for example, requests 100,000 samples.

For each editable MLP down-projection layer, the code:

1. Tokenizes Wikipedia text.
2. Collects the layer input activations.
3. Computes the second moment:

```text
C = E[k k^T]
```

4. Computes:

```text
C = U S U^T
```

5. Selects singular vectors below `nullspace_threshold`.
6. Builds:

```text
P = U_small U_small^T
```

AlphaEdit uses `P` in the weight-update solve so the edit is restricted to
directions with low support in the preservation statistics.

The original README also provides precomputed Llama 3 covariance files. If
those files are used, they must be decompressed under `data/stats/` with the
directory structure expected by `rome/layer_stats.py`.

## Cache and output locations

The paths are defined in `globals.yml`:

```yaml
RESULTS_DIR: "results"
DATA_DIR: "data"
STATS_DIR: "data/stats"
HPARAMS_DIR: "hparams"
```

Covariance statistics are cached under:

```text
data/stats/<model>/<dataset>_stats/
```

With `--use_cache`, the original evaluator can also reuse cached optimized
vectors and key/value representations.

Results are written under:

```text
results/<algorithm>/run_<id>/
```

The original evaluator writes per-case JSON files and can be summarized with
`experiments/summarize.py`.

## Models and baselines

The original AlphaEdit hyperparameters include configurations for GPT-2 XL,
GPT-J 6B, Llama 3 8B, and Phi-1.5 under:

```text
hparams/AlphaEdit/
```

The repository also includes:

- **FT**: direct fine-tuning of selected parameters
- **ROME**: rank-one editing for individual facts
- **MEMIT**: batched multi-layer editing
- **NSE**: sequential/null-space editing
- **MEND**: learned gradient-based editing
- **AlphaEdit**: MEMIT-style editing with null-space projection

`google/gemma-4-E2B` is not part of the original configurations. Using it
requires compatible module names, hyperparameters, and additional adaptation.

## Evaluation

`experiments/evaluate.py` supports:

- CounterFact and MultiCounterFact edit quality
- ZsRE question and paraphrase evaluation
- MQuAKE multi-hop evaluation
- generation tests
- neighborhood/locality tests
- optional GLUE-style downstream tests

Evaluation checks whether the new answer is produced, whether the edit
generalizes to paraphrases, and whether unrelated knowledge remains stable.
MQuAKE additionally checks whether the changed fact propagates through
multiple reasoning hops.

## Running the original implementation

First place the expected dataset files under `data/` and either compute or
install compatible covariance files under `data/stats/`.

Important arguments:

- `--alg_name`: editing algorithm, here `AlphaEdit`
- `--model_name`: pretrained model to edit
- `--hparams_fname`: file under `hparams/AlphaEdit/`
- `--ds_name`: `cf`, `mcf`, `zsre`, or `mquake`
- `--dataset_size_limit`: number of edit cases
- `--num_edits`: simultaneous edits per batch
- `--use_cache`: reuse cached edit vectors

The first run may take substantial GPU time and memory because it loads the
model, computes covariance statistics, performs SVD, and applies the edits.

## Running other original editors

The same evaluator supports the following algorithms:

| Algorithm | Default model | Hyperparameter file | Typical edits per batch |
|---|---|---|---:|
| `AlphaEdit` | `meta-llama/Meta-Llama-3-8B-Instruct` | `hparams/AlphaEdit/Llama3-8B.json` | 100 |
| `MEMIT` | `meta-llama/Meta-Llama-3-8B-Instruct` | `hparams/MEMIT/Llama3-8B.json` | 100 |
| `NSE` | `meta-llama/Meta-Llama-3-8B-Instruct` | `hparams/NSE/Llama3-8B.json` | 100 |
| `ROME` | `meta-llama/Meta-Llama-3-8B-Instruct` | `hparams/ROME/Llama3-8B.json` | 1 |
| `FT` | `meta-llama/Meta-Llama-3-8B-Instruct` | `hparams/FT/Llama3-8B.json` | 1 |
| `MEND` | `gpt2-xl` | `hparams/MEND/gpt2-xl.json` | 1 |

### AlphaEdit

```bash
python -m experiments.evaluate \
  --alg_name AlphaEdit \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --hparams_fname Llama3-8B.json \
  --ds_name mcf \
  --dataset_size_limit 2000 \
  --num_edits 100 \
  --downstream_eval_steps 5
```

### MEMIT

```bash
python -m experiments.evaluate \
  --alg_name MEMIT \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --hparams_fname Llama3-8B.json \
  --ds_name mcf \
  --dataset_size_limit 2000 \
  --num_edits 100 \
  --downstream_eval_steps 5
```

### NSE

```bash
python -m experiments.evaluate \
  --alg_name NSE \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --hparams_fname Llama3-8B.json \
  --ds_name mcf \
  --dataset_size_limit 2000 \
  --num_edits 100 \
  --downstream_eval_steps 5
```

### ROME

ROME is normally run as a single-edit baseline:

```bash
python -m experiments.evaluate \
  --alg_name ROME \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --hparams_fname Llama3-8B.json \
  --ds_name cf \
  --dataset_size_limit 2000 \
  --num_edits 1
```

### FT

```bash
python -m experiments.evaluate \
  --alg_name FT \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --hparams_fname Llama3-8B.json \
  --ds_name cf \
  --dataset_size_limit 2000 \
  --num_edits 1
```

### MEND

The original repository provides MEND hyperparameters for GPT-2 XL, not
Llama 3:

```bash
python -m experiments.evaluate \
  --alg_name MEND \
  --model_name gpt2-xl \
  --hparams_fname gpt2-xl.json \
  --ds_name cf \
  --dataset_size_limit 2000 \
  --num_edits 1
```

For any method, change `--ds_name` to `cf`, `mcf`, `zsre`, or `mquake` when
the corresponding original dataset file is available. The model name and
hyperparameter file must match: module paths, editable layers, and optimizer
settings are model-specific.
