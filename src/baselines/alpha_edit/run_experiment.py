"""AlphaEdit runner for CounterFact, ZsRE, and MQuAKE edits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov
from util import nethook


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EDIT_DATA = ROOT / "data" / "mquake_2000" / "CF3k.jsonl"


def _request(case_id, rewrite: dict) -> dict:
    return {
        "case_id": case_id,
        "prompt": rewrite["prompt"],
        "subject": rewrite["subject"],
        "target_new": {
            "str": rewrite["target_new"]["str"],
            "id": rewrite["target_new"].get("id", ""),
        },
        "target_true": {
            "str": rewrite["target_true"]["str"],
            "id": rewrite["target_true"].get("id", ""),
        },
    }


def load_requests(path: Path, dataset_type: str, limit: int | None) -> list[dict]:
    requests = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if dataset_type == "counterfact":
                requests.append(_request(row["case_id"], row["requested_rewrite"]))
            elif dataset_type == "mquake":
                rewrite = row["requested_rewrite"][0]
                requests.append(
                    _request(
                        row["case_id"],
                        {
                            "prompt": rewrite["prompt"],
                            "subject": rewrite["subject"],
                            "target_new": {
                                "id": rewrite.get("target_new_id", ""),
                                "str": rewrite["target_new_str"],
                            },
                            "target_true": {
                                "id": rewrite.get("target_true_id", ""),
                                "str": rewrite["target_true_str"],
                            },
                        },
                    )
                )
            elif dataset_type == "zsre":
                answers = row.get("output") or []
                alternatives = row.get("alternatives") or []
                if not answers or not alternatives:
                    continue
                requests.append(
                    _request(
                        row["id"],
                        {
                            "prompt": "{}",
                            "subject": row["input"],
                            "target_new": {"str": alternatives[0]},
                            "target_true": {"str": answers[0]["answer"]},
                        },
                    )
                )
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            if limit is not None and len(requests) >= limit:
                break
    if not requests:
        raise ValueError(f"No editable cases found in {path}")
    return requests


def choose_layers(model, requested: str | None) -> list[int]:
    if requested:
        return [int(value) for value in requested.split(",")]
    count = getattr(model.config, "num_hidden_layers", None)
    if count is None:
        raise ValueError("Pass --layers because this model has no num_hidden_layers")
    start = max(0, count // 4)
    end = max(start + 1, count // 2)
    return list(range(start, end))


def make_hparams(model_name: str, layers: list[int], samples: int, config: str):
    return AlphaEditHyperParams(
        model_name=model_name,
        layers=layers,
        layer_selection="all",
        fact_token="subject_last",
        v_num_grad_steps=25,
        v_lr=0.1,
        v_loss_layer=-1,
        v_weight_decay=0.5,
        clamp_norm_factor=0.75,
        kl_factor=0.0625,
        mom2_adjustment=True,
        mom2_update_weight=15000,
        rewrite_module_tmp="model.layers.{}.mlp.down_proj",
        layer_module_tmp="model.layers.{}",
        mlp_module_tmp="model.layers.{}.mlp",
        attn_module_tmp="model.layers.{}.self_attn",
        ln_f_module="model.norm",
        lm_head_module="lm_head",
        mom2_dataset="wikimedia/wikipedia",
        mom2_n_samples=samples,
        mom2_dtype="float32",
        nullspace_threshold=0.02,
        L2=10.0,
    )


def build_null_space(model, tokenizer, hparams, dataset_config: str) -> torch.Tensor:
    weight = nethook.get_parameter(
        model, f"{hparams.rewrite_module_tmp.format(hparams.layers[0])}.weight"
    )
    dimension = weight.shape[1]
    projectors = torch.empty(
        (len(hparams.layers), dimension, dimension), dtype=torch.float32
    )
    for index, layer in enumerate(hparams.layers):
        covariance = get_cov(
            model,
            tokenizer,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            dataset_config=dataset_config,
        ).cpu()
        vectors, values, _ = torch.linalg.svd(covariance, full_matrices=False)
        keep = values < hparams.nullspace_threshold
        projectors[index] = (
            vectors[:, keep] @ vectors[:, keep].T
            if keep.any()
            else torch.zeros_like(covariance)
        )
        print(
            f"layer={layer} covariance={tuple(covariance.shape)} "
            f"nullity={int(keep.sum())}"
        )
    return projectors


def generate(model, tokenizer, request, max_new_tokens: int = 16) -> str:
    prompt = request["prompt"].format(request["subject"])
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--edit-data", type=Path, default=DEFAULT_EDIT_DATA)
    parser.add_argument(
        "--dataset-type",
        choices=("counterfact", "zsre", "mquake"),
        default="mquake",
    )
    parser.add_argument("--edit-limit", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--dataset-config", default="20231101.en")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--run-label", default="mquake")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "AlphaEdit")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("AlphaEdit requires CUDA because its optimization path uses GPU tensors.")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device).eval()
    layers = choose_layers(model, args.layers)
    hparams = make_hparams(args.model, layers, args.samples, args.dataset_config)
    hparams.mom2_dataset = args.dataset
    requests = load_requests(args.edit_data, args.dataset_type, args.edit_limit)
    projectors = build_null_space(model, tokenizer, hparams, args.dataset_config)

    run_dir = args.output_dir / (
        f"run_{args.run_label}_gemma4_e2b_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / run_dir.name
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(projectors, run_dir / "null_space_projectors.pt")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "edit_data": str(args.edit_data),
                "dataset_type": args.dataset_type,
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "samples": args.samples,
                "layers": layers,
                "batch_size": args.batch_size,
                "num_requests": len(requests),
            },
            indent=2,
        )
        + "\n"
    )

    before = [generate(model, tokenizer, request) for request in requests]
    cache_c = torch.zeros(
        (
            len(layers),
            nethook.get_parameter(
                model, f"{hparams.rewrite_module_tmp.format(layers[0])}.weight"
            ).shape[1],
            nethook.get_parameter(
                model, f"{hparams.rewrite_module_tmp.format(layers[0])}.weight"
            ).shape[1],
        ),
        dtype=torch.float32,
    )
    for start in range(0, len(requests), args.batch_size):
        batch = requests[start : start + args.batch_size]
        model, cache_c = apply_AlphaEdit_to_model(
            model,
            tokenizer,
            batch,
            hparams,
            cache_c=cache_c,
            P=projectors,
        )
    after = [generate(model, tokenizer, request) for request in requests]
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "requests": requests,
                "before": before,
                "after": after,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    (run_dir / "checkpoint_path.txt").write_text(str(checkpoint_dir) + "\n")


if __name__ == "__main__":
    main()
