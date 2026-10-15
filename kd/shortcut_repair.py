"""Training-only group localization, guarded candidate selection and controls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .baseline_artifacts import TEACHER_FILES, read_json, verify_manifest, write_manifest
from .checkpoints import (fingerprint, load_model_checkpoint, metadata, save_checkpoint,
                          tensor_state_fingerprint, write_json)
from .data import build_data
from .evaluation import collect_predictions, prediction_metrics
from .group_metrics import add_group_metrics, dataset_metadata, group_accuracy
from .models import create_model
from .neuron_surgery import (ablate_channel, collect_teacher_diagnostics, consensus_channel_ranking,
                            differential_channel_scores, localization_stages, train_repair_candidate)
from .runtime import prepare_model, resolve_device, resolve_precision
from .training_provenance import training_provenance


GROUP_REPAIR_FILES = ("teacher_protocol.json", "config.json", "student.pt", "summary.json",
                      "repair_decision.json", "repair/protocol.json", "repair/report_manifest.json")


class ColorVariants(Dataset):
    """Canonical digit plus a valid same-foreground color-swapped companion."""
    def __init__(self, original):
        self.original = original

    def __len__(self):
        return 2 * len(self.original)

    def __getitem__(self, index):
        image, label = self.original[index % len(self.original)]
        return (image[[1, 0, 2]] if index >= len(self.original) else image), label


def hard_group_targets(records, groups, target_groups, fraction):
    selected = []
    for group in target_groups:
        rows = [dict(row) for row in records if groups[row["index"]] == group]
        if not rows:
            raise ValueError("Prespecified discordant training group is absent")
        mistakes = sorted((row for row in rows if not row["correct"]), key=lambda row: (-row["nll"], row["index"]))
        correct = sorted((row for row in rows if row["correct"]), key=lambda row: (-row["nll"], row["index"]))
        chosen = mistakes + correct[:max(0, math.ceil(len(rows) * fraction) - len(mistakes))]
        for row in chosen:
            row["group"] = int(group)
            row["selection_reason"] = "misclassified" if not row["correct"] else "high_loss_correct"
        selected.extend(chosen)
    return sorted(selected, key=lambda row: row["index"])


def group_preservation(records, groups, targets, count, seed):
    rng = np.random.default_rng(seed)
    excluded = {row["index"] for row in targets}
    selected, coverage = [], {}
    for group in sorted(np.unique(groups).tolist()):
        candidates = [row["index"] for row in records if groups[row["index"]] == group and row["index"] not in excluded and row["correct"]]
        policy = "correct_non_target"
        if not candidates:
            candidates = [row["index"] for row in records if groups[row["index"]] == group and row["index"] not in excluded]
            policy = "non_target_fallback"
        if not candidates:
            candidates = [row["index"] for row in records if groups[row["index"]] == group]
            policy = "target_overlap_fallback"
        chosen = rng.permutation(candidates)[:min(count, len(candidates))].tolist()
        selected.extend(chosen)
        coverage[str(group)] = {"available": len(candidates), "selected": len(chosen), "policy": policy}
    return sorted(selected), coverage


def nuisance_companions(dataset, records, embeddings, targets, count):
    groups = dataset_metadata(dataset, "group_ids")
    labels = dataset_metadata(dataset, "targets")
    ids = dataset_metadata(dataset, "source_ids")
    result = []
    for target in targets:
        index = target["index"]
        if hasattr(dataset, "colors"):
            values, policy = [len(dataset) + index], "same_foreground_color_flip"
        else:
            pool = np.flatnonzero((labels == labels[index]) & (groups != groups[index]))
            same_foreground = pool[ids[pool] == ids[index]]
            if len(same_foreground):
                pool, policy = same_foreground, "same_foreground_nuisance_discordant"
            else:
                policy = "same_label_nuisance_discordant_feature_match_not_counterfactual_proof"
                correct = np.asarray([records[i]["correct"] for i in pool])
                if correct.any():
                    pool = pool[correct]
            if not len(pool):
                raise ValueError("No nuisance-discordant companion for a prespecified target")
            similarities = embeddings[pool] @ embeddings[index]
            order = np.lexsort((pool, -similarities))
            values = pool[order[:count]].tolist()
        distinct = len(values)
        padded = [values[i % len(values)] for i in range(count)]
        result.append({"target": target, "companions": [{"index": int(i)} for i in padded],
                       "companion_policy": policy, "requested": count, "distinct": distinct,
                       "gradient_only": False})
    return result


def _measure(model, dataset, indices, device, batch_size):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    labels, predictions, margins = [], [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device)).logits.float()
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Non-finite group screening logits")
            y = targets.to(device)
            rival = logits.clone().scatter(1, y[:, None], -torch.inf).max(1).values
            margins.extend((logits.gather(1, y[:, None]).squeeze(1) - rival).cpu().tolist())
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    groups = dataset_metadata(dataset, "group_ids")[indices]
    result = group_accuracy(np.asarray(labels), np.asarray(predictions), groups, dataset.num_groups)
    result["accuracy"] = float(np.mean(np.asarray(labels) == predictions))
    result["mean_group_margin"] = float(np.mean([np.mean(np.asarray(margins)[groups == group]) for group in np.unique(groups)]))
    return result


def group_guardrails(candidate, baseline, policy):
    if not candidate["group_metrics_available"] or not baseline["group_metrics_available"]:
        return False
    return (candidate["accuracy"] >= baseline["accuracy"] - policy["overall_accuracy_guardrail"]
            and all(after["accuracy"] >= before["accuracy"] - policy["group_accuracy_guardrail"]
                    for after, before in zip(candidate["per_group"], baseline["per_group"], strict=True)))


def candidate_eligible(candidate, baseline, policy):
    return group_guardrails(candidate, baseline, policy) and candidate["worst_group_accuracy"] > baseline["worst_group_accuracy"]


def screen_groups(model, dataset, targets, preservation, consensus, device, policy):
    target_indices = [row["index"] for row in targets]
    baseline_target = _measure(model, dataset, target_indices, device, policy["localization_batch_size"])
    baseline_preservation = _measure(model, dataset, preservation, device, policy["localization_batch_size"])
    rows = []
    eligible = [row for row in consensus["channels"] if row["eligible"]]
    for row in eligible[:policy["screen_limit"]]:
        with ablate_channel(model, row["stage"], row["channel"]):
            target = _measure(model, dataset, target_indices, device, policy["localization_batch_size"])
            preserve = _measure(model, dataset, preservation, device, policy["localization_batch_size"])
        delta = target["mean_group_margin"] - baseline_target["mean_group_margin"]
        rows.append({**row, "target_margin_delta": delta, "preservation": preserve,
                     "retained": delta > 0 and group_guardrails(preserve, baseline_preservation, policy)})
    ranking = sorted((row for row in rows if row["retained"]), key=lambda row: (-row["target_margin_delta"], row["stage"], row["channel"]))
    return {"baseline_target": baseline_target, "baseline_preservation": baseline_preservation,
            "channels": rows, "screened": len(rows), "eligible_not_screened": max(0, len(eligible) - len(rows)),
            "ranking": [{"stage": row["stage"], "channel": row["channel"]} for row in ranking]}


def _validation(model, data, device, config):
    labels, probabilities = collect_predictions(model, data.val, device, precision=config.train.precision,
                                                channels_last=config.train.channels_last)
    return add_group_metrics(prediction_metrics(labels, probabilities, data.classes), labels, probabilities, data.val)


def _freeze(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Frozen group-repair identity changed: {path}")
    if not path.exists():
        write_json(path, value)


def train_shortcut_repair(spec, condition, config):
    from .matrix_binding import STUDY_SOURCE_FILES
    from .shortcut_baselines import DISCORDANT_GROUPS
    directory = Path(config.output_dir) / config.name
    control = directory.parent / f"teacher_control_seed{config.train.seed}"
    verify_manifest(control, "completion.json", TEACHER_FILES)
    base_summary = read_json(control / "summary.json")
    identity = {"version": 1, "condition": condition, "config": config.to_dict(),
                "group_policy": spec.group_policy, "repair_options": spec.repair.__dict__,
                "base_teacher_sha256": fingerprint(control / "student.pt"),
                "base_completion_sha256": fingerprint(control / "completion.json"),
                "source_sha256": {name: fingerprint(Path(__file__).parent / name) for name in STUDY_SOURCE_FILES}}
    # JSON-normalize tuples before comparing durable identity.
    import json
    identity = json.loads(json.dumps(identity, allow_nan=False))
    if directory.exists() and any(directory.iterdir()) and not (directory / "teacher_protocol.json").exists():
        raise FileExistsError("Nonempty group-repair directory has no protocol")
    _freeze(directory / "teacher_protocol.json", identity)
    _freeze(directory / "config.json", json.loads(json.dumps(config.to_dict())))
    if (directory / "completion.json").exists():
        verify_manifest(directory, "completion.json", GROUP_REPAIR_FILES)
        verify_manifest(directory / "repair", "report_manifest.json", ("protocol.json", "localization.json"))
        return read_json(directory / "summary.json")
    device = resolve_device(config.train.device)
    data = build_data(config.data, replace(config.train, workers=0), include_test=False, diagnostic=True)
    fitting = build_data(config.data, replace(config.train, workers=0), include_test=False, diagnostic=False)
    if training_provenance(data.provenance) != training_provenance(base_summary["data"]) or data.provenance != fitting.provenance:
        raise ValueError("Group-repair fitting/diagnostic data differs from its CE anchor")
    anchor = create_model(config.student.name, config.data.num_classes)
    contract = metadata(config.student.name, data.classes, config.data.image_size, config.data.source)
    load_model_checkpoint(anchor, control / "student.pt", contract)
    prepare_model(anchor, device, config.train.channels_last).requires_grad_(False).eval()
    anchor_tensor = tensor_state_fingerprint(anchor.state_dict())
    policy, options = spec.group_policy, spec.repair
    repair_dir = directory / "repair"
    _freeze(repair_dir / "protocol.json", {"identity": identity, "training_provenance": training_provenance(data.provenance),
        "target_groups": list(DISCORDANT_GROUPS[config.data.source]), "sampler_balance": "groups, not classes",
        "continuation_seed_policy": "one seed per channel-budget slot; seed + slot, same number of trials",
        "buffer_policy": "all buffers frozen in all repair/control arms"})
    groups = dataset_metadata(data.train.dataset, "group_ids")
    localization_manifest = repair_dir / "localization_completion.json"
    if localization_manifest.exists():
        verify_manifest(repair_dir, "localization_completion.json", ("protocol.json", "localization.json"))
    else:
        start = time.perf_counter()
        records, embeddings = collect_teacher_diagnostics(anchor, data.train, device, precision="float32",
                                                          channels_last=config.train.channels_last)
        targets = hard_group_targets(records, groups, DISCORDANT_GROUPS[config.data.source], options.target_fraction)
        preservation, preserve_coverage = group_preservation(records, groups, targets, policy["preservation_per_group"], policy["seed"])
        companions = nuisance_companions(data.train.dataset, records, embeddings, targets, options.companion_count)
        localization_dataset = ColorVariants(data.train.dataset) if config.data.source == "colored_mnist" else data.train.dataset
        if condition in {"teacher_repair", "repair_attribution_only"}:
            scores = differential_channel_scores(anchor, localization_dataset, companions, device,
                batch_size=policy["localization_batch_size"], stages=localization_stages(anchor))
            scores["labels"] = groups[scores["target_indices"]]
            consensus = consensus_channel_ranking(scores, target_classes=DISCORDANT_GROUPS[config.data.source],
                repetitions=policy["bootstrap_repetitions"], seed=policy["seed"], top_fraction=policy["top_fraction"],
                stability_threshold=policy["stability_threshold"], stages=localization_stages(anchor))
            for row in consensus["channels"]:
                row["per_group_score"] = row.pop("per_class_score")
        else:
            consensus = {"channels": [], "not_applicable": True}
        screening = (screen_groups(anchor, data.train.dataset, targets, preservation, consensus, device, policy)
                     if condition == "teacher_repair" else {"ranking": [], "channels": [], "not_applicable": True})
        all_channels = [{"stage": stage, "channel": channel} for stage in localization_stages(anchor)
                        for channel in range(anchor.feature_channels[stage])]
        if condition == "repair_random_channels":
            order = np.random.default_rng(policy["seed"]).permutation(len(all_channels))
            ranking = [all_channels[index] for index in order]
        elif condition == "repair_attribution_only":
            ranking = [{"stage": row["stage"], "channel": row["channel"]} for row in consensus["channels"] if row["eligible"]]
        elif condition == "repair_ce_continuation":
            ranking = all_channels
        else:
            ranking = screening["ranking"]
        localization_seconds = time.perf_counter() - start
        write_json(repair_dir / "localization.json", {"targets": targets, "companions": companions,
            "preservation_indices": preservation, "preservation_coverage": preserve_coverage,
            "consensus": consensus, "screening": screening, "ranking": ranking,
            "localization_seconds": localization_seconds})
        write_manifest(repair_dir, "localization_completion.json", ("protocol.json", "localization.json"))
    localization = read_json(repair_dir / "localization.json")
    targets = localization["targets"]
    preservation = localization["preservation_indices"]
    ranking = localization["ranking"]
    localization_seconds = localization["localization_seconds"]
    baseline = _validation(anchor, data, device, config)
    trials, selected = [], None
    selected_key = None
    for position, budget in enumerate(options.channel_budgets):
        if condition != "repair_ce_continuation" and len(ranking) < budget:
            trials.append({"budget": budget, "status": "infeasible", "reason": "Insufficient eligible/screened channels"})
            continue
        channels = [] if condition == "repair_ce_continuation" else ranking[:budget]
        for lr in options.learning_rates:
            fit_seed = policy["seed"] + position if condition == "repair_ce_continuation" else policy["seed"]
            trial_dir = repair_dir / f"candidate-b{budget}-lr{lr:g}"
            trial_protocol = {"parent_protocol_sha256": fingerprint(repair_dir / "protocol.json"),
                              "budget": budget, "learning_rate": lr, "channels": channels,
                              "fit_seed": fit_seed, "localization_sha256": fingerprint(repair_dir / "localization.json")}
            _freeze(trial_dir / "protocol.json", trial_protocol)
            if (trial_dir / "completion.json").exists():
                verify_manifest(trial_dir, "completion.json", ("protocol.json", "candidate.pt", "result.json", "history.json"))
                outcome = read_json(trial_dir / "result.json")
            else:
                candidate = deepcopy(anchor)
                fit_start = time.perf_counter()
                # Existing fitter uses these labels only for sampling weights;
                # actual CE/KD labels still come from the two-class dataset.
                history, verification = train_repair_candidate(candidate, anchor, fitting.train.dataset,
                    groups, [row["index"] for row in targets], preservation, channels, device,
                    epochs=options.repair_epochs, samples_per_epoch=options.repair_samples_per_epoch,
                    batch_size=options.repair_batch_size, learning_rate=lr, momentum=0.9,
                    temperature=4.0, kd_weight=1.0, feature_weight=0.25, seed=fit_seed,
                    precision=config.train.precision, channels_last=config.train.channels_last,
                    full_finetune=condition == "repair_ce_continuation",
                    failure_dir=trial_dir / "failures",
                    failure_context={"condition": condition, "config": config.to_dict(),
                        "trial_protocol": trial_protocol,
                        "base_teacher_sha256": identity["base_teacher_sha256"],
                        "base_completion_sha256": identity["base_completion_sha256"]})
                quality = _validation(candidate, data, device, config)
                outcome = {"budget": budget, "learning_rate": lr, "fit_seed": fit_seed,
                    "status": "eligible" if candidate_eligible(quality, baseline, policy) else "rejected",
                    "validation": quality, "verification": verification,
                    "updates": len(history) * math.ceil(options.repair_samples_per_epoch / options.repair_batch_size),
                    "channel_budget": None if condition == "repair_ce_continuation" else budget,
                    "fitting_seconds": time.perf_counter() - fit_start, "channels": channels,
                    "parameter_scope": "all parameters; frozen buffers" if condition == "repair_ce_continuation" else "masked connected channels"}
                save_checkpoint(trial_dir / "candidate.pt", {**contract, "kind": "inference",
                    "epoch": options.repair_epochs - 1, "student": {name: value.detach().cpu() for name, value in candidate.state_dict().items()},
                    "base_teacher_sha256": identity["base_teacher_sha256"],
                    "training_provenance": training_provenance(data.provenance)})
                write_json(trial_dir / "result.json", outcome)
                write_json(trial_dir / "history.json", history)
                write_manifest(trial_dir, "completion.json", ("protocol.json", "candidate.pt", "result.json", "history.json"))
                del candidate
            trials.append(outcome)
            if outcome["status"] == "eligible":
                quality = outcome["validation"]
                key = (quality["worst_group_accuracy"], quality["accuracy"], -quality["nll"], -budget, -lr)
                if selected_key is None or key > selected_key:
                    selected, selected_key = trial_dir, key
    if tensor_state_fingerprint(anchor.state_dict()) != anchor_tensor:
        raise RuntimeError("CE anchor changed during group repair")
    decision = {"status": "selected" if selected else "no_change", "condition": condition,
                "baseline_validation": baseline, "candidates": trials,
                "selected_candidate": str(selected.relative_to(directory)) if selected else None,
                "target_groups": list(DISCORDANT_GROUPS[config.data.source]), "group_policy": policy,
                "no_change_interpretation": "exact control alias, not a separate fit" if not selected else None}
    shutil.copyfile(selected / "candidate.pt" if selected else control / "student.pt", directory / "student.pt")
    chosen_validation = read_json(selected / "result.json")["validation"] if selected else baseline
    summary = {"checkpoint": str(directory / "student.pt"), "checkpoint_sha256": fingerprint(directory / "student.pt"),
        "initial_student_sha256": base_summary["initial_student_sha256"], "data": data.provenance,
        "class_counts": data.provenance["train_per_class"], "validation": chosen_validation,
        "selection_metric": "worst_group_accuracy", "repair_condition": condition,
        "training_seconds": base_summary["training_seconds"] + localization_seconds + sum(trial.get("fitting_seconds", 0) for trial in trials),
        "repair_search_seconds": localization_seconds + sum(trial.get("fitting_seconds", 0) for trial in trials),
        "timing_scope": "CE pretraining plus this arm's completed localization/candidate search; excludes interrupted work",
        "initialization": base_summary.get("initialization", {"weights": None}),
        "parameters": base_summary["parameters"]}
    write_json(directory / "repair_decision.json", decision)
    write_json(directory / "summary.json", summary)
    artifacts = ("protocol.json", "localization.json", "localization_completion.json",
        *(path.relative_to(repair_dir).as_posix() for path in repair_dir.glob("candidate-*/*") if path.is_file()))
    write_manifest(repair_dir, "report_manifest.json", artifacts)
    write_manifest(directory, "completion.json", GROUP_REPAIR_FILES)
    return summary
