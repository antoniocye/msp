from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"

sys.path.insert(0, str(VENDOR_DIR / "SAELens"))
sys.path.insert(0, str(VENDOR_DIR / "param-decomp"))

from sae_lens.saes.sae import TrainStepInput  # noqa: E402
from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig  # noqa: E402
from spd.configs import Config, TMSTaskConfig  # noqa: E402
from spd.models.component_model import ComponentModel  # noqa: E402
from spd.models.component_utils import calc_causal_importances  # noqa: E402
from spd.run_spd import optimize  # noqa: E402


def load_suite() -> Any:
    suite_path = ROOT / "experiments" / "src" / "sparse_concept_rare_event_suite.py"
    spec = importlib.util.spec_from_file_location("sparse_concept_rare_event_suite", suite_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {suite_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


suite = load_suite()


def _topk_codes(codes: np.ndarray, n_components: int) -> np.ndarray:
    keep = max(2, min(8, n_components // 6))
    if keep >= codes.shape[1]:
        return codes.astype(np.float32)
    sparse_codes = np.zeros_like(codes)
    idx = np.argpartition(np.abs(codes), -keep, axis=1)[:, -keep:]
    rows = np.arange(len(codes))[:, None]
    sparse_codes[rows, idx] = codes[rows, idx]
    return sparse_codes.astype(np.float32)


class SklearnSDLBasis:
    def __init__(self, n_components: int, alpha: float, seed: int, max_iter: int = 500) -> None:
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.scaler = StandardScaler()
        self.model: MiniBatchDictionaryLearning | None = None
        self.components_: np.ndarray | None = None

    def fit(self, h: np.ndarray, model: nn.Module | None = None) -> "SklearnSDLBasis":
        del model
        z = self.scaler.fit_transform(h).astype(np.float32)
        keep = max(2, min(8, self.n_components // 6))
        mdl = MiniBatchDictionaryLearning(
            n_components=self.n_components,
            alpha=max(self.alpha, 1e-4),
            max_iter=self.max_iter,
            batch_size=min(512, max(64, len(z))),
            fit_algorithm="cd",
            transform_algorithm="omp",
            transform_n_nonzero_coefs=keep,
            random_state=self.seed,
        )
        mdl.fit(z)
        comps = np.asarray(mdl.components_, dtype=np.float32)
        norms = np.linalg.norm(comps, axis=1, keepdims=True) + 1e-8
        self.components_ = comps / norms
        mdl.components_ = self.components_
        self.model = mdl
        return self

    def transform(self, h: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("basis not fit")
        z = self.scaler.transform(h).astype(np.float32)
        return np.asarray(self.model.transform(z), dtype=np.float32)

    def reconstruct_scaled(self, codes: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("basis not fit")
        return np.asarray(codes @ self.components_, dtype=np.float32)

    def component_in_h_space(self, k: int) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("basis not fit")
        return self.components_[k] * self.scaler.scale_


class SAELensBasis:
    def __init__(
        self,
        n_components: int,
        alpha: float,
        seed: int,
        steps: int = 250,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> None:
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.scaler = StandardScaler()
        self.sae: StandardTrainingSAE | None = None
        self.components_: np.ndarray | None = None

    def fit(self, h: np.ndarray, model: nn.Module | None = None) -> "SAELensBasis":
        del model
        suite.set_seed(self.seed)
        z = self.scaler.fit_transform(h).astype(np.float32)
        cfg = StandardTrainingSAEConfig(
            d_in=z.shape[1],
            d_sae=self.n_components,
            device="cpu",
            dtype="float32",
            l1_coefficient=max(self.alpha, 1e-4),
            l1_warm_up_steps=max(1, self.steps // 5),
            decoder_init_norm=0.1,
        )
        sae = StandardTrainingSAE(cfg)
        loader = DataLoader(
            TensorDataset(torch.tensor(z, dtype=torch.float32)),
            batch_size=min(self.batch_size, len(z)),
            shuffle=True,
            generator=suite.torch_generator(self.seed),
        )
        opt = torch.optim.AdamW(sae.parameters(), lr=self.lr, weight_decay=1e-4)
        data_iter = iter(loader)
        sae.train()
        for step in range(self.steps):
            try:
                (xb,) = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                (xb,) = next(data_iter)
            warm = min(1.0, (step + 1) / max(1, cfg.l1_warm_up_steps))
            step_input = TrainStepInput(
                sae_in=xb,
                coefficients={"l1": float(cfg.l1_coefficient * warm)},
                dead_neuron_mask=None,
                n_training_steps=step,
                is_logging_step=False,
            )
            out = sae.training_forward_pass(step_input)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
        sae.eval()
        sae.fold_W_dec_norm()
        self.sae = sae
        self.components_ = sae.W_dec.detach().cpu().numpy().astype(np.float32)
        return self

    def transform(self, h: np.ndarray) -> np.ndarray:
        if self.sae is None:
            raise RuntimeError("basis not fit")
        z = self.scaler.transform(h).astype(np.float32)
        with torch.no_grad():
            codes = self.sae.encode(torch.tensor(z, dtype=torch.float32)).cpu().numpy()
        return _topk_codes(np.asarray(codes, dtype=np.float32), self.n_components)

    def reconstruct_scaled(self, codes: np.ndarray) -> np.ndarray:
        if self.sae is None:
            raise RuntimeError("basis not fit")
        with torch.no_grad():
            recon = self.sae.decode(torch.tensor(codes, dtype=torch.float32)).cpu().numpy()
        return np.asarray(recon, dtype=np.float32)

    def component_in_h_space(self, k: int) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("basis not fit")
        return self.components_[k] * self.scaler.scale_


class HeadOnly(nn.Module):
    def __init__(self, head: nn.Linear) -> None:
        super().__init__()
        self.head = nn.Linear(head.in_features, head.out_features)
        with torch.no_grad():
            self.head.weight.copy_(head.weight.detach().cpu())
            self.head.bias.copy_(head.bias.detach().cpu())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h.float())


class SPDBasis:
    def __init__(
        self,
        n_components: int,
        alpha: float,
        seed: int,
        steps: int = 160,
        batch_size: int = 256,
        lr: float = 3e-3,
        out_dir: Path | None = None,
    ) -> None:
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.out_dir = out_dir
        self.component_model: ComponentModel | None = None
        self.components_: np.ndarray | None = None

    def fit(self, h: np.ndarray, model: nn.Module | None = None) -> "SPDBasis":
        if model is None or not hasattr(model, "head"):
            raise ValueError("SPD basis requires a model with a linear head")
        suite.set_seed(self.seed)
        target = HeadOnly(model.head).cpu().eval()
        x = torch.tensor(h[:, None, :], dtype=torch.float32)
        loader = DataLoader(
            x,
            batch_size=min(self.batch_size, len(x)),
            shuffle=True,
            generator=suite.torch_generator(self.seed),
        )
        config = Config(
            wandb_project=None,
            wandb_run_name=None,
            wandb_run_name_prefix="cifar10_head_",
            seed=self.seed,
            C=self.n_components,
            n_mask_samples=2,
            n_ci_mlp_neurons=0,
            target_module_patterns=["head"],
            faithfulness_coeff=1.0,
            recon_coeff=1.0,
            stochastic_recon_coeff=0.5,
            recon_layerwise_coeff=None,
            stochastic_recon_layerwise_coeff=None,
            importance_minimality_coeff=max(self.alpha, 1e-4),
            schatten_coeff=None,
            out_recon_coeff=0.2,
            embedding_recon_coeff=None,
            is_embed_unembed_recon=False,
            pnorm=0.8,
            output_loss_type="mse",
            lr=self.lr,
            steps=self.steps,
            batch_size=min(self.batch_size, len(x)),
            lr_schedule="constant",
            lr_exponential_halflife=None,
            lr_warmup_pct=0.05,
            n_eval_steps=1,
            image_freq=None,
            image_on_first_step=False,
            print_freq=max(1, self.steps // 4),
            save_freq=None,
            log_ce_losses=False,
            pretrained_model_class="external_tool_auditing.HeadOnly",
            pretrained_model_path=None,
            pretrained_model_name_hf=None,
            pretrained_model_output_attr=None,
            tokenizer_name=None,
            task_config=TMSTaskConfig(feature_probability=0.1),
        )
        out_dir = self.out_dir
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "spd_config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2), encoding="utf-8")
        optimize(
            target_model=target,
            config=config,
            device="cpu",
            train_loader=loader,
            eval_loader=loader,
            n_eval_steps=1,
            out_dir=out_dir,
            plot_results_fn=None,
        )
        component_model = ComponentModel(
            base_model=HeadOnly(model.head).cpu().eval(),
            target_module_patterns=["head"],
            C=self.n_components,
            n_ci_mlp_neurons=0,
            pretrained_model_output_attr=None,
        )
        if out_dir is not None:
            state_path = out_dir / f"model_{self.steps}.pth"
            component_model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
        self.component_model = component_model
        component = self.component_model.components["head"]
        self.components_ = component.A.detach().cpu().T.numpy().astype(np.float32)
        norms = np.linalg.norm(self.components_, axis=1, keepdims=True) + 1e-8
        self.components_ = self.components_ / norms
        return self

    def transform(self, h: np.ndarray) -> np.ndarray:
        if self.component_model is None:
            raise RuntimeError("basis not fit")
        x = torch.tensor(h[:, None, :], dtype=torch.float32)
        component = self.component_model.components["head"]
        gate = self.component_model.gates["head"]
        with torch.no_grad():
            ci, _ = calc_causal_importances(
                pre_weight_acts={"head": x},
                As={"head": component.A},
                gates={"head": gate},
                detach_inputs=True,
            )
        return np.asarray(ci["head"].squeeze(1).cpu().numpy(), dtype=np.float32)

    def reconstruct_scaled(self, codes: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("basis not fit")
        return np.asarray(codes @ self.components_, dtype=np.float32)

    def component_in_h_space(self, k: int) -> np.ndarray:
        if self.component_model is None:
            raise RuntimeError("basis not fit")
        return self.component_model.components["head"].A[:, k].detach().cpu().numpy()


def build_basis(kind: str, args: argparse.Namespace, seed: int, out_dir: Path) -> Any:
    if kind == "local_sdl":
        return suite.SparseDictionary(args.dictionary_components, args.dictionary_alpha, seed)
    if kind == "sklearn_sdl":
        return SklearnSDLBasis(args.dictionary_components, args.sklearn_alpha, seed, args.sklearn_max_iter)
    if kind == "saelens_sae":
        return SAELensBasis(args.dictionary_components, args.sae_l1, seed, args.sae_steps, args.sae_batch_size, args.sae_lr)
    if kind == "spd":
        return SPDBasis(args.dictionary_components, args.spd_importance_coeff, seed, args.spd_steps, args.spd_batch_size, args.spd_lr, out_dir)
    raise ValueError(kind)


def attach_basis(collected: Any, basis: Any) -> Any:
    collected.basis_codes = basis.transform(collected.internal)
    return collected


def clone_collected(collected: Any) -> Any:
    return replace(collected, basis_codes=None)


def fit_internal_bank(
    build: Any,
    concept_count: int,
    n_basis: int,
    n_classes: int,
    basis_hi: np.ndarray,
    basis_lo: np.ndarray,
    seed: int,
    basis: Any,
) -> Any:
    bank = suite.fit_atom_bank(
        "sparse_internal",
        build,
        concept_count,
        n_basis,
        n_classes,
        basis_hi,
        basis_lo,
        seed,
        basis,
    )
    bank.method = "internal"
    return bank


def basis_diagnostics(h: np.ndarray, basis: Any, seed: int) -> dict[str, float]:
    rng = suite.rng_for("external_basis_diag", seed)
    idx = rng.choice(len(h), size=min(3000, len(h)), replace=False)
    h_sub = h[idx]
    codes = basis.transform(h_sub)
    eps = 1e-4
    out = {
        "basis_code_density": float((np.abs(codes) > eps).mean()),
        "basis_active_components": float((np.abs(codes) > eps).sum(axis=1).mean()),
        "basis_reconstruction_r2": math.nan,
    }
    try:
        z = basis.scaler.transform(h_sub)
        recon = basis.reconstruct_scaled(codes)
        ss_res = float(np.square(z - recon).sum())
        ss_tot = float(np.square(z - z.mean(axis=0)).sum())
        out["basis_reconstruction_r2"] = 1.0 - ss_res / max(ss_tot, 1e-12)
    except Exception:
        pass
    return out


def literal_to_text(lit: Any, source: Any, concept_spec: Any) -> str:
    class_names = source.class_names
    values = [class_names[v] if 0 <= int(v) < len(class_names) else str(v) for v in lit.values]
    if lit.kind == "label_in":
        return "label in {" + ", ".join(values) + "}"
    if lit.kind == "label_not_in":
        return "label not in {" + ", ".join(values) + "}"
    if lit.kind == "pred_in":
        return "prediction in {" + ", ".join(values) + "}"
    if lit.kind == "pred_not_in":
        return "prediction not in {" + ", ".join(values) + "}"
    if lit.kind == "error":
        return "model error"
    if lit.kind == "confidence_gt":
        return f"confidence > {lit.threshold:.2f}"
    if lit.kind == "confidence_lt":
        return f"confidence < {lit.threshold:.2f}"
    if lit.kind == "concept":
        if 0 <= int(lit.concept) < len(concept_spec.names):
            return f"image concept: {concept_spec.names[int(lit.concept)]}"
        return f"image concept {lit.concept}"
    if lit.kind == "basis_high":
        return f"basis {lit.concept} >= {lit.threshold:.6f}"
    if lit.kind == "basis_low":
        return f"basis {lit.concept} <= {lit.threshold:.6f}"
    return str(lit)


def query_to_text(query: Any, source: Any, concept_spec: Any) -> str:
    clauses = []
    for clause in query.clauses:
        clauses.append(" AND ".join(literal_to_text(lit, source, concept_spec) for lit in clause))
    return " OR ".join(f"({clause})" for clause in clauses)


def query_uses_basis(query: Any) -> bool:
    return any(lit.kind in {"basis_high", "basis_low"} for clause in query.clauses for lit in clause)


def query_row(query: Any, source: Any, concept_spec: Any, seed: int, dictionary_seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "dictionary_seed": dictionary_seed,
        "query": query.query_id,
        "family": query.family,
        "select_rate": query.select_rate,
        "uses_basis": query_uses_basis(query),
        "uses_concept": query.uses_concept(),
        "signature": query.signature(),
        "description": query_to_text(query, source, concept_spec),
    }


def select_independent_queries(args: argparse.Namespace, source: Any, concept_spec: Any, select: Any, seed: int, dictionary_seed: int) -> list[Any]:
    n_classes = len(source.class_names)
    dummy_hi = np.zeros(max(1, args.dictionary_components), dtype=np.float32)
    dummy_lo = np.zeros(max(1, args.dictionary_components), dtype=np.float32)
    query_mode = "external"
    print(
        f"[{source.name}/m{seed}/d{dictionary_seed}] selecting {args.query_count} independent external queries",
        flush=True,
    )
    queries = suite.select_queries(
        select,
        len(concept_spec.names),
        dummy_hi,
        dummy_lo,
        args.query_count,
        suite.rng_for(source.name, seed, dictionary_seed, "independent_external_query_gen"),
        args.rate_min,
        args.rate_max,
        args.max_candidates,
        n_classes,
        query_mode,
    )
    if len(queries) < min(args.query_count, max(2, args.query_count // 3)):
        raise RuntimeError(f"only selected {len(queries)} independent queries; relax rarity band or increase candidates")
    bad = [q.query_id for q in queries if query_uses_basis(q)]
    if bad:
        raise RuntimeError(f"independent query selection produced basis queries: {bad}")
    print(f"[{source.name}/m{seed}/d{dictionary_seed}] selected {len(queries)} independent queries", flush=True)
    return queries


def run_one_basis(
    args: argparse.Namespace,
    basis_kind: str,
    model: nn.Module,
    source: Any,
    concept_spec: Any,
    streams: dict[str, Any],
    seed: int,
    dictionary_seed: int,
    train_info: dict[str, float],
    fixed_queries: list[Any] | None,
) -> dict[str, list[dict[str, object]]]:
    dataset_key = source.name
    run_key = f"{dataset_key}/{basis_kind}/m{seed}/d{dictionary_seed}"
    print(f"[{run_key}] fitting basis", flush=True)
    basis_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(basis_kind))
    basis_seed = seed + dictionary_seed * 100000 + basis_offset
    basis_out_dir = RESULTS_DIR / "spd_checkpoints" / f"m{seed}_d{dictionary_seed}" if basis_kind == "spd" else RESULTS_DIR / "basis_artifacts" / basis_kind
    basis = build_basis(basis_kind, args, basis_seed, basis_out_dir).fit(streams["build"].internal, model=model)

    build = attach_basis(clone_collected(streams["build"]), basis)
    basis_hi = np.quantile(build.basis_codes, 0.75, axis=0)
    basis_lo = np.quantile(build.basis_codes, 0.25, axis=0)
    n_classes = len(source.class_names)
    banks = {
        "internal": fit_internal_bank(build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 100, basis),
        "output_comp": suite.fit_atom_bank("output_comp", build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 200, dictionary=None),
        "input_concept_comp": suite.fit_atom_bank("input_concept_comp", build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 300, dictionary=None),
        "embedding_comp": suite.fit_atom_bank("embedding_comp", build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 400, dictionary=None),
        "pca_comp": suite.fit_atom_bank("pca_comp", build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 500, dictionary=None),
        "random_comp": suite.fit_atom_bank("random_comp", build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 600 + dictionary_seed * 1000, dictionary=None),
    }
    label_priors = np.bincount(build.labels, minlength=n_classes).astype(float) / len(build.labels)
    concept_priors = build.concepts.mean(axis=0)
    basis_high_priors = (build.basis_codes >= basis_hi).mean(axis=0)
    basis_low_priors = (build.basis_codes <= basis_lo).mean(axis=0)

    if fixed_queries is None:
        select = attach_basis(clone_collected(streams["select"]), basis)
        queries = suite.select_queries(
            select,
            len(concept_spec.names),
            basis_hi,
            basis_lo,
            args.query_count,
            suite.rng_for(dataset_key, seed, dictionary_seed, basis_kind, "query_gen"),
            args.rate_min,
            args.rate_max,
            args.max_candidates,
            n_classes,
            args.query_mode,
        )
        if len(queries) < min(args.query_count, max(2, args.query_count // 3)):
            raise RuntimeError(f"[{run_key}] only selected {len(queries)} queries; relax rarity band or increase candidates")
        print(f"[{run_key}] selected {len(queries)} representation-native queries", flush=True)
    else:
        queries = fixed_queries
        print(f"[{run_key}] using {len(queries)} fixed independent queries", flush=True)

    per_query_scorers = {q.query_id: suite.fit_query_risk_scorer(build, q, seed + 10_000 + q.query_id) for q in queries}
    ref = attach_basis(clone_collected(streams["ref"]), basis)
    risk_eval = attach_basis(clone_collected(streams["risk_eval"]), basis)
    risk_atoms = {m: b.atoms(risk_eval) for m, b in banks.items()}
    intervention_pool = attach_basis(clone_collected(streams["intervention"]), basis)

    dict_rows = [
        {
            "basis_kind": basis_kind,
            "seed": seed,
            "dictionary_seed": dictionary_seed,
            **train_info,
            **basis_diagnostics(build.internal, basis, basis_seed),
        }
    ]
    interventions = suite.intervention_analysis(model, basis, intervention_pool, seed, source.class_names)
    for row in interventions:
        row.update({"basis_kind": basis_kind, "seed": seed, "dictionary_seed": dictionary_seed})

    costs: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    refs: dict[int, tuple[float, float]] = {}
    for query in queries:
        ref_event = suite.query_mask(ref, query)
        ref_rate = float(ref_event.mean())
        ref_ci = 1.96 * math.sqrt(max(ref_rate * (1.0 - ref_rate), 1e-12) / len(ref_event))
        refs[query.query_id] = (ref_rate, ref_ci)
        risk_event = suite.query_mask(risk_eval, query)
        for method in banks:
            score = suite.compile_score_from_atoms(risk_atoms[method], query, len(risk_event))
            risk_rows.append(
                {
                    "basis_kind": basis_kind,
                    "seed": seed,
                    "dictionary_seed": dictionary_seed,
                    "query": query.query_id,
                    "family": query.family,
                    "model_query_id": f"{run_key}/{query.query_id}",
                    "method": method,
                    "query_signature": query.signature(),
                    "query_description": query_to_text(query, source, concept_spec),
                    "reference_rate": ref_rate,
                    "select_rate": query.select_rate,
                    "uses_concept": query.uses_concept(),
                    **suite.ranking_metrics(score, risk_event),
                }
            )
        out_score = suite.output_active_score(risk_eval, query, concept_priors, basis_high_priors, basis_low_priors, label_priors)
        risk_rows.append(
            {
                "basis_kind": basis_kind,
                "seed": seed,
                "dictionary_seed": dictionary_seed,
                "query": query.query_id,
                "family": query.family,
                "model_query_id": f"{run_key}/{query.query_id}",
                "method": "output_active",
                "query_signature": query.signature(),
                "query_description": query_to_text(query, source, concept_spec),
                "reference_rate": ref_rate,
                "select_rate": query.select_rate,
                "uses_concept": query.uses_concept(),
                **suite.ranking_metrics(out_score, risk_event),
            }
        )
        per_query_score = per_query_scorers[query.query_id].score(risk_eval)
        risk_rows.append(
            {
                "basis_kind": basis_kind,
                "seed": seed,
                "dictionary_seed": dictionary_seed,
                "query": query.query_id,
                "family": query.family,
                "model_query_id": f"{run_key}/{query.query_id}",
                "method": "per_query_rf",
                "query_signature": query.signature(),
                "query_description": query_to_text(query, source, concept_spec),
                "reference_rate": ref_rate,
                "select_rate": query.select_rate,
                "uses_concept": query.uses_concept(),
                **suite.ranking_metrics(per_query_score, risk_event),
            }
        )
        costs.append(
            {
                "basis_kind": basis_kind,
                "seed": seed,
                "dictionary_seed": dictionary_seed,
                "query": query.query_id,
                "family": query.family,
                "model_query_id": f"{run_key}/{query.query_id}",
                "query_signature": query.signature(),
                "query_description": query_to_text(query, source, concept_spec),
                "select_rate": query.select_rate,
                "reference_rate": ref_rate,
                "reference_ci_half_width": ref_ci,
                "uses_concept": query.uses_concept(),
                "build_n": args.build_n,
            }
        )

    methods = ["mc", "random_stratified", "output_active", "ase_output", "embedding_ase", "per_query_rf", *banks.keys()]
    for rep in range(args.reps):
        eval_pool = attach_basis(clone_collected(streams[f"eval_pool_{rep}"]), basis)
        mc_pool = attach_basis(clone_collected(streams[f"mc_pool_{rep}"]), basis)
        eval_atoms = {m: b.atoms(eval_pool) for m, b in banks.items()}
        output_feat_eval = suite.output_features(eval_pool)
        embedding_ase_feat_eval = suite.supervised_query_features(eval_pool)
        output_active_scores = {q.query_id: suite.output_active_score(eval_pool, q, concept_priors, basis_high_priors, basis_low_priors, label_priors) for q in queries}
        per_query_scores = {q.query_id: per_query_scorers[q.query_id].score(eval_pool) for q in queries}
        score_cache: dict[tuple[str, int], np.ndarray] = {}
        for q in queries:
            for m in banks:
                score_cache[(m, q.query_id)] = suite.compile_score_from_atoms(eval_atoms[m], q, len(eval_pool.labels))
        for query in queries:
            eval_event = suite.query_mask(eval_pool, query)
            mc_event = suite.query_mask(mc_pool, query)
            ref_rate, ref_ci = refs[query.query_id]
            for budget in args.budgets:
                for method in methods:
                    est_rng = suite.rng_for(dataset_key, seed, dictionary_seed, basis_kind, "estimate", method, query.query_id, rep, budget)
                    if method == "mc":
                        estimate = suite.mc_estimate(mc_event, budget, est_rng)
                        build_label_cost = 0
                        forward_cost = budget
                    elif method == "random_stratified":
                        scores = suite.rng_for(dataset_key, seed, dictionary_seed, basis_kind, "rand_score", query.query_id, rep, budget).random(len(eval_event))
                        estimate = suite.stratified_estimate(scores, eval_event, budget, est_rng)
                        build_label_cost = 0
                        forward_cost = args.pool_n
                    elif method == "output_active":
                        estimate = suite.stratified_estimate(output_active_scores[query.query_id], eval_event, budget, est_rng)
                        build_label_cost = 0
                        forward_cost = args.pool_n
                    elif method == "ase_output":
                        estimate = suite.ase_output_estimate(output_feat_eval, eval_event, budget, est_rng)
                        build_label_cost = 0
                        forward_cost = args.pool_n
                    elif method == "embedding_ase":
                        estimate = suite.ase_output_estimate(embedding_ase_feat_eval, eval_event, budget, est_rng)
                        build_label_cost = 0
                        forward_cost = args.pool_n
                    elif method == "per_query_rf":
                        estimate = suite.stratified_estimate(per_query_scores[query.query_id], eval_event, budget, est_rng)
                        build_label_cost = args.build_n
                        forward_cost = args.build_n + args.pool_n
                    else:
                        estimate = suite.stratified_estimate(score_cache[(method, query.query_id)], eval_event, budget, est_rng)
                        build_label_cost = args.build_n
                        forward_cost = args.build_n + args.pool_n
                    runs.append(
                        {
                            "basis_kind": basis_kind,
                            "seed": seed,
                            "dictionary_seed": dictionary_seed,
                            "query": query.query_id,
                            "family": query.family,
                            "model_query_id": f"{run_key}/{query.query_id}",
                            "query_signature": query.signature(),
                            "query_description": query_to_text(query, source, concept_spec),
                            "method": method,
                            "budget": budget,
                            "rep": rep,
                            "estimate": estimate,
                            "reference_rate": ref_rate,
                            "reference_ci_half_width": ref_ci,
                            "select_rate": query.select_rate,
                            "uses_concept": query.uses_concept(),
                            "abs_error": abs(estimate - ref_rate),
                            "squared_error": (estimate - ref_rate) ** 2,
                            "build_label_cost": build_label_cost,
                            "query_label_budget": budget,
                            "first_query_label_cost": build_label_cost + budget,
                            "forward_cost": forward_cost,
                            **train_info,
                        }
                    )
    return {"runs": runs, "risk": risk_rows, "costs": costs, "basis": dict_rows, "interventions": interventions}


def summarize_by_basis(run_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        run_df.groupby(["basis_kind", "method", "budget"], as_index=False)
        .agg(
            cells=("squared_error", "count"),
            model_queries=("model_query_id", "nunique"),
            mean_reference_rate=("reference_rate", "mean"),
            rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
            mae=("abs_error", "mean"),
        )
        .sort_values(["basis_kind", "budget", "rmse"])
    )
    mc = out[out["method"] == "mc"][["basis_kind", "budget", "rmse"]].rename(columns={"rmse": "mc_rmse"})
    out = out.merge(mc, on=["basis_kind", "budget"], how="left")
    out["rmse_ratio_vs_mc"] = out["rmse"] / out["mc_rmse"]
    out["effective_mc_multiplier"] = 1.0 / np.square(out["rmse_ratio_vs_mc"])
    return out.sort_values(["basis_kind", "budget", "rmse_ratio_vs_mc"])


def bootstrap_ci_by_basis(run_df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rng = suite.rng_for("bootstrap", "external_tool_audit")
    rows = []
    cell = (
        run_df.groupby(["basis_kind", "budget", "model_query_id", "method"], as_index=False)
        .agg(squared_error=("squared_error", "mean"))
    )
    for (basis_kind, budget), group in cell.groupby(["basis_kind", "budget"]):
        ids = np.array(sorted(group["model_query_id"].unique()))
        by_method = {m: sub.set_index("model_query_id")["squared_error"].reindex(ids).to_numpy() for m, sub in group.groupby("method")}
        mc = by_method["mc"]
        for method, errors in by_method.items():
            vals = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                idx = rng.integers(0, len(ids), size=len(ids))
                denom = math.sqrt(float(np.mean(mc[idx])))
                numer = math.sqrt(float(np.mean(errors[idx])))
                vals[i] = numer / denom if denom > 0 else np.nan
            rmse = math.sqrt(float(np.mean(errors)))
            mc_rmse = math.sqrt(float(np.mean(mc)))
            rows.append(
                {
                    "basis_kind": basis_kind,
                    "budget": budget,
                    "method": method,
                    "model_queries": len(ids),
                    "rmse": rmse,
                    "mc_rmse": mc_rmse,
                    "rmse_ratio_vs_mc": rmse / mc_rmse,
                    "rmse_ratio_ci_low": float(np.nanquantile(vals, 0.025)),
                    "rmse_ratio_ci_high": float(np.nanquantile(vals, 0.975)),
                }
            )
    return pd.DataFrame(rows).sort_values(["basis_kind", "budget", "rmse_ratio_vs_mc"])


def risk_summary_by_basis(risk_df: pd.DataFrame) -> pd.DataFrame:
    return (
        risk_df.groupby(["basis_kind", "method"], as_index=False)
        .agg(
            cells=("auroc", "count"),
            mean_auroc=("auroc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_top_decile_lift=("top_decile_lift", "mean"),
        )
        .sort_values(["basis_kind", "mean_average_precision"], ascending=[True, False])
    )


def write_report(
    args: argparse.Namespace,
    ci: pd.DataFrame,
    summary: pd.DataFrame,
    risk: pd.DataFrame,
    basis_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    query_df: pd.DataFrame,
    elapsed: float,
) -> None:
    query_dist = (
        cost_df.groupby(["basis_kind", "family"], as_index=False)
        .agg(queries=("model_query_id", "nunique"), mean_reference_rate=("reference_rate", "mean"), mean_select_rate=("select_rate", "mean"))
        .sort_values(["basis_kind", "family"])
    )
    internal = ci[ci["method"] == "internal"].copy()
    if query_df.empty:
        query_table = cost_df[["query", "family", "select_rate", "query_signature", "query_description"]].drop_duplicates().sort_values("query")
    else:
        query_table = query_df[["query", "family", "select_rate", "uses_basis", "uses_concept", "description"]].drop_duplicates().sort_values("query")
    setup_text = (
        "This run compares external reusable bases on CIFAR-10. The rare-event queries are selected once, before fitting any SAE/SDL/SPD basis, using only labels, predictions, confidence, and pixel/image concepts."
        if args.query_policy == "independent_external"
        else "This run compares external reusable bases on CIFAR-10. Each `basis_kind` gets its own basis-consistent rare-query set."
    )
    lines = [
        "# CIFAR-10 External Tool Auditing Run",
        "",
        f"Elapsed seconds: `{elapsed:.1f}`",
        "",
        setup_text,
        "",
        f"Query policy: `{args.query_policy}`",
        "",
        "## Internal Method RMSE Ratios",
        "",
        suite.markdown_table(internal[["basis_kind", "budget", "model_queries", "rmse_ratio_vs_mc", "rmse_ratio_ci_low", "rmse_ratio_ci_high"]]),
        "",
        "## Full Summary",
        "",
        suite.markdown_table(summary[["basis_kind", "budget", "method", "model_queries", "rmse_ratio_vs_mc", "effective_mc_multiplier"]], max_rows=120),
        "",
        "## Query Distribution",
        "",
        suite.markdown_table(query_dist, max_rows=120),
        "",
        "## Frozen Rare Events",
        "",
        suite.markdown_table(query_table, max_rows=120),
        "",
        "## Basis Diagnostics",
        "",
        suite.markdown_table(basis_df, max_rows=80),
        "",
        "## Risk Ranking",
        "",
        suite.markdown_table(risk, max_rows=120),
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(vars(args), indent=2),
        "```",
    ]
    (RESULTS_DIR / "cifar10_external_tool_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_figure(ci: pd.DataFrame) -> None:
    internal = ci[ci["method"] == "internal"].copy()
    if internal.empty:
        return
    budgets = sorted(internal["budget"].unique())
    basis_kinds = list(internal["basis_kind"].drop_duplicates())
    x = np.arange(len(basis_kinds))
    width = 0.8 / max(1, len(budgets))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, budget in enumerate(budgets):
        sub = internal[internal["budget"] == budget].set_index("basis_kind").reindex(basis_kinds)
        vals = sub["rmse_ratio_vs_mc"].to_numpy(dtype=float)
        lo = vals - sub["rmse_ratio_ci_low"].to_numpy(dtype=float)
        hi = sub["rmse_ratio_ci_high"].to_numpy(dtype=float) - vals
        ax.bar(x + (i - (len(budgets) - 1) / 2) * width, vals, width=width, label=f"budget {budget}", yerr=np.vstack([lo, hi]), capsize=3)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xticks(x, basis_kinds, rotation=15, ha="right")
    ax.set_ylabel("Internal method RMSE ratio vs MC")
    ax.set_title("CIFAR-10 external basis audit comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cifar10_external_tool_internal_ratios.png", dpi=220)
    plt.close(fig)


def collect_streams(args: argparse.Namespace, source: Any, model: nn.Module, concept_spec: Any, seed: int, dictionary_seed: int) -> dict[str, Any]:
    dataset_key = source.name
    def collect_named(name: str, n: int, *rng_parts: object) -> Any:
        print(f"[{dataset_key}/m{seed}/d{dictionary_seed}] collecting {name} n={n}", flush=True)
        out = suite.collect(source, model, concept_spec, n, suite.rng_for(dataset_key, seed, dictionary_seed, *rng_parts))
        print(f"[{dataset_key}/m{seed}/d{dictionary_seed}] collected {name}", flush=True)
        return out

    streams = {
        "build": collect_named("build", args.build_n, "build"),
        "select": collect_named("select", args.select_n, "select"),
        "ref": collect_named("reference", args.ref_n, "reference"),
        "risk_eval": collect_named("risk_eval", args.risk_eval_n, "risk_eval"),
        "intervention": collect_named("intervention", args.intervention_n, "intervention"),
    }
    for rep in range(args.reps):
        streams[f"eval_pool_{rep}"] = collect_named(f"eval_pool_{rep}", args.pool_n, "eval_pool", rep)
        streams[f"mc_pool_{rep}"] = collect_named(f"mc_pool_{rep}", max(args.budgets), "mc_pool", rep)
    return streams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--basis-kinds", nargs="+", choices=["local_sdl", "sklearn_sdl", "saelens_sae", "spd"], default=["sklearn_sdl", "saelens_sae", "spd"])
    p.add_argument("--query-policy", choices=["independent_external", "basis_specific"], default="independent_external")
    p.add_argument("--seeds", nargs="+", type=int, default=[20260521])
    p.add_argument("--dictionary-seeds", nargs="+", type=int, default=[0])
    p.add_argument("--train-n", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--concept-calib-n", type=int, default=1500)
    p.add_argument("--build-n", type=int, default=1800)
    p.add_argument("--select-n", type=int, default=2600)
    p.add_argument("--ref-n", type=int, default=3500)
    p.add_argument("--risk-eval-n", type=int, default=2000)
    p.add_argument("--intervention-n", type=int, default=1200)
    p.add_argument("--pool-n", type=int, default=1536)
    p.add_argument("--query-count", type=int, default=6)
    p.add_argument("--max-candidates", type=int, default=5000)
    p.add_argument("--rate-min", type=float, default=0.001)
    p.add_argument("--rate-max", type=float, default=0.04)
    p.add_argument("--query-mode", choices=["mixed", "external", "basis"], default="external")
    p.add_argument("--dictionary-components", type=int, default=24)
    p.add_argument("--dictionary-alpha", type=float, default=0.35)
    p.add_argument("--budgets", nargs="+", type=int, default=[256, 512])
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--torch-inter-op-threads", type=int, default=1)
    p.add_argument("--sklearn-alpha", type=float, default=0.35)
    p.add_argument("--sklearn-max-iter", type=int, default=350)
    p.add_argument("--sae-l1", type=float, default=0.03)
    p.add_argument("--sae-steps", type=int, default=180)
    p.add_argument("--sae-batch-size", type=int, default=256)
    p.add_argument("--sae-lr", type=float, default=1e-3)
    p.add_argument("--spd-importance-coeff", type=float, default=0.03)
    p.add_argument("--spd-steps", type=int, default=120)
    p.add_argument("--spd-batch-size", type=int, default=256)
    p.add_argument("--spd-lr", type=float, default=3e-3)
    return p.parse_args()


def run() -> None:
    global RESULTS_DIR
    args = parse_args()
    RESULTS_DIR = Path(args.results_dir)
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.set_num_interop_threads(max(1, int(args.torch_inter_op_threads)))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_runs: list[dict[str, object]] = []
    all_risk: list[dict[str, object]] = []
    all_costs: list[dict[str, object]] = []
    all_basis: list[dict[str, object]] = []
    all_interventions: list[dict[str, object]] = []
    all_queries: list[dict[str, object]] = []

    for seed in args.seeds:
        source = suite.build_source("cifar10")
        print(f"[cifar10/m{seed}] training frozen public ResNet-18 head", flush=True)
        model, train_info = suite.train_model(source, seed, args.train_n, args.epochs, 1, "resnet18", True, True)
        concept_calib_images, _ = source.sample_eval_np(args.concept_calib_n, suite.rng_for("cifar10", seed, "concept_calib"))
        concept_spec = suite.ConceptSpec.fit(concept_calib_images)
        for dictionary_seed in args.dictionary_seeds:
            print(f"[cifar10/m{seed}/d{dictionary_seed}] collecting shared streams", flush=True)
            streams = collect_streams(args, source, model, concept_spec, seed, dictionary_seed)
            fixed_queries = None
            if args.query_policy == "independent_external":
                fixed_queries = select_independent_queries(args, source, concept_spec, streams["select"], seed, dictionary_seed)
                all_queries.extend(query_row(q, source, concept_spec, seed, dictionary_seed) for q in fixed_queries)
            for basis_kind in args.basis_kinds:
                result = run_one_basis(args, basis_kind, model, source, concept_spec, streams, seed, dictionary_seed, train_info, fixed_queries)
                all_runs.extend(result["runs"])
                all_risk.extend(result["risk"])
                all_costs.extend(result["costs"])
                all_basis.extend(result["basis"])
                all_interventions.extend(result["interventions"])

    run_df = pd.DataFrame(all_runs)
    risk_df = pd.DataFrame(all_risk)
    cost_df = pd.DataFrame(all_costs)
    basis_df = pd.DataFrame(all_basis)
    inter_df = pd.DataFrame(all_interventions)
    query_df = pd.DataFrame(all_queries)
    summary = summarize_by_basis(run_df)
    ci = bootstrap_ci_by_basis(run_df)
    risk = risk_summary_by_basis(risk_df)

    run_df.to_csv(RESULTS_DIR / "cifar10_external_tool_runs.csv", index=False)
    risk_df.to_csv(RESULTS_DIR / "cifar10_external_tool_risk.csv", index=False)
    cost_df.to_csv(RESULTS_DIR / "cifar10_external_tool_costs.csv", index=False)
    basis_df.to_csv(RESULTS_DIR / "cifar10_external_tool_basis.csv", index=False)
    inter_df.to_csv(RESULTS_DIR / "cifar10_external_tool_interventions.csv", index=False)
    query_df.to_csv(RESULTS_DIR / "cifar10_external_tool_queries.csv", index=False)
    summary.to_csv(RESULTS_DIR / "cifar10_external_tool_summary.csv", index=False)
    ci.to_csv(RESULTS_DIR / "cifar10_external_tool_ci.csv", index=False)
    risk.to_csv(RESULTS_DIR / "cifar10_external_tool_risk_summary.csv", index=False)
    write_figure(ci)
    elapsed = time.perf_counter() - started
    write_report(args, ci, summary, risk, basis_df, cost_df, query_df, elapsed)
    (RESULTS_DIR / "cifar10_external_tool_results.json").write_text(
        json.dumps({"args": vars(args), "elapsed_seconds": elapsed, "rows": len(run_df)}, indent=2),
        encoding="utf-8",
    )
    print(ci.to_string(index=False), flush=True)
    print(f"wrote {RESULTS_DIR / 'cifar10_external_tool_report.md'}", flush=True)


if __name__ == "__main__":
    run()
