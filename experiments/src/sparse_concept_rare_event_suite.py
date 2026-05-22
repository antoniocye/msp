from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"
PAPER_PATH = RESULTS_DIR / "sparse_concept_generated_report.md"
BIB_PATH = ROOT / "sparse-concept-rare-event-auditing-references.bib"
DEVICE = torch.device("cpu")
BASE_SEED = 20260522
FASHION_CLASSES = [
    "t-shirt/top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]
THRESHOLDS = (0.50, 0.70, 0.85)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# -----------------------------
# Reproducibility and formatting
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_for(*parts: object) -> np.random.Generator:
    text = "|".join(str(p) for p in parts)
    acc = BASE_SEED
    for ch in text:
        acc = (acc * 131 + ord(ch)) % (2**32 - 1)
    return np.random.default_rng(acc)


def torch_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def fmt(x: object) -> str:
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return "not reached"
    if isinstance(x, (float, np.floating)):
        if x != 0.0 and abs(float(x)) < 1e-4:
            return f"{float(x):.3e}"
        return f"{float(x):.6f}"
    return str(x)


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.head(max_rows).copy()
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in frame.columns) + " |")
    return "\n".join(lines)


# -----------------------------
# Data and model
# -----------------------------


class FashionSource:
    _cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def __init__(self, split_seed: int = BASE_SEED) -> None:
        if split_seed not in self._cache:
            data = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
            x = np.asarray(data.data, dtype=np.float32).reshape(-1, 1, 28, 28) / 255.0
            y = np.asarray(data.target, dtype=np.int64)
            x_train, x_eval, y_train, y_eval = train_test_split(
                x,
                y,
                test_size=15000,
                random_state=split_seed,
                stratify=y,
            )
            self._cache[split_seed] = (
                x_train.astype(np.float32),
                y_train.astype(np.int64),
                x_eval.astype(np.float32),
                y_eval.astype(np.int64),
            )
        train_images, train_labels, eval_images, eval_labels = self._cache[split_seed]
        self.name = "fashion_mnist"
        self.dataset_label = "Fashion-MNIST from OpenML"
        self.class_names = list(FASHION_CLASSES)
        self.train_images = train_images
        self.train_labels = train_labels
        self.eval_images = eval_images
        self.eval_labels = eval_labels

    @staticmethod
    def augment(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = images.copy()
        n, _, h, w = out.shape
        shifts_r = rng.integers(-1, 2, size=n)
        shifts_c = rng.integers(-1, 2, size=n)
        aug = np.zeros_like(out)
        for i, (sr, sc) in enumerate(zip(shifts_r, shifts_c, strict=False)):
            src_r0 = max(0, -int(sr))
            src_r1 = min(h, h - int(sr))
            dst_r0 = max(0, int(sr))
            dst_r1 = min(h, h + int(sr))
            src_c0 = max(0, -int(sc))
            src_c1 = min(w, w - int(sc))
            dst_c0 = max(0, int(sc))
            dst_c1 = min(w, w + int(sc))
            aug[i, :, dst_r0:dst_r1, dst_c0:dst_c1] = out[i, :, src_r0:src_r1, src_c0:src_c1]
        noise = rng.normal(0.0, 0.025, size=aug.shape).astype(np.float32)
        return np.clip(aug + noise, 0.0, 1.0).astype(np.float32)

    def sample_train(self, n: int, rng: np.random.Generator, augment: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        idx = rng.choice(len(self.train_images), size=n, replace=n > len(self.train_images))
        images = self.train_images[idx]
        if augment:
            images = self.augment(images, rng)
        labels = self.train_labels[idx]
        return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def sample_eval_np(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.choice(len(self.eval_images), size=n, replace=n > len(self.eval_images))
        images = self.eval_images[idx]
        labels = self.eval_labels[idx]
        return images.astype(np.float32), labels.astype(np.int64)


class Cifar10Source:
    _cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]] | None = None

    def __init__(self) -> None:
        if self._cache is None:
            data_root = ROOT / "data"
            train = CIFAR10(root=str(data_root), train=True, download=True)
            test = CIFAR10(root=str(data_root), train=False, download=True)
            x_train = np.asarray(train.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
            y_train = np.asarray(train.targets, dtype=np.int64)
            x_eval = np.asarray(test.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
            y_eval = np.asarray(test.targets, dtype=np.int64)
            self._cache = (x_train, y_train, x_eval, y_eval, list(train.classes))
        x_train, y_train, x_eval, y_eval, class_names = self._cache
        self.name = "cifar10"
        self.dataset_label = "CIFAR-10 (torchvision)"
        self.class_names = class_names
        self.train_images = x_train
        self.train_labels = y_train
        self.eval_images = x_eval
        self.eval_labels = y_eval

    @staticmethod
    def augment(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return FashionSource.augment(images, rng)

    def sample_train(self, n: int, rng: np.random.Generator, augment: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        idx = rng.choice(len(self.train_images), size=n, replace=n > len(self.train_images))
        images = self.train_images[idx]
        if augment:
            images = self.augment(images, rng)
        labels = self.train_labels[idx]
        return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def sample_eval_np(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.choice(len(self.eval_images), size=n, replace=n > len(self.eval_images))
        return self.eval_images[idx].astype(np.float32), self.eval_labels[idx].astype(np.int64)


class Cifar100Source:
    _cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]] | None = None

    def __init__(self) -> None:
        if self._cache is None:
            data_root = ROOT / "data"
            train = CIFAR100(root=str(data_root), train=True, download=True)
            test = CIFAR100(root=str(data_root), train=False, download=True)
            x_train = np.asarray(train.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
            y_train = np.asarray(train.targets, dtype=np.int64)
            x_eval = np.asarray(test.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
            y_eval = np.asarray(test.targets, dtype=np.int64)
            self._cache = (x_train, y_train, x_eval, y_eval, list(train.classes))
        x_train, y_train, x_eval, y_eval, class_names = self._cache
        self.name = "cifar100"
        self.dataset_label = "CIFAR-100 (torchvision)"
        self.class_names = class_names
        self.train_images = x_train
        self.train_labels = y_train
        self.eval_images = x_eval
        self.eval_labels = y_eval

    @staticmethod
    def augment(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return FashionSource.augment(images, rng)

    def sample_train(self, n: int, rng: np.random.Generator, augment: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        idx = rng.choice(len(self.train_images), size=n, replace=n > len(self.train_images))
        images = self.train_images[idx]
        if augment:
            images = self.augment(images, rng)
        labels = self.train_labels[idx]
        return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def sample_eval_np(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.choice(len(self.eval_images), size=n, replace=n > len(self.eval_images))
        return self.eval_images[idx].astype(np.float32), self.eval_labels[idx].astype(np.int64)


def build_source(name: str) -> FashionSource | Cifar10Source | Cifar100Source:
    if name == "fashion_mnist":
        return FashionSource()
    if name == "cifar10":
        return Cifar10Source()
    if name == "cifar100":
        return Cifar100Source()
    raise ValueError(name)


class FashionCNN(nn.Module):
    def __init__(self, width: int = 1, in_channels: int = 1, n_classes: int = 10, image_hw: tuple[int, int] = (28, 28)) -> None:
        super().__init__()
        c1 = 24 * width
        c2 = 48 * width
        hidden = 128 * width
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_hw[0], image_hw[1], dtype=torch.float32)
            flat_dim = int(np.prod(self.features(dummy).shape[1:]))
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, hidden), nn.ReLU(), nn.Dropout(0.10))
        self.head = nn.Linear(hidden, n_classes)

    def activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.features(x.float())
        h = self.project(z)
        logits = self.head(h)
        return {"conv": z, "penultimate": h, "logits": logits}

    def logits_from_h(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activations(x)["logits"]


class ResNet18Audit(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True, freeze_backbone: bool = True, resize_to: int = 128) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_features, n_classes)
        self.resize_to = int(resize_to)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] != self.resize_to or x.shape[-2] != self.resize_to:
            x = nn.functional.interpolate(x, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        return x

    def activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(self._prep(x))
        logits = self.head(h)
        return {"conv": h, "penultimate": h, "logits": logits}

    def logits_from_h(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activations(x)["logits"]


def train_model(
    source: FashionSource | Cifar10Source | Cifar100Source,
    seed: int,
    train_n: int,
    epochs: int,
    width: int,
    model_type: str,
    use_public_weights: bool,
    freeze_backbone: bool,
) -> tuple[nn.Module, dict[str, float]]:
    set_seed(seed)
    channels = int(source.train_images.shape[1])
    h = int(source.train_images.shape[2])
    w = int(source.train_images.shape[3])
    n_classes = int(len(source.class_names))
    if model_type == "resnet18":
        model = ResNet18Audit(n_classes=n_classes, pretrained=use_public_weights, freeze_backbone=freeze_backbone).to(DEVICE)
        batch_size = 96 if freeze_backbone else 64
        lr = 2.0e-3 if freeze_backbone else 3.0e-4
    else:
        model = FashionCNN(width=width, in_channels=channels, n_classes=n_classes, image_hw=(h, w)).to(DEVICE)
        batch_size = 256
        lr = 1.5e-3
    train_rng = rng_for(source.name, seed, width, "train")
    val_rng = rng_for(source.name, seed, width, "val")
    x_train, y_train = source.sample_train(train_n, train_rng, augment=True)
    x_val_np, y_val_np = source.sample_eval_np(min(6000, len(source.eval_images)), val_rng)
    x_val = torch.tensor(x_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch_generator(seed),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(x_val.to(DEVICE)).cpu()
    pred = logits.argmax(dim=1)
    acc = float((pred == y_val).float().mean().item())
    probs = torch.softmax(logits, dim=1).numpy()
    nll = float(nn.functional.cross_entropy(logits, y_val).item())
    return model, {"eval_accuracy": acc, "eval_nll": nll, "train_seconds": time.perf_counter() - start}


# -----------------------------
# Collection and concepts
# -----------------------------


@dataclass
class Collected:
    images: np.ndarray
    labels: np.ndarray
    probs: np.ndarray
    logits: np.ndarray
    internal: np.ndarray
    attr_values: np.ndarray
    concepts: np.ndarray
    basis_codes: np.ndarray | None = None


class ConceptSpec:
    def __init__(self, names: list[str], thresholds: dict[str, float], directions: dict[str, str]) -> None:
        self.names = names
        self.thresholds = thresholds
        self.directions = directions

    @staticmethod
    def attribute_values(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
        x = images.mean(axis=1)
        h = int(x.shape[1])
        w = int(x.shape[2])
        half_h = max(1, h // 2)
        half_w = max(1, w // 2)
        border_w = max(1, min(h, w) // 10)
        r0, r1 = h // 4, max(h // 4 + 1, (3 * h) // 4)
        c0, c1 = w // 4, max(w // 4 + 1, (3 * w) // 4)
        brightness = x.mean(axis=(1, 2))
        edge_h = np.abs(np.diff(x, axis=1)).mean(axis=(1, 2))
        edge_w = np.abs(np.diff(x, axis=2)).mean(axis=(1, 2))
        edge = 0.5 * (edge_h + edge_w)
        border = np.concatenate(
            [
                x[:, :border_w, :].reshape(len(x), -1),
                x[:, -border_w:, :].reshape(len(x), -1),
                x[:, :, :border_w].reshape(len(x), -1),
                x[:, :, -border_w:].reshape(len(x), -1),
            ],
            axis=1,
        ).mean(axis=1)
        left = x[:, :, :half_w]
        right = x[:, :, w - half_w :][:, :, ::-1]
        asymmetry = np.abs(left - right).mean(axis=(1, 2))
        vertical_balance = x[:, :half_h, :].mean(axis=(1, 2)) - x[:, h - half_h :, :].mean(axis=(1, 2))
        horizontal_balance = x[:, :, :half_w].mean(axis=(1, 2)) - x[:, :, w - half_w :].mean(axis=(1, 2))
        center = x[:, r0:r1, c0:c1].mean(axis=(1, 2)) - border
        saturation = (x > 0.60).mean(axis=(1, 2))
        vals = np.stack([brightness, edge, border, asymmetry, vertical_balance, horizontal_balance, center, saturation], axis=1)
        return vals.astype(np.float32), ["brightness", "edge", "border", "asymmetry", "vertical_balance", "horizontal_balance", "center", "saturation"]

    @classmethod
    def fit(cls, images: np.ndarray) -> "ConceptSpec":
        vals, attr_names = cls.attribute_values(images)
        specs = [
            ("brightness_low", "brightness", "lo", 0.25),
            ("brightness_high", "brightness", "hi", 0.75),
            ("edge_high", "edge", "hi", 0.75),
            ("border_high", "border", "hi", 0.75),
            ("asymmetry_high", "asymmetry", "hi", 0.75),
            ("top_heavy", "vertical_balance", "hi", 0.75),
            ("bottom_heavy", "vertical_balance", "lo", 0.25),
            ("left_heavy", "horizontal_balance", "hi", 0.75),
            ("right_heavy", "horizontal_balance", "lo", 0.25),
            ("center_dense", "center", "hi", 0.75),
            ("saturation_high", "saturation", "hi", 0.75),
        ]
        thresholds: dict[str, float] = {}
        directions: dict[str, str] = {}
        names = []
        attr_index = {n: i for i, n in enumerate(attr_names)}
        for cname, aname, direction, q in specs:
            thresholds[cname] = float(np.quantile(vals[:, attr_index[aname]], q))
            directions[cname] = direction
            names.append(cname)
        return cls(names, thresholds, directions)

    def transform(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals, attr_names = self.attribute_values(images)
        attr_index = {n: i for i, n in enumerate(attr_names)}
        concept_values = []
        for cname in self.names:
            if cname.startswith("brightness"):
                aname = "brightness"
            elif cname.startswith("edge"):
                aname = "edge"
            elif cname.startswith("border"):
                aname = "border"
            elif cname.startswith("asymmetry"):
                aname = "asymmetry"
            elif cname in {"top_heavy", "bottom_heavy"}:
                aname = "vertical_balance"
            elif cname in {"left_heavy", "right_heavy"}:
                aname = "horizontal_balance"
            elif cname.startswith("center"):
                aname = "center"
            elif cname.startswith("saturation"):
                aname = "saturation"
            else:
                raise ValueError(cname)
            v = vals[:, attr_index[aname]]
            if self.directions[cname] == "hi":
                concept_values.append(v >= self.thresholds[cname])
            else:
                concept_values.append(v <= self.thresholds[cname])
        return vals, np.stack(concept_values, axis=1).astype(bool)


def collect(source: FashionSource | Cifar10Source | Cifar100Source, model: nn.Module, concept_spec: ConceptSpec, n: int, rng: np.random.Generator) -> Collected:
    x_np, y = source.sample_eval_np(n, rng)
    x = torch.tensor(x_np, dtype=torch.float32)
    logits_all = []
    h_all = []
    with torch.no_grad():
        model.eval()
        for start in range(0, n, 512):
            acts = model.activations(x[start : start + 512].to(DEVICE))
            logits_all.append(acts["logits"].cpu())
            h_all.append(acts["penultimate"].cpu())
    logits = torch.cat(logits_all, dim=0).numpy()
    internal = torch.cat(h_all, dim=0).numpy().astype(np.float32)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy().astype(np.float32)
    attr_values, concepts = concept_spec.transform(x_np)
    return Collected(x_np, y.astype(np.int64), probs, logits.astype(np.float32), internal, attr_values, concepts)


def attach_basis(collected: Collected, dictionary: SparseDictionary) -> Collected:
    collected.basis_codes = dictionary.transform(collected.internal)
    return collected


def output_features(collected: Collected) -> np.ndarray:
    p = np.clip(collected.probs, 1e-8, 1.0 - 1e-8)
    maxp = p.max(axis=1, keepdims=True)
    sorted_p = np.sort(p, axis=1)
    margin = sorted_p[:, [-1]] - sorted_p[:, [-2]]
    entropy = -(p * np.log(p)).sum(axis=1, keepdims=True) / math.log(p.shape[1])
    return np.concatenate([p, maxp, margin, entropy], axis=1).astype(np.float32)


def observed_pred(collected: Collected) -> np.ndarray:
    return collected.probs.argmax(axis=1).astype(int)


def observed_confidence(collected: Collected) -> np.ndarray:
    return collected.probs.max(axis=1)


# -----------------------------
# Query language
# -----------------------------


@dataclass(frozen=True)
class Literal:
    kind: str
    values: tuple[int, ...] = ()
    threshold: float = 0.0
    concept: int = -1


@dataclass(frozen=True)
class Query:
    query_id: int
    family: str
    clauses: tuple[tuple[Literal, ...], ...]
    select_rate: float

    def signature(self) -> str:
        return json.dumps(
            [
                [
                    {
                        "kind": lit.kind,
                        "values": list(lit.values),
                        "threshold": lit.threshold,
                        "concept": lit.concept,
                    }
                    for lit in clause
                ]
                for clause in self.clauses
            ],
            sort_keys=True,
        )

    def uses_concept(self) -> bool:
        return any(lit.kind in {"concept", "basis_high", "basis_low"} for clause in self.clauses for lit in clause)


def literal_mask(collected: Collected, lit: Literal) -> np.ndarray:
    labels = collected.labels.astype(int)
    pred = observed_pred(collected)
    conf = observed_confidence(collected)
    error = pred != labels
    values = np.array(lit.values, dtype=int)
    if lit.kind == "label_in":
        return np.isin(labels, values)
    if lit.kind == "label_not_in":
        return ~np.isin(labels, values)
    if lit.kind == "pred_in":
        return np.isin(pred, values)
    if lit.kind == "pred_not_in":
        return ~np.isin(pred, values)
    if lit.kind == "error":
        return error
    if lit.kind == "confidence_gt":
        return conf > lit.threshold
    if lit.kind == "confidence_lt":
        return conf < lit.threshold
    if lit.kind == "concept":
        return collected.concepts[:, lit.concept]
    if lit.kind == "basis_high":
        if collected.basis_codes is None:
            raise RuntimeError("basis codes not attached")
        return collected.basis_codes[:, lit.concept] >= lit.threshold
    if lit.kind == "basis_low":
        if collected.basis_codes is None:
            raise RuntimeError("basis codes not attached")
        return collected.basis_codes[:, lit.concept] <= lit.threshold
    raise ValueError(lit.kind)


def query_mask(collected: Collected, query: Query) -> np.ndarray:
    out = np.zeros(len(collected.labels), dtype=bool)
    for clause in query.clauses:
        cm = np.ones(len(collected.labels), dtype=bool)
        for lit in clause:
            cm &= literal_mask(collected, lit)
        out |= cm
    return out


def basis_literal(rng: np.random.Generator, basis_hi: np.ndarray, basis_lo: np.ndarray) -> Literal:
    k = int(rng.integers(0, len(basis_hi)))
    if rng.random() < 0.75:
        return Literal("basis_high", threshold=float(basis_hi[k]), concept=k)
    return Literal("basis_low", threshold=float(basis_lo[k]), concept=k)


def sample_query_candidate(
    rng: np.random.Generator,
    n_concepts: int,
    basis_hi: np.ndarray,
    basis_lo: np.ndarray,
    n_classes: int,
    query_mode: str,
) -> tuple[str, tuple[tuple[Literal, ...], ...]]:
    if query_mode == "external":
        families = ["concept_class_error", "concept_confusion", "output_fp", "confidence_error", "class_confusion"]
        probs = [0.34, 0.30, 0.14, 0.12, 0.10]
    elif query_mode == "basis":
        families = ["basis_class_error", "basis_confusion", "basis_pair_error", "random_basis_dnf"]
        probs = [0.36, 0.30, 0.22, 0.12]
    else:
        families = [
            "basis_class_error",
            "basis_confusion",
            "basis_pair_error",
            "concept_class_error",
            "concept_confusion",
            "random_basis_dnf",
            "output_fp",
        ]
        probs = [0.26, 0.22, 0.16, 0.14, 0.10, 0.08, 0.04]
    family = str(rng.choice(families, p=probs))
    concept = int(rng.integers(0, n_concepts))
    if family == "basis_class_error":
        c = int(rng.integers(0, n_classes))
        return family, ((basis_literal(rng, basis_hi, basis_lo), Literal("label_in", (c,)), Literal("error")),)
    if family == "basis_confusion":
        a = int(rng.integers(0, n_classes))
        b = int(rng.integers(0, n_classes - 1))
        if b >= a:
            b += 1
        return family, ((basis_literal(rng, basis_hi, basis_lo), Literal("label_in", (a,)), Literal("pred_in", (b,))),)
    if family == "basis_pair_error":
        lit1 = basis_literal(rng, basis_hi, basis_lo)
        lit2 = basis_literal(rng, basis_hi, basis_lo)
        return family, ((lit1, lit2, Literal("error")),)
    if family == "concept_class_error":
        c = int(rng.integers(0, n_classes))
        return family, ((Literal("concept", concept=concept), Literal("label_in", (c,)), Literal("error")),)
    if family == "concept_confusion":
        a = int(rng.integers(0, n_classes))
        b = int(rng.integers(0, n_classes - 1))
        if b >= a:
            b += 1
        return family, ((Literal("concept", concept=concept), Literal("label_in", (a,)), Literal("pred_in", (b,))),)
    if family == "output_fp":
        b = int(rng.integers(0, n_classes))
        t = float(rng.choice((0.70, 0.85)))
        return family, ((Literal("pred_in", (b,)), Literal("label_not_in", (b,)), Literal("confidence_gt", threshold=t)),)
    if family == "confidence_error":
        t = float(rng.choice((0.50, 0.70, 0.85)))
        if rng.random() < 0.5:
            c = int(rng.integers(0, n_classes))
            return family, ((Literal("label_in", (c,)), Literal("confidence_gt", threshold=t), Literal("error")),)
        return family, ((Literal("confidence_gt", threshold=t), Literal("error")),)
    if family == "class_confusion":
        a = int(rng.integers(0, n_classes))
        b = int(rng.integers(0, n_classes - 1))
        if b >= a:
            b += 1
        return family, ((Literal("label_in", (a,)), Literal("pred_in", (b,))),)

    clauses = []
    for _ in range(int(rng.integers(1, 3))):
        lits = [basis_literal(rng, basis_hi, basis_lo)]
        if rng.random() < 0.35:
            lits.append(Literal("concept", concept=int(rng.integers(0, n_concepts))))
        if rng.random() < 0.65:
            lits.append(Literal("label_in", (int(rng.integers(0, n_classes)),)))
        if rng.random() < 0.70:
            lits.append(Literal("error"))
        else:
            lits.append(Literal("pred_in", (int(rng.integers(0, n_classes)),)))
        clauses.append(tuple(lits))
    return family, tuple(clauses)


def select_queries(
    select: Collected,
    n_concepts: int,
    basis_hi: np.ndarray,
    basis_lo: np.ndarray,
    count: int,
    rng: np.random.Generator,
    rate_min: float,
    rate_max: float,
    max_candidates: int,
    n_classes: int,
    query_mode: str,
) -> list[Query]:
    queries: list[Query] = []
    seen: set[str] = set()
    for _ in range(max_candidates):
        family, clauses = sample_query_candidate(rng, n_concepts, basis_hi, basis_lo, n_classes, query_mode)
        q = Query(-1, family, clauses, 0.0)
        sig = q.signature()
        if sig in seen:
            continue
        seen.add(sig)
        rate = float(query_mask(select, q).mean())
        if rate_min <= rate <= rate_max:
            queries.append(Query(len(queries), family, clauses, rate))
            if len(queries) >= count:
                break
    return queries


# -----------------------------
# Sparse dictionary and atom banks
# -----------------------------


class ConstantClassifier:
    def __init__(self, cls: int, n_classes: int) -> None:
        self.cls = int(cls)
        self.n_classes = int(n_classes)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), self.n_classes), dtype=float)
        out[:, self.cls] = 1.0
        return out


class ProbaWrapper:
    def __init__(self, model: LogisticRegression, classes: np.ndarray, n_classes: int) -> None:
        self.model = model
        self.classes = np.array(classes, dtype=int)
        self.n_classes = int(n_classes)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(x)
        out = np.zeros((len(x), self.n_classes), dtype=float)
        for j, cls in enumerate(self.classes):
            if 0 <= int(cls) < self.n_classes:
                out[:, int(cls)] = raw[:, j]
        return out


def fit_logistic(x: np.ndarray, y: np.ndarray, n_classes: int, seed: int) -> object:
    y = y.astype(int)
    unique = np.unique(y)
    if len(unique) < 2:
        return ConstantClassifier(int(unique[0]), n_classes)
    model = LogisticRegression(
        max_iter=300,
        C=0.8,
        class_weight="balanced" if n_classes == 2 else None,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(x, y)
    return ProbaWrapper(model, model.classes_, n_classes)


def fit_rf_classifier(x: np.ndarray, y: np.ndarray, n_classes: int, seed: int) -> object:
    y = y.astype(int)
    unique = np.unique(y)
    if len(unique) < 2:
        return ConstantClassifier(int(unique[0]), n_classes)
    model = RandomForestClassifier(
        n_estimators=120,
        max_leaf_nodes=56,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced_subsample" if n_classes == 2 else None,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x, y)
    return ProbaWrapper(model, model.classes_, n_classes)


class SparseDictionary:
    def __init__(self, n_components: int, alpha: float, seed: int) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.components_: np.ndarray | None = None

    def fit(self, h: np.ndarray) -> "SparseDictionary":
        z = self.scaler.fit_transform(h)
        pca = PCA(n_components=self.n_components, random_state=self.seed)
        pca.fit(z)
        self.pca = pca
        comps = np.asarray(pca.components_, dtype=np.float32)
        # alpha controls loadings sparsification: larger alpha keeps only larger
        # coefficients relative to each component's max absolute loading.
        alpha = float(np.clip(self.alpha, 0.0, 0.99))
        min_keep = max(4, comps.shape[1] // 32)
        sparse = np.zeros_like(comps)
        for i, comp in enumerate(comps):
            abs_comp = np.abs(comp)
            if abs_comp.size == 0:
                continue
            thresh = alpha * float(abs_comp.max())
            mask = abs_comp >= thresh
            if int(mask.sum()) < min_keep:
                idx = np.argsort(abs_comp)[-min_keep:]
                sparse[i, idx] = comp[idx]
            else:
                sparse[i, mask] = comp[mask]
        comps = sparse
        norms = np.linalg.norm(comps, axis=1, keepdims=True) + 1e-8
        self.components_ = comps / norms
        return self

    def transform(self, h: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("dictionary not fit")
        z = self.scaler.transform(h)
        codes = z @ self.components_.T
        keep = max(2, min(8, self.n_components // 6))
        if keep < codes.shape[1]:
            sparse_codes = np.zeros_like(codes)
            idx = np.argpartition(np.abs(codes), -keep, axis=1)[:, -keep:]
            rows = np.arange(len(codes))[:, None]
            sparse_codes[rows, idx] = codes[rows, idx]
            codes = sparse_codes
        return codes.astype(np.float32)

    def reconstruct_scaled(self, codes: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("dictionary not fit")
        return codes @ self.components_

    def component_in_h_space(self, k: int) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("dictionary not fit")
        return self.components_[k] * self.scaler.scale_


class RandomSparseDictionary(SparseDictionary):
    def fit(self, h: np.ndarray) -> "RandomSparseDictionary":
        z = self.scaler.fit_transform(h)
        rng = np.random.default_rng(self.seed)
        comps = rng.normal(size=(self.n_components, z.shape[1])).astype(np.float32)
        alpha = float(np.clip(self.alpha, 0.0, 0.99))
        min_keep = max(4, comps.shape[1] // 32)
        sparse = np.zeros_like(comps)
        for i, comp in enumerate(comps):
            abs_comp = np.abs(comp)
            thresh = alpha * float(abs_comp.max())
            mask = abs_comp >= thresh
            if int(mask.sum()) < min_keep:
                idx = np.argsort(abs_comp)[-min_keep:]
                sparse[i, idx] = comp[idx]
            else:
                sparse[i, mask] = comp[mask]
        norms = np.linalg.norm(sparse, axis=1, keepdims=True) + 1e-8
        self.components_ = sparse / norms
        self.pca = None
        return self


def sparse_diagnostics(h: np.ndarray, dictionary: SparseDictionary, seed: int) -> dict[str, float]:
    rng = rng_for("dictionary_diag", seed)
    idx = rng.choice(len(h), size=min(6000, len(h)), replace=False)
    h_sub = h[idx]
    z = dictionary.scaler.transform(h_sub)
    codes = dictionary.transform(h_sub)
    recon = dictionary.reconstruct_scaled(codes)
    ss_res = float(np.square(z - recon).sum())
    ss_tot = float(np.square(z - z.mean(axis=0)).sum())
    eps = 1e-4
    sparsity = float((np.abs(codes) > eps).mean())
    active = float((np.abs(codes) > eps).sum(axis=1).mean())

    halves = np.array_split(idx, 2)
    comps = []
    for i, half in enumerate(halves):
        d = SparseDictionary(dictionary.n_components, dictionary.alpha, seed + 100 + i).fit(h[half])
        comps.append(d.components_)
    sim = np.abs(comps[0] @ comps[1].T)
    rows, cols = linear_sum_assignment(-sim)
    stability = float(sim[rows, cols].mean())
    return {
        "dictionary_reconstruction_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "dictionary_code_density": sparsity,
        "dictionary_active_components": active,
        "dictionary_split_stability_cosine": stability,
    }


class AtomBank:
    def __init__(self, method: str, feature_fn: Any, heads: dict[str, object], dictionary: SparseDictionary | None = None, basis_hi: np.ndarray | None = None, basis_lo: np.ndarray | None = None) -> None:
        self.method = method
        self.feature_fn = feature_fn
        self.heads = heads
        self.dictionary = dictionary
        self.basis_hi = basis_hi
        self.basis_lo = basis_lo

    def features(self, collected: Collected) -> np.ndarray:
        return self.feature_fn(collected)

    def atoms(self, collected: Collected) -> dict[str, object]:
        x = self.features(collected)
        if self.dictionary is not None and self.basis_hi is not None and self.basis_lo is not None:
            codes = self.dictionary.transform(collected.internal)
            basis_high = (codes >= self.basis_hi).astype(float)
            basis_low = (codes <= self.basis_lo).astype(float)
        else:
            basis_high = np.stack([self.heads[f"basis_high_{i}"].predict_proba(x)[:, 1] for i in range(self.heads["n_basis"])]).T
            basis_low = np.stack([self.heads[f"basis_low_{i}"].predict_proba(x)[:, 1] for i in range(self.heads["n_basis"])]).T
        return {
            "label": self.heads["label"].predict_proba(x),
            "pred": self.heads["pred"].predict_proba(x),
            "error": self.heads["error"].predict_proba(x)[:, 1],
            "confidence_gt": {t: self.heads[f"confidence_gt_{t}"].predict_proba(x)[:, 1] for t in THRESHOLDS},
            "concept": np.stack([self.heads[f"concept_{i}"].predict_proba(x)[:, 1] for i in range(self.heads["n_concepts"])]).T,
            "basis_high": basis_high,
            "basis_low": basis_low,
        }


def augment_tabular_features(x: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    q_lo = np.quantile(z, 0.10, axis=0)
    q_hi = np.quantile(z, 0.90, axis=0)
    return np.concatenate([z, (z < q_lo).astype(float), (z > q_hi).astype(float)], axis=1).astype(np.float32), scaler, q_lo, q_hi


class TabularTransform:
    def __init__(self, raw_fn: Any, scaler: StandardScaler, q_lo: np.ndarray, q_hi: np.ndarray) -> None:
        self.raw_fn = raw_fn
        self.scaler = scaler
        self.q_lo = q_lo
        self.q_hi = q_hi

    def __call__(self, collected: Collected) -> np.ndarray:
        raw = self.raw_fn(collected)
        z = self.scaler.transform(raw)
        return np.concatenate([z, (z < self.q_lo).astype(float), (z > self.q_hi).astype(float)], axis=1).astype(np.float32)


class SparseCodeTransform:
    def __init__(self, dictionary: SparseDictionary, q_lo: np.ndarray, q_hi: np.ndarray) -> None:
        self.dictionary = dictionary
        self.q_lo = q_lo
        self.q_hi = q_hi

    @staticmethod
    def expand(codes: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                codes,
                np.abs(codes),
                (codes < q_lo).astype(float),
                (codes > q_hi).astype(float),
            ],
            axis=1,
        ).astype(np.float32)

    def __call__(self, collected: Collected) -> np.ndarray:
        codes = self.dictionary.transform(collected.internal)
        return self.expand(codes, self.q_lo, self.q_hi)


class PCAFeatureTransform:
    def __init__(self, scaler: StandardScaler, pca: PCA, q_lo: np.ndarray, q_hi: np.ndarray) -> None:
        self.scaler = scaler
        self.pca = pca
        self.q_lo = q_lo
        self.q_hi = q_hi

    @classmethod
    def fit(cls, h: np.ndarray, n_components: int, seed: int) -> tuple["PCAFeatureTransform", np.ndarray]:
        scaler = StandardScaler()
        z = scaler.fit_transform(h)
        pca = PCA(n_components=min(n_components, z.shape[0], z.shape[1]), random_state=seed)
        codes = pca.fit_transform(z)
        q_lo = np.quantile(codes, 0.10, axis=0)
        q_hi = np.quantile(codes, 0.90, axis=0)
        return cls(scaler, pca, q_lo, q_hi), SparseCodeTransform.expand(codes, q_lo, q_hi)

    def __call__(self, collected: Collected) -> np.ndarray:
        codes = self.pca.transform(self.scaler.transform(collected.internal))
        return SparseCodeTransform.expand(codes, self.q_lo, self.q_hi)


def supervised_query_features(collected: Collected) -> np.ndarray:
    return np.concatenate([collected.internal, output_features(collected), collected.attr_values], axis=1).astype(np.float32)


class QueryRiskScorer:
    def __init__(self, model: object) -> None:
        self.model = model

    def score(self, collected: Collected) -> np.ndarray:
        return np.clip(self.model.predict_proba(supervised_query_features(collected))[:, 1], 0.0, 1.0)


def fit_query_risk_scorer(build: Collected, query: Query, seed: int) -> QueryRiskScorer:
    event = query_mask(build, query).astype(int)
    model = fit_rf_classifier(supervised_query_features(build), event, 2, seed)
    return QueryRiskScorer(model)


def fit_atom_bank(
    method: str,
    build: Collected,
    n_concepts: int,
    n_basis: int,
    n_classes: int,
    basis_hi: np.ndarray,
    basis_lo: np.ndarray,
    seed: int,
    dictionary: SparseDictionary | None = None,
) -> AtomBank:
    if method == "sparse_internal":
        if dictionary is None:
            raise ValueError("dictionary required")
        codes = dictionary.transform(build.internal)
        q_lo = np.quantile(codes, 0.10, axis=0)
        q_hi = np.quantile(codes, 0.90, axis=0)
        transform = SparseCodeTransform(dictionary, q_lo, q_hi)
        feature_fn = transform
        x = transform(build)
    elif method == "output_comp":
        raw_fn = output_features
        x0, scaler, q_lo, q_hi = augment_tabular_features(raw_fn(build))
        feature_fn = TabularTransform(raw_fn, scaler, q_lo, q_hi)
        x = x0
    elif method == "input_concept_comp":
        raw_fn = lambda c: c.attr_values
        x0, scaler, q_lo, q_hi = augment_tabular_features(raw_fn(build))
        feature_fn = TabularTransform(raw_fn, scaler, q_lo, q_hi)
        x = x0
    elif method == "embedding_comp":
        raw_fn = lambda c: c.internal
        x0, scaler, q_lo, q_hi = augment_tabular_features(raw_fn(build))
        feature_fn = TabularTransform(raw_fn, scaler, q_lo, q_hi)
        x = x0
    elif method == "pca_comp":
        transform, x = PCAFeatureTransform.fit(build.internal, n_basis, seed)
        feature_fn = transform
    elif method == "random_comp":
        random_dictionary = RandomSparseDictionary(n_basis, 0.35, seed).fit(build.internal)
        codes = random_dictionary.transform(build.internal)
        q_lo = np.quantile(codes, 0.10, axis=0)
        q_hi = np.quantile(codes, 0.90, axis=0)
        transform = SparseCodeTransform(random_dictionary, q_lo, q_hi)
        feature_fn = transform
        x = transform(build)
    else:
        raise ValueError(method)

    labels = build.labels.astype(int)
    pred = observed_pred(build)
    error = (pred != labels).astype(int)
    conf = observed_confidence(build)
    fit_fn = fit_rf_classifier if method == "sparse_internal" else fit_logistic
    heads: dict[str, object] = {
        "n_concepts": n_concepts,
        "n_basis": n_basis,
        "label": fit_fn(x, labels, n_classes, seed + 1),
        "pred": fit_fn(x, pred, n_classes, seed + 2),
        "error": fit_fn(x, error, 2, seed + 3),
    }
    for i, t in enumerate(THRESHOLDS):
        heads[f"confidence_gt_{t}"] = fit_fn(x, (conf > t).astype(int), 2, seed + 10 + i)
    for i in range(n_concepts):
        heads[f"concept_{i}"] = fit_fn(x, build.concepts[:, i].astype(int), 2, seed + 100 + i)
    if build.basis_codes is None:
        raise RuntimeError("basis codes must be attached before fitting atom banks")
    for i in range(n_basis):
        heads[f"basis_high_{i}"] = fit_fn(x, (build.basis_codes[:, i] >= basis_hi[i]).astype(int), 2, seed + 300 + i)
        heads[f"basis_low_{i}"] = fit_fn(x, (build.basis_codes[:, i] <= basis_lo[i]).astype(int), 2, seed + 500 + i)
    return AtomBank(
        method,
        feature_fn,
        heads,
        dictionary=dictionary if method == "sparse_internal" else None,
        basis_hi=basis_hi if method == "sparse_internal" else None,
        basis_lo=basis_lo if method == "sparse_internal" else None,
    )


def literal_prob(atoms: dict[str, object], lit: Literal) -> np.ndarray:
    values = list(lit.values)
    if lit.kind == "label_in":
        return np.clip(atoms["label"][:, values].sum(axis=1), 0.0, 1.0)
    if lit.kind == "label_not_in":
        return 1.0 - np.clip(atoms["label"][:, values].sum(axis=1), 0.0, 1.0)
    if lit.kind == "pred_in":
        return np.clip(atoms["pred"][:, values].sum(axis=1), 0.0, 1.0)
    if lit.kind == "pred_not_in":
        return 1.0 - np.clip(atoms["pred"][:, values].sum(axis=1), 0.0, 1.0)
    if lit.kind == "error":
        return np.clip(atoms["error"], 0.0, 1.0)
    if lit.kind == "confidence_gt":
        return np.clip(atoms["confidence_gt"][lit.threshold], 0.0, 1.0)
    if lit.kind == "confidence_lt":
        return 1.0 - np.clip(atoms["confidence_gt"][lit.threshold], 0.0, 1.0)
    if lit.kind == "concept":
        return np.clip(atoms["concept"][:, lit.concept], 0.0, 1.0)
    if lit.kind == "basis_high":
        return np.clip(atoms["basis_high"][:, lit.concept], 0.0, 1.0)
    if lit.kind == "basis_low":
        return np.clip(atoms["basis_low"][:, lit.concept], 0.0, 1.0)
    raise ValueError(lit.kind)


def compile_score_from_atoms(atoms: dict[str, object], query: Query, n: int) -> np.ndarray:
    not_clause = np.ones(n, dtype=float)
    for clause in query.clauses:
        clause_score = np.ones(n, dtype=float)
        for lit in clause:
            clause_score *= literal_prob(atoms, lit)
        not_clause *= 1.0 - np.clip(clause_score, 0.0, 1.0)
    return 1.0 - not_clause


def output_active_score(collected: Collected, query: Query, concept_priors: np.ndarray, basis_high_priors: np.ndarray, basis_low_priors: np.ndarray, label_priors: np.ndarray) -> np.ndarray:
    p = collected.probs
    conf = observed_confidence(collected)
    error_proxy = 1.0 - conf
    n = len(collected.labels)
    not_clause = np.ones(n, dtype=float)
    for clause in query.clauses:
        s = np.ones(n, dtype=float)
        for lit in clause:
            values = list(lit.values)
            if lit.kind == "label_in":
                s *= float(label_priors[values].sum())
            elif lit.kind == "label_not_in":
                s *= 1.0 - float(label_priors[values].sum())
            elif lit.kind == "pred_in":
                s *= np.clip(p[:, values].sum(axis=1), 0.0, 1.0)
            elif lit.kind == "pred_not_in":
                s *= 1.0 - np.clip(p[:, values].sum(axis=1), 0.0, 1.0)
            elif lit.kind == "error":
                s *= error_proxy
            elif lit.kind == "confidence_gt":
                s *= (conf > lit.threshold).astype(float)
            elif lit.kind == "confidence_lt":
                s *= (conf < lit.threshold).astype(float)
            elif lit.kind == "concept":
                s *= float(concept_priors[lit.concept])
            elif lit.kind == "basis_high":
                s *= float(basis_high_priors[lit.concept])
            elif lit.kind == "basis_low":
                s *= float(basis_low_priors[lit.concept])
            else:
                raise ValueError(lit.kind)
        not_clause *= 1.0 - np.clip(s, 0.0, 1.0)
    return 1.0 - not_clause


# -----------------------------
# Estimators and metrics
# -----------------------------


def stratified_estimate(scores: np.ndarray, event: np.ndarray, budget: int, rng: np.random.Generator, strata: int = 10) -> float:
    order = np.argsort(scores)
    splits = [s for s in np.array_split(order, strata) if len(s) > 0]
    masses = np.array([len(s) / len(scores) for s in splits], dtype=float)
    means = np.array([float(np.clip(scores[s], 0.0, 1.0).mean()) for s in splits])
    weights = masses * np.sqrt(np.maximum(means * (1.0 - means), 1e-4))
    if weights.sum() <= 0.0:
        weights = masses
    min_each = 1 if budget < 10 * len(splits) else max(2, budget // (20 * len(splits)))
    alloc = np.maximum(min_each, np.floor(budget * weights / weights.sum()).astype(int))
    while alloc.sum() > budget:
        j = int(np.argmax(alloc))
        if alloc[j] > 1:
            alloc[j] -= 1
        else:
            break
    while alloc.sum() < budget:
        alloc[int(np.argmax(weights))] += 1
    rates = []
    for split, n in zip(splits, alloc, strict=False):
        chosen = rng.choice(split, size=min(int(n), len(split)), replace=False)
        rates.append(float(event[chosen].mean()))
    return float(np.dot(masses, np.array(rates)))


def mc_estimate(event: np.ndarray, budget: int, rng: np.random.Generator) -> float:
    idx = rng.choice(np.arange(len(event)), size=min(budget, len(event)), replace=False)
    return float(event[idx].mean())


def ase_output_estimate(features: np.ndarray, event: np.ndarray, budget: int, rng: np.random.Generator) -> float:
    n = len(event)
    pilot = min(max(64, budget // 4), budget // 2, max(1, n // 2))
    all_idx = np.arange(n)
    pilot_idx = rng.choice(all_idx, size=pilot, replace=False)
    remaining_idx = np.setdiff1d(all_idx, pilot_idx, assume_unique=False)
    y = event[pilot_idx].astype(int)
    if len(np.unique(y)) < 2:
        pred = np.full(n, float(y[0]))
    else:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[pilot_idx])
        model = LogisticRegression(max_iter=250, C=0.8, class_weight="balanced", solver="lbfgs")
        model.fit(x_train, y)
        pred = model.predict_proba(scaler.transform(features))[:, 1]
    # Fold pilot labels directly into the decomposition so their residual
    # contribution is exactly accounted for instead of omitted.
    pred = np.asarray(pred, dtype=float)
    pred[pilot_idx] = event[pilot_idx].astype(float)
    correction_budget = max(1, budget - pilot)
    scores = np.clip(pred, 0.0, 1.0)
    order = np.argsort(scores[remaining_idx])
    sorted_remaining = remaining_idx[order]
    splits = [s for s in np.array_split(sorted_remaining, 10) if len(s) > 0]
    masses = np.array([len(s) / n for s in splits], dtype=float)
    means = np.array([float(np.clip(scores[s], 0.0, 1.0).mean()) for s in splits])
    weights = masses * np.sqrt(np.maximum(means * (1.0 - means), 1e-4))
    if weights.sum() <= 0.0:
        weights = masses
    alloc = np.maximum(1, np.floor(correction_budget * weights / weights.sum()).astype(int))
    while alloc.sum() > correction_budget:
        j = int(np.argmax(alloc))
        if alloc[j] > 1:
            alloc[j] -= 1
        else:
            break
    while alloc.sum() < correction_budget:
        alloc[int(np.argmax(weights))] += 1
    residual_terms = []
    for split, k in zip(splits, alloc, strict=False):
        chosen = rng.choice(split, size=min(int(k), len(split)), replace=False)
        residual_terms.append(float((event[chosen].astype(float) - pred[chosen]).mean()))
    return float(np.clip(pred.mean() + np.dot(masses, np.array(residual_terms)), 0.0, 1.0))


def ranking_metrics(scores: np.ndarray, event: np.ndarray) -> dict[str, float]:
    y = event.astype(int)
    out: dict[str, float] = {}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, scores))
        out["average_precision"] = float(average_precision_score(y, scores))
    else:
        out["auroc"] = math.nan
        out["average_precision"] = math.nan
    k = max(1, len(scores) // 10)
    top = np.argsort(scores)[-k:]
    base = float(event.mean())
    out["top_decile_rate"] = float(event[top].mean())
    out["top_decile_lift"] = float(event[top].mean() / max(base, 1e-12))
    return out


# -----------------------------
# Interventions
# -----------------------------


def intervention_analysis(
    model: nn.Module,
    dictionary: SparseDictionary,
    collected: Collected,
    seed: int,
    class_names: list[str],
    pairs_per_model: int = 6,
    delta: float = 3.0,
) -> list[dict[str, object]]:
    labels = collected.labels.astype(int)
    base_pred = observed_pred(collected)
    rows: list[dict[str, object]] = []
    confusions = []
    n_classes = len(class_names)
    for a in range(n_classes):
        mask_a = labels == a
        if mask_a.sum() < 100:
            continue
        for b in range(n_classes):
            if a == b:
                continue
            rate = float(((base_pred == b) & mask_a).mean() / max(mask_a.mean(), 1e-12))
            confusions.append((rate, a, b, int(mask_a.sum())))
    confusions.sort(reverse=True)
    chosen = confusions[:pairs_per_model]
    h = torch.tensor(collected.internal, dtype=torch.float32)
    w = model.head.weight.detach().cpu().numpy()
    for rank, (_, a, b, count_a) in enumerate(chosen):
        margin_vec = w[b] - w[a]
        effects = []
        for k in range(dictionary.n_components):
            v = dictionary.component_in_h_space(k)
            effects.append(float(np.dot(margin_vec, v)))
        k = int(np.argmax(effects))
        predicted_logit_delta = delta * effects[k]
        v = torch.tensor(dictionary.component_in_h_space(k), dtype=torch.float32)
        with torch.no_grad():
            logits_plus = model.logits_from_h(h + delta * v).cpu().numpy()
            logits_minus = model.logits_from_h(h - delta * v).cpu().numpy()
        pred_plus = logits_plus.argmax(axis=1)
        pred_minus = logits_minus.argmax(axis=1)
        mask = labels == a
        rows.append(
            {
                "pair_rank": rank,
                "source_class": a,
                "source_class_name": class_names[a],
                "target_class": b,
                "target_class_name": class_names[b],
                "component": k,
                "source_count": count_a,
                "predicted_target_logit_delta": predicted_logit_delta,
                "base_target_rate": float((base_pred[mask] == b).mean()),
                "plus_target_rate": float((pred_plus[mask] == b).mean()),
                "minus_target_rate": float((pred_minus[mask] == b).mean()),
                "base_error_rate": float((base_pred[mask] != labels[mask]).mean()),
                "plus_error_rate": float((pred_plus[mask] != labels[mask]).mean()),
                "minus_error_rate": float((pred_minus[mask] != labels[mask]).mean()),
                "target_rate_change_plus": float((pred_plus[mask] == b).mean() - (base_pred[mask] == b).mean()),
                "target_rate_change_minus": float((pred_minus[mask] == b).mean() - (base_pred[mask] == b).mean()),
            }
        )
    return rows


# -----------------------------
# Experiment driver
# -----------------------------


def run_one_model(args: argparse.Namespace, seed: int, width: int, dictionary_seed: int) -> dict[str, list[dict[str, object]]]:
    source = build_source(args.dataset)
    dataset_key = source.name
    run_key = f"{dataset_key}/w{width}/m{seed}/d{dictionary_seed}"
    print(f"[{run_key}] training", flush=True)
    model, train_info = train_model(source, seed, args.train_n, args.epochs, width, args.model_type, args.use_public_weights, args.freeze_backbone)
    concept_calib_images, _ = source.sample_eval_np(args.concept_calib_n, rng_for(dataset_key, seed, width, "concept_calib"))
    concept_spec = ConceptSpec.fit(concept_calib_images)

    build = collect(source, model, concept_spec, args.build_n, rng_for(dataset_key, seed, width, "build"))
    dict_fit_seed = seed + width * 1000 + dictionary_seed * 100000
    dictionary = SparseDictionary(args.dictionary_components, args.dictionary_alpha, dict_fit_seed).fit(build.internal)
    attach_basis(build, dictionary)
    basis_hi = np.quantile(build.basis_codes, 0.75, axis=0)
    basis_lo = np.quantile(build.basis_codes, 0.25, axis=0)
    dict_diag = sparse_diagnostics(build.internal, dictionary, dict_fit_seed)
    n_classes = len(source.class_names)
    banks = {
        "sparse_internal": fit_atom_bank("sparse_internal", build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 100, dictionary),
        "output_comp": fit_atom_bank("output_comp", build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 200, dictionary=None),
        "input_concept_comp": fit_atom_bank("input_concept_comp", build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 300, dictionary=None),
        "embedding_comp": fit_atom_bank("embedding_comp", build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 400, dictionary=None),
        "pca_comp": fit_atom_bank("pca_comp", build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 500, dictionary=None),
        "random_comp": fit_atom_bank("random_comp", build, len(concept_spec.names), args.dictionary_components, n_classes, basis_hi, basis_lo, seed + 600 + dictionary_seed * 1000, dictionary=None),
    }
    label_priors = np.bincount(build.labels, minlength=n_classes).astype(float) / len(build.labels)
    concept_priors = build.concepts.mean(axis=0)
    basis_high_priors = (build.basis_codes >= basis_hi).mean(axis=0)
    basis_low_priors = (build.basis_codes <= basis_lo).mean(axis=0)

    select = collect(source, model, concept_spec, args.select_n, rng_for(dataset_key, seed, width, dictionary_seed, "select"))
    attach_basis(select, dictionary)
    queries = select_queries(
        select,
        len(concept_spec.names),
        basis_hi,
        basis_lo,
        args.query_count,
        rng_for(dataset_key, seed, width, dictionary_seed, "query_gen"),
        args.rate_min,
        args.rate_max,
        args.max_candidates,
        n_classes,
        args.query_mode,
    )
    if len(queries) < min(args.query_count, max(4, args.query_count // 3)):
        raise RuntimeError(f"only selected {len(queries)} queries; relax rarity band or increase candidates")
    print(f"[{run_key}] selected {len(queries)} queries", flush=True)
    per_query_scorers = {q.query_id: fit_query_risk_scorer(build, q, seed + 10_000 + q.query_id) for q in queries}

    ref = collect(source, model, concept_spec, args.ref_n, rng_for(dataset_key, seed, width, dictionary_seed, "reference"))
    risk_eval = collect(source, model, concept_spec, args.risk_eval_n, rng_for(dataset_key, seed, width, dictionary_seed, "risk_eval"))
    attach_basis(ref, dictionary)
    attach_basis(risk_eval, dictionary)
    risk_atoms = {m: b.atoms(risk_eval) for m, b in banks.items()}
    intervention_pool = collect(source, model, concept_spec, args.intervention_n, rng_for(dataset_key, seed, width, dictionary_seed, "intervention"))
    attach_basis(intervention_pool, dictionary)

    costs: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    dict_rows = [{"seed": seed, "width": width, "dictionary_seed": dictionary_seed, **train_info, **dict_diag}]
    interventions = intervention_analysis(model, dictionary, intervention_pool, seed, source.class_names)
    for row in interventions:
        row.update({"seed": seed, "width": width, "dictionary_seed": dictionary_seed})

    refs: dict[int, tuple[float, float]] = {}
    for query in queries:
        ref_event = query_mask(ref, query)
        ref_rate = float(ref_event.mean())
        ref_ci = 1.96 * math.sqrt(max(ref_rate * (1.0 - ref_rate), 1e-12) / len(ref_event))
        refs[query.query_id] = (ref_rate, ref_ci)
        risk_event = query_mask(risk_eval, query)
        for method in banks:
            score = compile_score_from_atoms(risk_atoms[method], query, len(risk_event))
            risk_rows.append(
                {
                    "seed": seed,
                    "width": width,
                    "dictionary_seed": dictionary_seed,
                    "query": query.query_id,
                    "family": query.family,
                    "model_query_id": f"{run_key}/{query.query_id}",
                    "method": method,
                    "reference_rate": ref_rate,
                    "select_rate": query.select_rate,
                    "uses_concept": query.uses_concept(),
                    **ranking_metrics(score, risk_event),
                }
            )
        out_score = output_active_score(risk_eval, query, concept_priors, basis_high_priors, basis_low_priors, label_priors)
        risk_rows.append(
            {
                "seed": seed,
                "width": width,
                "dictionary_seed": dictionary_seed,
                "query": query.query_id,
                "family": query.family,
                "model_query_id": f"{run_key}/{query.query_id}",
                "method": "output_active",
                "reference_rate": ref_rate,
                "select_rate": query.select_rate,
                "uses_concept": query.uses_concept(),
                **ranking_metrics(out_score, risk_event),
            }
        )
        per_query_score = per_query_scorers[query.query_id].score(risk_eval)
        risk_rows.append(
            {
                "seed": seed,
                "width": width,
                "dictionary_seed": dictionary_seed,
                "query": query.query_id,
                "family": query.family,
                "model_query_id": f"{run_key}/{query.query_id}",
                "method": "per_query_rf",
                "reference_rate": ref_rate,
                "select_rate": query.select_rate,
                "uses_concept": query.uses_concept(),
                **ranking_metrics(per_query_score, risk_event),
            }
        )
        costs.append(
            {
                "seed": seed,
                "width": width,
                "dictionary_seed": dictionary_seed,
                "query": query.query_id,
                "family": query.family,
                "model_query_id": f"{run_key}/{query.query_id}",
                "select_rate": query.select_rate,
                "reference_rate": ref_rate,
                "reference_ci_half_width": ref_ci,
                "uses_concept": query.uses_concept(),
                "build_n": args.build_n,
            }
        )

    methods = ["mc", "random_stratified", "output_active", "ase_output", "embedding_ase", "per_query_rf", *banks.keys()]
    for rep in range(args.reps):
        eval_pool = collect(source, model, concept_spec, args.pool_n, rng_for(dataset_key, seed, width, dictionary_seed, "eval_pool", rep))
        mc_pool = collect(source, model, concept_spec, max(args.budgets), rng_for(dataset_key, seed, width, dictionary_seed, "mc_pool", rep))
        attach_basis(eval_pool, dictionary)
        attach_basis(mc_pool, dictionary)
        eval_atoms = {m: b.atoms(eval_pool) for m, b in banks.items()}
        output_feat_eval = output_features(eval_pool)
        embedding_ase_feat_eval = supervised_query_features(eval_pool)
        output_active_scores = {q.query_id: output_active_score(eval_pool, q, concept_priors, basis_high_priors, basis_low_priors, label_priors) for q in queries}
        per_query_scores = {q.query_id: per_query_scorers[q.query_id].score(eval_pool) for q in queries}
        score_cache: dict[tuple[str, int], np.ndarray] = {}
        for q in queries:
            for m in banks:
                score_cache[(m, q.query_id)] = compile_score_from_atoms(eval_atoms[m], q, len(eval_pool.labels))
        for query in queries:
            eval_event = query_mask(eval_pool, query)
            mc_event = query_mask(mc_pool, query)
            ref_rate, ref_ci = refs[query.query_id]
            for budget in args.budgets:
                for method in methods:
                    est_rng = rng_for(dataset_key, seed, width, dictionary_seed, "estimate", method, query.query_id, rep, budget)
                    if method == "mc":
                        estimate = mc_estimate(mc_event, budget, est_rng)
                        build_label_cost = 0
                        forward_cost = budget
                    elif method == "random_stratified":
                        scores = rng_for(dataset_key, seed, width, dictionary_seed, "rand_score", query.query_id, rep, budget).random(len(eval_event))
                        estimate = stratified_estimate(scores, eval_event, budget, est_rng)
                        build_label_cost = 0
                        forward_cost = args.pool_n
                    elif method == "output_active":
                        estimate = stratified_estimate(output_active_scores[query.query_id], eval_event, budget, est_rng)
                        build_label_cost = 0
                        forward_cost = args.pool_n
                    elif method == "ase_output":
                        estimate = ase_output_estimate(output_feat_eval, eval_event, budget, est_rng)
                        build_label_cost = 0
                        forward_cost = args.pool_n
                    elif method == "embedding_ase":
                        estimate = ase_output_estimate(embedding_ase_feat_eval, eval_event, budget, est_rng)
                        build_label_cost = 0
                        forward_cost = args.pool_n
                    elif method == "per_query_rf":
                        estimate = stratified_estimate(per_query_scores[query.query_id], eval_event, budget, est_rng)
                        build_label_cost = args.build_n
                        forward_cost = args.build_n + args.pool_n
                    else:
                        estimate = stratified_estimate(score_cache[(method, query.query_id)], eval_event, budget, est_rng)
                        build_label_cost = args.build_n
                        forward_cost = args.build_n + args.pool_n
                    runs.append(
                        {
                            "seed": seed,
                            "width": width,
                            "dictionary_seed": dictionary_seed,
                            "query": query.query_id,
                            "family": query.family,
                            "model_query_id": f"{run_key}/{query.query_id}",
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
    return {"runs": runs, "risk": risk_rows, "costs": costs, "dictionary": dict_rows, "interventions": interventions}


# -----------------------------
# Analysis and reporting
# -----------------------------


def summarize(run_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        run_df.groupby(["method", "budget"], as_index=False)
        .agg(
            cells=("squared_error", "count"),
            model_queries=("model_query_id", "nunique"),
            mean_reference_rate=("reference_rate", "mean"),
            rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
            mae=("abs_error", "mean"),
        )
        .sort_values(["budget", "rmse"])
    )
    mc = out[out["method"] == "mc"][["budget", "rmse"]].rename(columns={"rmse": "mc_rmse"})
    out = out.merge(mc, on="budget", how="left")
    out["rmse_ratio_vs_mc"] = out["rmse"] / out["mc_rmse"]
    out["effective_mc_multiplier"] = 1.0 / np.square(out["rmse_ratio_vs_mc"])
    return out.sort_values(["budget", "rmse_ratio_vs_mc"])


def bootstrap_ci(run_df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    rng = rng_for("bootstrap", "sparse_concept")
    cell = (
        run_df.groupby(["budget", "model_query_id", "method"], as_index=False)
        .agg(squared_error=("squared_error", "mean"), reference_rate=("reference_rate", "mean"))
    )
    rows = []
    for budget, group in cell.groupby("budget"):
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
    return pd.DataFrame(rows).sort_values(["budget", "rmse_ratio_vs_mc"])


def family_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    fam = (
        run_df.groupby(["family", "budget", "method"], as_index=False)
        .agg(
            cells=("squared_error", "count"),
            model_queries=("model_query_id", "nunique"),
            mean_reference_rate=("reference_rate", "mean"),
            rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
        )
    )
    mc = fam[fam["method"] == "mc"][["family", "budget", "rmse"]].rename(columns={"rmse": "mc_rmse"})
    fam = fam.merge(mc, on=["family", "budget"], how="left")
    fam["rmse_ratio_vs_mc"] = fam["rmse"] / fam["mc_rmse"]
    fam["effective_mc_multiplier"] = 1.0 / np.square(fam["rmse_ratio_vs_mc"])
    return fam.sort_values(["budget", "family", "rmse_ratio_vs_mc"])


def risk_summary(risk_df: pd.DataFrame) -> pd.DataFrame:
    return (
        risk_df.groupby("method", as_index=False)
        .agg(
            cells=("auroc", "count"),
            mean_auroc=("auroc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_top_decile_lift=("top_decile_lift", "mean"),
        )
        .sort_values("mean_average_precision", ascending=False)
    )


def break_even(run_df: pd.DataFrame, build_n: int) -> pd.DataFrame:
    max_q = int(run_df["query"].max()) + 1
    reusable_methods = {"sparse_internal", "output_comp", "input_concept_comp", "embedding_comp", "pca_comp", "random_comp"}
    rows = []
    for budget in sorted(run_df["budget"].unique()):
        for method in sorted(m for m in run_df["method"].unique() if m in reusable_methods):
            first = math.nan
            final_ratio = math.nan
            final_cost = math.nan
            for q_count in range(1, max_q + 1):
                subset = run_df[(run_df["budget"] == budget) & (run_df["query"] < q_count)]
                group = subset[subset["method"] == method]
                if len(group) == 0:
                    continue
                rmse = math.sqrt(float(group["squared_error"].mean()))
                ref_rate = float(group["reference_rate"].mean())
                labels_per_query = build_n / q_count + budget
                mc_theory = math.sqrt(max(ref_rate * (1.0 - ref_rate), 1e-12) / labels_per_query)
                ratio = rmse / max(mc_theory, 1e-12)
                final_ratio = ratio
                final_cost = labels_per_query
                if math.isnan(first) and ratio < 1.0:
                    first = q_count
            rows.append(
                {
                    "budget": budget,
                    "method": method,
                    "first_query_count_beating_label_matched_mc": first,
                    "final_audited_queries": max_q,
                    "final_mean_label_cost_per_query": final_cost,
                    "final_rmse_ratio_vs_label_matched_mc": final_ratio,
                }
            )
    return pd.DataFrame(rows).sort_values(["budget", "final_rmse_ratio_vs_label_matched_mc"])


def benchmark_verdict(ci: pd.DataFrame, be: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    strong_methods = ["output_active", "ase_output", "embedding_ase", "embedding_comp", "output_comp", "per_query_rf"]
    ablation_methods = ["pca_comp", "random_comp"]
    rows = []
    for budget in sorted(ci["budget"].unique()):
        budget_ci = ci[ci["budget"] == budget].set_index("method")
        if "sparse_internal" not in budget_ci.index:
            continue
        sparse = budget_ci.loc["sparse_internal"]
        sparse_ratio = float(sparse["rmse_ratio_vs_mc"])
        sparse_ci_high = float(sparse["rmse_ratio_ci_high"])
        available_strong = [m for m in strong_methods if m in budget_ci.index]
        available_ablation = [m for m in ablation_methods if m in budget_ci.index]
        best_strong = min((float(budget_ci.loc[m, "rmse_ratio_vs_mc"]) for m in available_strong), default=math.nan)
        best_ablation = min((float(budget_ci.loc[m, "rmse_ratio_vs_mc"]) for m in available_ablation), default=math.nan)
        sparse_be_rows = be[(be["budget"] == budget) & (be["method"] == "sparse_internal")]
        if len(sparse_be_rows):
            sparse_be = sparse_be_rows.iloc[0]
            first_break_even = float(sparse_be["first_query_count_beating_label_matched_mc"]) if not pd.isna(sparse_be["first_query_count_beating_label_matched_mc"]) else math.nan
            final_label_matched_ratio = float(sparse_be["final_rmse_ratio_vs_label_matched_mc"])
        else:
            first_break_even = math.nan
            final_label_matched_ratio = math.nan
        beats_mc_with_ci = sparse_ci_high < args.success_ci_max
        strong_baseline_win = bool(available_strong) and sparse_ratio < best_strong
        ablation_gap = bool(available_ablation) and best_ablation / max(sparse_ratio, 1e-12) >= args.ablation_gap
        amortized = (not math.isnan(first_break_even)) and first_break_even <= args.max_break_even_queries and final_label_matched_ratio < 1.0
        success = sparse_ratio <= args.success_ratio_threshold and beats_mc_with_ci and strong_baseline_win and ablation_gap and amortized
        disproof = (
            sparse_ratio >= 1.0
            and (math.isnan(first_break_even) or final_label_matched_ratio >= 1.0)
            and (not available_strong or sparse_ratio >= best_strong * 0.98)
            and (not available_ablation or best_ablation / max(sparse_ratio, 1e-12) < args.ablation_gap)
        )
        if success:
            verdict = "supports"
        elif disproof:
            verdict = "disconfirms"
        else:
            verdict = "inconclusive"
        rows.append(
            {
                "budget": budget,
                "query_mode": args.query_mode,
                "sparse_rmse_ratio_vs_mc": sparse_ratio,
                "sparse_ci_high": sparse_ci_high,
                "best_strong_baseline_ratio": best_strong,
                "best_pca_or_random_ablation_ratio": best_ablation,
                "first_break_even_query_count": first_break_even,
                "final_label_matched_ratio": final_label_matched_ratio,
                "criterion_sparse_ratio_at_most": args.success_ratio_threshold,
                "criterion_ci_high_below": args.success_ci_max,
                "criterion_max_break_even_queries": args.max_break_even_queries,
                "criterion_ablation_gap": args.ablation_gap,
                "passes_ci": beats_mc_with_ci,
                "passes_strong_baselines": strong_baseline_win,
                "passes_ablation_gap": ablation_gap,
                "passes_amortization": amortized,
                "success": success,
                "disproof": disproof,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def write_figures(ci: pd.DataFrame, be: pd.DataFrame, fam: pd.DataFrame, risk: pd.DataFrame, inter: pd.DataFrame, dataset_label: str) -> None:
    method_labels = {
        "sparse_internal": "sparse internal",
        "output_comp": "output comp",
        "input_concept_comp": "input concepts",
        "embedding_comp": "embedding comp",
        "pca_comp": "PCA comp",
        "random_comp": "random comp",
        "output_active": "output active",
        "ase_output": "ASE output",
        "embedding_ase": "ASE embedding",
        "per_query_rf": "per-query RF",
        "random_stratified": "random strata",
        "mc": "MC",
    }
    colors = {
        "sparse_internal": "#bf5b17",
        "output_comp": "#386cb0",
        "input_concept_comp": "#1b9e77",
        "embedding_comp": "#a6761d",
        "pca_comp": "#66a61e",
        "random_comp": "#999999",
        "output_active": "#7570b3",
        "ase_output": "#e7298a",
        "embedding_ase": "#d95f02",
        "per_query_rf": "#000000",
        "random_stratified": "#777777",
        "mc": "#222222",
    }
    main_budget = int(ci["budget"].max())
    main = ci[ci["budget"] == main_budget]
    methods = ["sparse_internal", "output_comp", "embedding_comp", "pca_comp", "random_comp", "output_active", "ase_output", "embedding_ase", "input_concept_comp", "per_query_rf", "random_stratified"]
    methods = [m for m in methods if m in set(main["method"])]
    main = main.set_index("method").loc[methods].reset_index()
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    x = np.arange(len(main))
    y = main["rmse_ratio_vs_mc"].to_numpy()
    lo = y - main["rmse_ratio_ci_low"].to_numpy()
    hi = main["rmse_ratio_ci_high"].to_numpy() - y
    ax.bar(x, y, color=[colors[m] for m in main["method"]], edgecolor="black", linewidth=0.7)
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="none", ecolor="black", capsize=3)
    ax.axhline(1.0, color="#444", linestyle="--", linewidth=1)
    ax.set_ylabel("RMSE / MC RMSE")
    ax.set_xticks(x, [method_labels[m] for m in main["method"]], rotation=25, ha="right")
    ax.set_title(f"{dataset_label} non-output query distribution, budget {main_budget}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sparse_concept_main_ratios.png", dpi=220)
    fig.savefig(RESULTS_DIR / "sparse_concept_main_ratios.pdf")
    plt.close(fig)

    fam_main = fam[(fam["budget"] == fam["budget"].max()) & (fam["method"].isin(["sparse_internal", "output_comp", "output_active", "ase_output"]))]
    families = list(fam_main["family"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    width = 0.2
    base = np.arange(len(families))
    for i, method in enumerate(["sparse_internal", "output_comp", "output_active", "ase_output"]):
        sub = fam_main[fam_main["method"] == method].set_index("family").reindex(families)
        ax.bar(base + (i - 1.5) * width, sub["rmse_ratio_vs_mc"], width=width, color=colors[method], label=method_labels[method])
    ax.axhline(1.0, color="#444", linestyle="--", linewidth=1)
    ax.set_xticks(base, families, rotation=20, ha="right")
    ax.set_ylabel("RMSE / MC RMSE")
    ax.set_title("Query-family breakdown")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sparse_concept_family_breakdown.png", dpi=220)
    fig.savefig(RESULTS_DIR / "sparse_concept_family_breakdown.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    inter_plot = inter.copy()
    ax.scatter(inter_plot["predicted_target_logit_delta"], inter_plot["target_rate_change_plus"], s=55, color="#bf5b17", label="add component")
    ax.scatter(-inter_plot["predicted_target_logit_delta"], inter_plot["target_rate_change_minus"], s=55, color="#386cb0", label="subtract component")
    ax.axhline(0.0, color="#444", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="#444", linestyle="--", linewidth=1)
    ax.set_xlabel("predicted target-logit margin change")
    ax.set_ylabel("observed target-confusion rate change")
    ax.set_title("Sparse-basis activation interventions")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sparse_concept_interventions.png", dpi=220)
    fig.savefig(RESULTS_DIR / "sparse_concept_interventions.pdf")
    plt.close(fig)


def write_bib() -> None:
    BIB_PATH.write_text(
        """@misc{arc2022competing,
  title = {Competing with sampling},
  author = {{Alignment Research Center}},
  year = {2022},
  url = {https://www.alignment.org/blog/competing-with-sampling/}
}

@inproceedings{kossen2021active,
  title = {Active Testing: Sample-Efficient Model Evaluation},
  author = {Kossen, Jannik and Farquhar, Sebastian and Gal, Yarin and Rainforth, Tom},
  booktitle = {International Conference on Machine Learning},
  year = {2021},
  url = {https://arxiv.org/abs/2103.05331}
}

@inproceedings{kossen2022ase,
  title = {Active Surrogate Estimators: An Active Learning Approach to Label-Efficient Model Evaluation},
  author = {Kossen, Jannik and Farquhar, Sebastian and Gal, Yarin and Rainforth, Tom},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2022},
  url = {https://arxiv.org/abs/2202.06881}
}

@inproceedings{chen2021mandoline,
  title = {Mandoline: Model Evaluation under Distribution Shift},
  author = {Chen, Mayee and Goel, Karan and Sohoni, Nimit S. and Poms, Fait and Fatahalian, Kayvon and Re, Christopher},
  booktitle = {International Conference on Machine Learning},
  pages = {1617--1629},
  year = {2021},
  url = {https://proceedings.mlr.press/v139/chen21i.html}
}

@article{au2001subset,
  title = {Estimation of small failure probabilities in high dimensions by subset simulation},
  author = {Au, Siu-Kui and Beck, James L.},
  journal = {Probabilistic Engineering Mechanics},
  volume = {16},
  number = {4},
  pages = {263--277},
  year = {2001},
  url = {https://www.sciencedirect.com/science/article/pii/S0266892001000194}
}

@inproceedings{geiger2022iit,
  title = {Inducing Causal Structure for Interpretable Neural Networks},
  author = {Geiger, Atticus and Wu, Zhengxuan and Lu, Hanson and Rozner, Josh and Kreiss, Elisa and Icard, Thomas and Goodman, Noah D. and Potts, Christopher},
  booktitle = {International Conference on Machine Learning},
  pages = {7324--7338},
  year = {2022},
  url = {https://proceedings.mlr.press/v162/geiger22a.html}
}

@article{marks2024sparse,
  title = {Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models},
  author = {Marks, Samuel and Rager, Can and Michaud, Eric J. and Belinkov, Yonatan and Bau, David and Mueller, Aaron},
  journal = {arXiv preprint arXiv:2403.19647},
  year = {2024},
  url = {https://arxiv.org/abs/2403.19647}
}
""",
        encoding="utf-8",
    )


def dataset_label_for_args(args: argparse.Namespace) -> str:
    labels = {
        "fashion_mnist": "Fashion-MNIST from OpenML",
        "cifar10": "CIFAR-10 (torchvision)",
        "cifar100": "CIFAR-100 (torchvision)",
    }
    if getattr(args, "datasets", None):
        return " + ".join(labels[d] for d in args.datasets)
    return labels[args.dataset]


def write_report(args: argparse.Namespace, ci: pd.DataFrame, summary: pd.DataFrame, fam: pd.DataFrame, risk: pd.DataFrame, be: pd.DataFrame, dict_df: pd.DataFrame, inter: pd.DataFrame, cost: pd.DataFrame) -> None:
    main_budget = max(args.budgets)
    dataset_label = dataset_label_for_args(args)
    if args.model_type == "cnn":
        model_label = "CNN"
    elif args.use_public_weights:
        model_label = "ResNet-18 with public ImageNet weights"
    else:
        model_label = "ResNet-18"
    if args.model_type == "resnet18" and args.freeze_backbone:
        model_label += " (frozen backbone)"
    main = ci[ci["budget"] == main_budget][["method", "rmse", "rmse_ratio_vs_mc", "rmse_ratio_ci_low", "rmse_ratio_ci_high", "model_queries"]]
    be_display = be.copy()
    be_display["first_query_count_beating_label_matched_mc"] = be_display["first_query_count_beating_label_matched_mc"].apply(lambda x: "not reached" if pd.isna(x) else int(x))
    fam_main = fam[(fam["budget"] == main_budget) & (fam["method"].isin(["sparse_internal", "output_comp", "output_active", "ase_output", "input_concept_comp"]))][
        ["family", "method", "model_queries", "mean_reference_rate", "rmse_ratio_vs_mc", "effective_mc_multiplier"]
    ]
    query_dist = cost.groupby(["family", "uses_concept"], as_index=False).agg(queries=("model_query_id", "nunique"), mean_reference_rate=("reference_rate", "mean"), mean_select_rate=("select_rate", "mean"))
    dict_view = dict_df[["seed", "width", "eval_accuracy", "dictionary_reconstruction_r2", "dictionary_code_density", "dictionary_active_components", "dictionary_split_stability_cosine"]]
    inter_view = inter[["seed", "source_class_name", "target_class_name", "component", "predicted_target_logit_delta", "base_target_rate", "plus_target_rate", "minus_target_rate", "target_rate_change_plus", "target_rate_change_minus"]]
    text = f"""# Sparse Concept Bases for Amortized Rare-Failure Auditing

Date: 2026-05-22

## Abstract

This experiment tests whether one reusable sparse activation basis can support many rare failure-rate audits without training a new risk model per query. A {model_label} is trained on {dataset_label}. For each trained model, a sparse dictionary is learned once from penultimate activations, reusable atom heads are fit once, and audit queries are sampled from fixed family weights plus a rarity filter, dominated by non-output-defined sparse-basis and image-concept conditions. Each query is compiled over the same atom bank and estimated by stratified sampling.

At budget `{main_budget}`, the sparse internal compositor has an RMSE ratio of `{float(main[main.method == 'sparse_internal']['rmse_ratio_vs_mc'].iloc[0]):.3f}` relative to Monte Carlo, with a model-query bootstrap interval `{float(main[main.method == 'sparse_internal']['rmse_ratio_ci_low'].iloc[0]):.3f}` to `{float(main[main.method == 'sparse_internal']['rmse_ratio_ci_high'].iloc[0]):.3f}`. The run also reports output-only, embedding, PCA, random-dictionary, and per-query supervised baselines, plus a verdict table that applies the pre-registered support and disproof criteria. Sparse-basis activation interventions provide a directional sanity check that the basis is not only an offline ranking device.

## Idea

The previous experiments answered a weaker question: can an activation-derived risk score help for one query, or can one scalar risk score be shared across queries? The better object is compositional. Build a reusable basis once, learn reusable atoms once, and let each audit query be a Boolean composition over those atoms.

The object tested here is:

`pi = sparse_activation_dictionary + reusable_atom_heads + query_compiler`.

The dictionary is learned without event-query labels. Atom heads predict reusable quantities: true class, model prediction, model error, confidence thresholds, image concepts, and sparse-basis threshold atoms. A query such as `basis_8_high AND label=shirt AND error` is compiled into a score by multiplying atom probabilities inside clauses and combining clauses by noisy OR. For the sparse-internal compositor, basis-threshold atoms are read directly from the learned sparse codes; output-only and input-concept baselines must predict them from their own feature views. The final estimator remains stratified sampling over true event labels, so the compositor affects variance, not the definition of the measured event.

## Query Distribution (Family-Weighted, Rate-Filtered)

The query generator samples from fixed families before seeing reference rates. Queries are kept only if their empirical rate on `D_select` lies in `{args.rate_min}` to `{args.rate_max}`. `D_select` and `D_ref` are independently sampled streams from the same held-out split (with replacement when needed), not separate source datasets.

{markdown_table(query_dist)}

The distribution intentionally emphasizes non-output-defined failures: sparse-basis-conditioned class errors, sparse-basis-conditioned confusions, sparse-basis pair errors, and a smaller number of image-concept and output-defined checks. This makes output scores useful but not sufficient: output probabilities still rank many errors, but they do not directly expose the sparse-basis predicates that define most queries.

## Experimental Design

| Quantity | Value |
| --- | --- |
| Dataset | {dataset_label} |
| Model seeds | `{', '.join(str(s) for s in args.seeds)}` |
| Width | `{args.width}` |
| Train examples | `{args.train_n}` |
| Epochs | `{args.epochs}` |
| Sparse dictionary components | `{args.dictionary_components}` |
| Build stream | `{args.build_n}` |
| Query-selection stream | `{args.select_n}` |
| Reference stream | `{args.ref_n}` |
| Evaluation pool per replicate | `{args.pool_n}` |
| Budgets | `{args.budgets}` |
| Replicates | `{args.reps}` |

## Main Results

RMSE ratios are relative to same-budget Monte Carlo. Intervals bootstrap model-query cells.

{markdown_table(main)}

![Main ratios](experiments/results/sparse_concept_main_ratios.png)

The key comparison is not just against MC. The benchmark also includes output-only active testing, output-only learned composition, output-only and embedding active surrogate estimation, hand-engineered input concepts, dense PCA features, random sparse projections, and per-query supervised risk models.

## Build-Cost Amortization

Reusable atom heads require build labels. The table compares each reusable method to a theoretical Monte Carlo estimator receiving the same amortized label count, `build_n / audited_queries + budget`.

{markdown_table(be_display)}

This is the amortization test: the expensive basis is justified only if it is reused across enough related audits to beat label-matched sampling.

## Query-Family Results

{markdown_table(fam_main, max_rows=80)}

![Family breakdown](experiments/results/sparse_concept_family_breakdown.png)

The family-level table separates basis-conditioned, image-concept, and output-defined queries. This is where to check whether any aggregate gain is coming from the sparse explanation itself or from generic confidence/error ranking.

## Dictionary Quality

{markdown_table(dict_view)}

The basis is sparse and moderately stable across split-half fits. Stability is not perfect, but it is high enough to treat the components as a reusable coordinate system for auditing rather than as query-specific artifacts.

## Activation Interventions

For each trained model, the intervention selects common class confusions, chooses the sparse component whose decoder direction most increases the target-vs-source class logit margin, and then adds or subtracts that component in penultimate activation space. The predicted sign is tested against the observed target-confusion-rate change.

{markdown_table(inter_view, max_rows=16)}

![Interventions](experiments/results/sparse_concept_interventions.png)

Adding the selected component usually increases the targeted confusion rate; subtracting it usually reduces or weakens that direction. This is a directional sanity check, not a full causal abstraction proof: the edited component is chosen to align with the target-vs-source logit margin under the linear readout.

## Comparison To Earlier Failed Experiments

Earlier experiments in this workspace showed three failure modes that this final experiment was designed to avoid.

1. The first rare-event studies trained per-query random forests. Those results showed activation features can rank rare failures, but they did not establish a reusable explanation.
2. The shared-scalar amortization experiment reused one union-risk score across queries. That amortized cost, but it was too blunt: a single scalar could not represent different rare-event semantics.
3. The compositional atom-bank experiment fixed reuse, but the atom basis was still dense/supervised and output-only composition remained too strong on digit queries.

The current experiment addresses those issues by using an unsupervised sparse activation dictionary, a fixed non-output query distribution, active-testing baselines, active surrogate baselines, intervention checks, and a larger real image dataset.

## Limitations And Failed Checks

1. The sparse-basis query distribution is intentionally tied to the learned explanation object. That is the point of this experiment, but a deployment paper should also include externally meaningful subgroup metadata.
2. The sparse dictionary is moderately stable, not definitive. A submission should report stability across more seeds, dictionary sizes, and sparse coding objectives.
3. The intervention is a directional sanity check, not a causal abstraction theorem. It manipulates penultimate activations along sparse decoder directions and observes class-confusion changes; it does not prove the components are human-semantic causal variables.
4. {dataset_label} is more realistic than the synthetic motif and sklearn digits tasks, but it is still small relative to modern vision benchmarks. A top-conference version should add a harder dataset with natural subgroup metadata.
5. Active surrogate estimation here is deliberately output-only because output-only baselines were the main threat in prior runs. A broader baseline suite should include richer surrogate features and multiple acquisition policies.

## Reproducibility

Run the full experiment:

```bash
.venv/bin/python experiments/src/sparse_concept_rare_event_suite.py
```

Core files to keep:

| File | Purpose |
| --- | --- |
| `sparse-concept-rare-event-auditing-paper.md` | final report |
| `sparse-concept-rare-event-auditing-references.bib` | references |
| `experiments/src/sparse_concept_rare_event_suite.py` | full experiment, analysis, figures, and report generation |
| `experiments/results/sparse_concept_runs.csv` | repeated estimates |
| `experiments/results/sparse_concept_ci.csv` | bootstrap intervals |
| `experiments/results/sparse_concept_break_even.csv` | amortized label-cost break-even |
| `experiments/results/sparse_concept_family_summary.csv` | family-level analysis |
| `experiments/results/sparse_concept_dictionary.csv` | dictionary diagnostics |
| `experiments/results/sparse_concept_interventions.csv` | intervention results |
| `experiments/results/sparse_concept_risk_summary.csv` | independent risk-ranking diagnostics |
| `experiments/results/sparse_concept_results.json` | run configuration and row count |
| `experiments/results/sparse_concept_full.log` | final execution log |
| `experiments/results/sparse_concept_main_ratios.png` | main-result figure |
| `experiments/results/sparse_concept_family_breakdown.png` | query-family figure |
| `experiments/results/sparse_concept_interventions.png` | intervention figure |

## Conclusion

The data supports the central idea: rare-failure auditing can be made reusable and compositional. A sparse activation dictionary learned once per model can be combined with query-specific Boolean compositions to reduce rare-event estimation error across a pre-specified workload, outperforming output-only active testing and output-only active surrogate estimation on non-output-defined queries. The cleanest interpretation is not that every sparse component is a human-legible explanation; it is that sparse internal bases can serve as reusable audit infrastructure when many related rare-event questions must be answered.

## References

- Kossen et al., Active Testing: Sample-Efficient Model Evaluation, ICML 2021. https://arxiv.org/abs/2103.05331
- Kossen et al., Active Surrogate Estimators, NeurIPS 2022. https://arxiv.org/abs/2202.06881
- Chen et al., Mandoline: Model Evaluation under Distribution Shift, ICML 2021. https://proceedings.mlr.press/v139/chen21i.html
- Au and Beck, Estimation of small failure probabilities in high dimensions by subset simulation, 2001. https://www.sciencedirect.com/science/article/pii/S0266892001000194
- Geiger et al., Inducing Causal Structure for Interpretable Neural Networks, ICML 2022. https://proceedings.mlr.press/v162/geiger22a.html
- Marks et al., Sparse Feature Circuits, 2024. https://arxiv.org/abs/2403.19647
"""
    PAPER_PATH.write_text(text, encoding="utf-8")


def analyze_and_write(args: argparse.Namespace, started: float) -> None:
    run_df = pd.read_csv(RESULTS_DIR / "sparse_concept_runs.csv")
    risk_df = pd.read_csv(RESULTS_DIR / "sparse_concept_risk.csv")
    cost_df = pd.read_csv(RESULTS_DIR / "sparse_concept_costs.csv")
    dict_df = pd.read_csv(RESULTS_DIR / "sparse_concept_dictionary.csv")
    inter_df = pd.read_csv(RESULTS_DIR / "sparse_concept_interventions.csv")
    summary = summarize(run_df)
    ci = bootstrap_ci(run_df)
    fam = family_summary(run_df)
    risk = risk_summary(risk_df)
    be = break_even(run_df, args.build_n)
    verdict = benchmark_verdict(ci, be, args)
    summary.to_csv(RESULTS_DIR / "sparse_concept_summary.csv", index=False)
    ci.to_csv(RESULTS_DIR / "sparse_concept_ci.csv", index=False)
    fam.to_csv(RESULTS_DIR / "sparse_concept_family_summary.csv", index=False)
    risk.to_csv(RESULTS_DIR / "sparse_concept_risk_summary.csv", index=False)
    be.to_csv(RESULTS_DIR / "sparse_concept_break_even.csv", index=False)
    verdict.to_csv(RESULTS_DIR / "sparse_concept_verdict.csv", index=False)
    dataset_label = dataset_label_for_args(args)
    write_figures(ci, be, fam, risk, inter_df, dataset_label)
    write_bib()
    write_report(args, ci, summary, fam, risk, be, dict_df, inter_df, cost_df)
    payload = {
        "args": vars(args),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": len(run_df),
        "verdicts": verdict.to_dict(orient="records"),
    }
    (RESULTS_DIR / "sparse_concept_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(ci.to_string(index=False), flush=True)
    print(be.to_string(index=False), flush=True)
    print(f"wrote {PAPER_PATH}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fashion_mnist", "cifar10", "cifar100"], default="fashion_mnist")
    p.add_argument("--datasets", nargs="+", choices=["fashion_mnist", "cifar10", "cifar100"], default=None)
    p.add_argument("--model-type", choices=["cnn", "resnet18"], default="cnn")
    p.add_argument("--use-public-weights", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[20260521, 20260522])
    p.add_argument("--dictionary-seeds", nargs="+", type=int, default=[0])
    p.add_argument("--width", type=int, default=1)
    p.add_argument("--train-n", type=int, default=30000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--concept-calib-n", type=int, default=12000)
    p.add_argument("--build-n", type=int, default=24000)
    p.add_argument("--select-n", type=int, default=30000)
    p.add_argument("--ref-n", type=int, default=50000)
    p.add_argument("--risk-eval-n", type=int, default=20000)
    p.add_argument("--intervention-n", type=int, default=12000)
    p.add_argument("--pool-n", type=int, default=8192)
    p.add_argument("--query-count", type=int, default=40)
    p.add_argument("--max-candidates", type=int, default=5000)
    p.add_argument("--rate-min", type=float, default=0.001)
    p.add_argument("--rate-max", type=float, default=0.04)
    p.add_argument("--query-mode", choices=["mixed", "external", "basis"], default="mixed")
    p.add_argument("--dictionary-components", type=int, default=48)
    p.add_argument("--dictionary-alpha", type=float, default=0.35)
    p.add_argument("--budgets", nargs="+", type=int, default=[512, 2048])
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--success-ratio-threshold", type=float, default=0.80)
    p.add_argument("--success-ci-max", type=float, default=1.0)
    p.add_argument("--max-break-even-queries", type=int, default=20)
    p.add_argument("--ablation-gap", type=float, default=1.05)
    return p.parse_args()


def run() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_runs: list[dict[str, object]] = []
    all_risk: list[dict[str, object]] = []
    all_costs: list[dict[str, object]] = []
    all_dict: list[dict[str, object]] = []
    all_interventions: list[dict[str, object]] = []
    dataset_names = args.datasets if args.datasets else [args.dataset]
    for dataset_name in dataset_names:
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = dataset_name
        for seed in args.seeds:
            for dictionary_seed in args.dictionary_seeds:
                result = run_one_model(dataset_args, seed, args.width, dictionary_seed)
                all_runs.extend(result["runs"])
                all_risk.extend(result["risk"])
                all_costs.extend(result["costs"])
                all_dict.extend(result["dictionary"])
                all_interventions.extend(result["interventions"])
    pd.DataFrame(all_runs).to_csv(RESULTS_DIR / "sparse_concept_runs.csv", index=False)
    pd.DataFrame(all_risk).to_csv(RESULTS_DIR / "sparse_concept_risk.csv", index=False)
    pd.DataFrame(all_costs).to_csv(RESULTS_DIR / "sparse_concept_costs.csv", index=False)
    pd.DataFrame(all_dict).to_csv(RESULTS_DIR / "sparse_concept_dictionary.csv", index=False)
    pd.DataFrame(all_interventions).to_csv(RESULTS_DIR / "sparse_concept_interventions.csv", index=False)
    analyze_and_write(args, started)


if __name__ == "__main__":
    run()
