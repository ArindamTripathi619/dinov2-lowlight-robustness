"""LoRA for DINOv2 attention (and optionally MLP) projections.

Implementation notes that matter for correctness:
- B is initialized to ZERO (Hu et al. 2021), so the adapted model is exactly
  the pretrained model at step 0. The original repo script used randn for both
  factors, which injects a random feature perturbation before training starts.
- Optimizer groups are constructed disjointly (the head must not appear in both
  groups — the original script's duplication made AdamW raise).
- Seeds are set inside train_lora so different (rank, target) configs are
  comparable run-to-run.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:  # allows analysis-only environments to import the module
    TORCH_AVAILABLE = False

from src.corruptions import low_light

if not TORCH_AVAILABLE:
    raise ImportError(
        "src.lora requires torch; install requirements.txt to run LoRA experiments")


class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(r, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora(model: nn.Module, rank: int, alpha: int, target: str = "attn"):
    """target: 'attn' -> qkv+proj; 'attn_mlp' -> qkv+proj+fc1+fc2."""
    for block in model.blocks:
        block.attn.qkv = LoRALayer(block.attn.qkv, r=rank, alpha=alpha)
        block.attn.proj = LoRALayer(block.attn.proj, r=rank, alpha=alpha)
        if target == "attn_mlp":
            block.mlp.fc1 = LoRALayer(block.mlp.fc1, r=rank, alpha=alpha)
            block.mlp.fc2 = LoRALayer(block.mlp.fc2, r=rank, alpha=alpha)
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
    return model


class LowLightAugCIFAR(Dataset):
    """70% of samples randomly degraded to severity 2-4, 50% horizontal flip."""

    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i].copy()
        # Per-sample RNG (seeded by index) instead of the global np.random
        # stream: DataLoader workers fork without re-seeding NumPy, so a
        # global-RNG draw can be correlated across workers and depends on
        # worker scheduling. Deterministic per image, independent across
        # images — same pattern as run_lora_simple_colab.py (the Phase-3 run
        # of record). No committed artifact came from this path (Phase-3
        # results of record came from the Colab runner), so no re-baseline.
        rng = np.random.default_rng(seed=i)
        if rng.random() < 0.7:
            img = low_light(img, int(rng.integers(2, 5)))
        if rng.random() < 0.5:
            img = np.flip(img, axis=1).copy()
        return self.transform(img), self.labels[i]


class BackboneClassifier(nn.Module):
    def __init__(self, backbone, num_classes: int = 10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        if out.dim() == 3:
            out = out[:, 0, :]
        return self.head(out)


class AnchoredBackboneClassifier(nn.Module):
    """Backbone + head + an invariance anchor: a projection head g() whose
    contrastive loss pulls the CLS embeddings of two augs of the SAME image
    together and pushes different images apart (SimCLR-style, Chen et al. 2020).

    This is the constructive counterpart to the negative LoRA result: the
    failure mode identified there is objective conflict (adapters receive only
    class-discrimination gradients). The anchor supplies the missing
    invariance signal directly, on top of the same adapter budget.
    """

    def __init__(self, backbone, num_classes: int = 10, embed_dim: int = None,
                 proj_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)
        d = embed_dim or backbone.embed_dim
        self.projector = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, proj_dim))

    def forward(self, x):
        out = self.backbone(x)
        if out.dim() == 3:
            out = out[:, 0, :]
        return self.head(out), self.projector(out)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE between two augmented views (batch B each)."""
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature                       # (B, B)
    labels = torch.arange(z1.shape[0], device=z1.device)
    return (nn.functional.cross_entropy(logits, labels)
            + nn.functional.cross_entropy(logits.T, labels)) / 2


def train_lora(pretrained_model, train_images, train_labels, preprocess, cfg: dict,
               rank: int, target: str, device: str, epochs: int = None, verbose: bool = True):
    """Train a fresh LoRA adapter set + head. Returns the eval-mode classifier."""
    import time

    rank = int(rank)
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    model = pretrained_model  # reused; apply_lora mutates in place
    apply_lora(model, rank, alpha=cfg.get("alpha_multiplier", 2) * rank, target=target)
    clf = BackboneClassifier(model).to(device)

    head_params = list(clf.head.parameters())
    head_ids = {id(p) for p in head_params}
    lora_params = [p for n, p in clf.named_parameters()
                   if p.requires_grad and id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": cfg.get("lr_lora", 5e-5), "weight_decay": cfg.get("weight_decay", 0.01)},
        {"params": head_params, "lr": cfg.get("lr_head", 1e-3), "weight_decay": cfg.get("weight_decay", 0.01)},
    ])
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(LowLightAugCIFAR(train_images, train_labels, preprocess),
                        batch_size=cfg.get("batch_size", 64), shuffle=True, num_workers=2)

    epochs = epochs or cfg.get("epochs", 10)
    clf.train()
    t0 = time.time()
    for epoch in range(epochs):
        running, correct, total = 0.0, 0, 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = clf(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(lbls).sum().item()
            total += imgs.size(0)
        if verbose:
            print(f"  [r={rank} {target}] epoch {epoch+1}/{epochs}: "
                  f"loss={running/total:.4f} acc={correct/total:.4f} ({time.time()-t0:.0f}s)")
    clf.eval()
    return clf


def train_lora_anchored(pretrained_model, train_images, train_labels, preprocess, cfg: dict,
                        rank: int, target: str, device: str, epochs: int = None,
                        verbose: bool = True):
    """LoRA + invariance anchor (InfoNCE over two augmented views, weight
    lambda). Same adapter budget as train_lora; only the OBJECTIVE differs —
    the controlled comparison the paper's Discussion promises.
    """
    rank = int(rank)
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    model = pretrained_model
    apply_lora(model, rank, alpha=cfg.get("alpha_multiplier", 2) * rank, target=target)
    clf = AnchoredBackboneClassifier(model).to(device)

    head_params = list(clf.head.parameters()) + list(clf.projector.parameters())
    head_ids = {id(p) for p in head_params}
    lora_params = [p for n, p in clf.named_parameters()
                   if p.requires_grad and id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": cfg.get("lr_lora", 5e-5), "weight_decay": cfg.get("weight_decay", 0.01)},
        {"params": head_params, "lr": cfg.get("lr_head", 1e-3), "weight_decay": cfg.get("weight_decay", 0.01)},
    ])
    criterion = nn.CrossEntropyLoss()
    lam = cfg.get("anchored", {}).get("lambda_anchor", 1.0)

    loader = DataLoader(LowLightAugCIFAR(train_images, train_labels, preprocess),
                        batch_size=cfg.get("batch_size", 64), shuffle=True, num_workers=2)
    epochs = epochs or cfg.get("epochs", 10)
    clf.train()
    for epoch in range(epochs):
        running, correct, total = 0.0, 0, 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits, proj = clf(imgs)
            ce = criterion(logits, lbls)
            # second view: horizontal flip of the same (already randomly
            # corrupted by the dataset) images — a cheap invariance pair
            imgs2 = torch.flip(imgs, dims=[3])
            feats2 = clf.backbone(imgs2)
            if feats2.dim() == 3:
                feats2 = feats2[:, 0, :]
            proj2 = clf.projector(feats2)
            loss = ce + lam * info_nce_loss(proj, proj2)
            loss.backward()
            optimizer.step()
            running += ce.item() * imgs.size(0)
            correct += logits.argmax(1).eq(lbls).sum().item()
            total += imgs.size(0)
        if verbose:
            print(f"  [r={rank} {target}+anchor] epoch {epoch+1}/{epochs}: "
                  f"ce={running/total:.4f} acc={correct/total:.4f}")
    clf.eval()
    return clf


@torch.no_grad()
def extract_backbone(clf, image_list, preprocess, device, batch_size: int = 64):
    import torch as _torch

    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        out = clf.backbone(tensors)
        if out.dim() == 3:
            out = out[:, 0, :]
        feats.append(out.float().cpu())
    return _torch.cat(feats, dim=0).numpy()
