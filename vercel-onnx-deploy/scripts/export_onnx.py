"""Export the trained PyTorch model to ONNX for lightweight (torch-free) inference.

Usage:  python scripts/export_onnx.py
Outputs (in ./model): breadfruit.onnx, heads.npz, meta.json

The ONNX graph returns the last conv feature map (features.8, the Grad-CAM
target layer) plus both heads' logits. Because the classifier heads are a
single Linear layer after global average pooling, Grad-CAM for class c equals
ReLU(sum_k W[c,k] * A_k) (the gradient w.r.t. every spatial position is
W[c]/49), so the heatmap can be computed in numpy with no autograd.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "breadfruit_classifier"))
from config import Config  # noqa: E402
from model import BreadfruitClassifier  # noqa: E402


class Export(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        act = self.m.backbone.features(x)
        feat = torch.flatten(self.m.backbone.avgpool(act), 1)
        return act, self.m.maturity_head(feat), self.m.quality_head(feat)


def main():
    cfg = Config()
    cfg.model.pretrained = False  # weights come from the checkpoint
    model = BreadfruitClassifier(cfg)
    ckpt = torch.load(ROOT / "breadfruit_model" / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out = ROOT / "model"
    out.mkdir(exist_ok=True)
    dummy = torch.randn(1, 3, *cfg.data.image_size)
    torch.onnx.export(
        Export(model).eval(), dummy, str(out / "breadfruit.onnx"),
        input_names=["input"], output_names=["act", "maturity_logits", "quality_logits"],
        dynamic_axes={"input": {0: "batch"}}, opset_version=17, dynamo=False,
    )
    np.savez(
        out / "heads.npz",
        maturity_w=model.maturity_head.classifier[1].weight.detach().numpy(),
        quality_w=model.quality_head.classifier[1].weight.detach().numpy(),
    )
    (out / "meta.json").write_text(json.dumps({
        "image_size": list(cfg.data.image_size),
        "mean": list(cfg.data.mean),
        "std": list(cfg.data.std),
        "maturity_classes": cfg.model.maturity_classes,
        "quality_classes": cfg.model.quality_classes,
    }, indent=2))
    print("exported to", out)


if __name__ == "__main__":
    main()
