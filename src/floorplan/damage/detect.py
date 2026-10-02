"""Zero-shot damage detection on keyframes (OWL-ViT, CPU). Boxes only: there is no segmentation model, so the extent of a damage patch is
the footprint of its detection box (an upper bound), and accuracy is untested because no labelled damage exists."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    frame: int
    cls: str
    prompt: str
    score: float
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 in RGB pixels (1920 x 1440)


class Detector:
    def __init__(self, cfg: dict):
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.cfg, self.torch = cfg["damage"], torch
        self.proc = OwlViTProcessor.from_pretrained(self.cfg["model"], revision=self.cfg["revision"])
        self.model = OwlViTForObjectDetection.from_pretrained(self.cfg["model"], revision=self.cfg["revision"]).eval()
        self.prompts = [(c, p) for c, ps in self.cfg["classes"].items() for p in ps]

    def detect(self, bgr: np.ndarray, frame: int) -> list[Detection]:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.proc(text=[[p for _, p in self.prompts]], images=rgb, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_grounded_object_detection(out, threshold=self.cfg["score_min"], target_sizes=self.torch.tensor([[h, w]]))[0]
        dets = []
        for s, lab, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            x0, y0, x1, y1 = max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h)
            if (x1 - x0) * (y1 - y0) > self.cfg["max_box_fraction"] * w * h or x1 <= x0 or y1 <= y0:
                continue
            cls, prompt = self.prompts[lab]
            dets.append(Detection(frame, cls, prompt, round(s, 4), (x0, y0, x1, y1)))
        return nms(dets)


def iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets: list[Detection], thr: float = 0.5) -> list[Detection]:
    """Per frame and class: keep the best-scoring box among boxes that overlap by IoU >= thr (prompts of one class repeat the same patch)."""
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if not any(k.cls == d.cls and iou(k.box, d.box) >= thr for k in keep):
            keep.append(d)
    return keep
