import argparse
import json
import math
import os

import torch

from model import TinyTransformer
from data import get_batch


def lr_warm(step, warm_steps=20):
    return min(1.0, step / warm_steps)


class GradientRelay:
    def __init__(self, model, optimizer, scaler, span_count):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.span_count = span_count
        self._count = 0
        self.commit_trace = []

    def intake(self, loss, step):
        self.scaler.scale(loss).backward()
        self._count += 1
        offset = (self._count - 1) % self.span_count
        clip_cap = 1.0 + 0.1 * math.log2(self.span_count + 1)
        if offset == self.span_count - 1:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_cap)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.commit_trace.append([step, self._count])
            return float(grad_norm), self.scaler.get_scale()
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--span_count", type=int, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--chunk_batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    model = TinyTransformer()
    base_lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True)
    relay = GradientRelay(model, optimizer, scaler, args.span_count)

    checkpoints = {50, 100, 200, 400}
    loss_curve = {}
    grad_curve = {}
    scale_curve = {}

    for step in range(1, args.steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = base_lr * lr_warm(step)

        step_loss = 0.0
        grad_norm, scale_val = None, None
        for chunk_idx in range(args.span_count):
            x, y = get_batch(step, chunk_idx, batch_size=args.chunk_batch)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                loss = model(x, y) / args.span_count
            result = relay.intake(loss, step)
            step_loss += loss.item()
            if result is not None:
                grad_norm, scale_val = result

        if step in checkpoints:
            loss_curve[str(step)] = step_loss
            grad_curve[str(step)] = grad_norm
            scale_curve[str(step)] = scale_val

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "span_count": args.span_count,
                "loss_curve": loss_curve,
                "grad_curve": grad_curve,
                "scale_curve": scale_curve,
                "commit_trace": relay.commit_trace,
            },
            f,
        )


if __name__ == "__main__":
    main()