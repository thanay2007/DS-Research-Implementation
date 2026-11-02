"""
Train one site subset on one fold and save its out-of-fold predictions.

Run from the repo root after preprocess_images.py:
  python train.py --subset C+N --pipeline shared --fold 0 --images C:/tmp/ds-research/images

Model and training follow the base paper's Appendices 2, 3 and 5: a ResNet of
2 stages with one 3-convolution block each, trained from scratch, its features
concatenated with age, sex and pregnancy status into an MLP that predicts Hb.
Adam with exponential decay, batch 32, 150 epochs, Huber loss with inverse-bin
weights, and one constant bias correction chosen on validation.

Differs from the paper's stated lr of 0.013 (Appendix 5; the main text says
0.02 and batch 64): with our model it never settled, so lr is 0.0001, chosen
on validation.

Not stated in the paper, chosen here and listed as assumptions in the README:
filter counts, stem, MLP widths, Huber delta 1.0, 1 g/dL bins, square-root
weights, decay per epoch, horizontal flips, keeping the epoch with the best
validation MAE.

For outer test fold k, fold k+1 is validation and the other three are training.
The field group is predicted by every model but never trained or selected on.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

SITE_VIEWS = {"C": ["conjunctiva"], "T": ["tongue"], "N": ["nails_open", "nails_closed"]}
SUBSETS = ["C", "T", "N", "C+T", "C+N", "T+N", "C+T+N"]
SEXES = ["Female", "Male", "Other"]


class Block(nn.Module):
    """Appendix 3: three convolution / batch norm stages with a residual sum."""

    def __init__(self, channels_in, width, stride):
        super().__init__()
        out = width * 4
        self.body = nn.Sequential(
            nn.Conv2d(channels_in, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, out, 1, bias=False), nn.BatchNorm2d(out),
        )
        self.skip = nn.Sequential(nn.Conv2d(channels_in, out, 1, stride=stride, bias=False), nn.BatchNorm2d(out))

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class Encoder(nn.Module):
    features = 512

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            Block(64, 64, 1), Block(256, 128, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class HbModel(nn.Module):
    """One encoder per site. The two nail views share the nail encoder."""

    def __init__(self, subset, metadata_size):
        super().__init__()
        self.sites = subset.split("+")
        self.encoders = nn.ModuleDict({site: Encoder() for site in self.sites})
        views = sum(len(SITE_VIEWS[site]) for site in self.sites)
        self.head = nn.Sequential(
            nn.Linear(views * Encoder.features + metadata_size, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, images, metadata):
        parts = [self.encoders[site](images[view]) for site in self.sites for view in SITE_VIEWS[site]]
        return self.head(torch.cat(parts + [metadata], dim=1)).squeeze(1)


def inverse_bin_weights(hb, train_hb, power):
    """Rare Hb values weigh more. Bins are 1 g/dL, weights average 1 over training.

    power 1 is plain inverse frequency, which lets one patient in a near-empty
    bin swamp a batch. power 0.5 is the square-root form from deep imbalanced
    regression (Yang et al. 2021), the method the base paper cites.
    """
    edges = np.arange(0, 26)
    counts = np.histogram(train_hb, edges)[0].astype(np.float64)
    weight = np.maximum(counts, 1) ** -power
    weight /= (weight[np.digitize(train_hb, edges) - 1]).mean()
    return weight[np.digitize(hb, edges) - 1]


def metadata_matrix(df):
    sex = np.stack([(df.sex == s).to_numpy(np.float32) for s in SEXES], axis=1)
    return np.column_stack([df.age.to_numpy(np.float32) / 100, sex, df.pregnant.to_numpy(np.float32)]).astype(np.float32)


class Batches:
    def __init__(self, arrays, df, device):
        self.arrays, self.device = arrays, device
        self.rows = df.index.to_numpy()
        self.metadata = metadata_matrix(df)
        self.hb = df.hb.to_numpy(np.float32)

    def images(self, positions, flip):
        rows = self.rows[positions]
        # Memmaps read fastest in ascending order.
        order = np.argsort(rows)
        batch = {}
        for view, array in self.arrays.items():
            data = np.empty((len(rows), 256, 256, 3), dtype=np.uint8)
            data[order] = array[rows[order]]
            # contiguous() matters: cuDNN is about 18x slower on permuted memory.
            tensor = torch.from_numpy(data).to(self.device, non_blocking=True).permute(0, 3, 1, 2).contiguous().float() / 127.5 - 1
            if flip:
                tensor = torch.where(torch.rand(len(rows), 1, 1, 1, device=self.device) < 0.5, tensor.flip(3), tensor)
            batch[view] = tensor
        return batch


@torch.no_grad()
def predict(model, data, batch_size, amp):
    model.eval()
    out = []
    for start in range(0, len(data.rows), batch_size):
        positions = np.arange(start, min(start + batch_size, len(data.rows)))
        metadata = torch.from_numpy(data.metadata[positions]).to(data.device)
        with torch.autocast(data.device.type, enabled=amp):
            out.append(model(data.images(positions, False), metadata).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=SUBSETS, required=True)
    ap.add_argument("--pipeline", choices=["shared", "specific"], default="shared")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--cohort", default="data/derived/cohort.csv")
    ap.add_argument("--out", default="data/runs")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--decay", type=float, default=0.97)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--reweight-power", type=float, default=0.5, help="0 turns reweighting off")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    views = [view for site in args.subset.split("+") for view in SITE_VIEWS[site]]
    arrays = {view: np.load(Path(args.images) / f"{args.pipeline}_{view}.npy", mmap_mode="r") for view in views}

    cohort = pd.read_csv(args.cohort)
    usable = cohort[cohort.exclude_reason.isna()]
    has = {"conjunctiva": "has_conjunctiva", "tongue": "has_tongue", "nails_open": "has_fingernails_open", "nails_closed": "has_fingernails_closed"}
    usable = usable[usable[[has[view] for view in views]].all(axis=1)]
    valid_fold = (args.fold + 1) % 5
    parts = {
        "train": usable[(usable.fold >= 0) & (usable.fold != args.fold) & (usable.fold != valid_fold)],
        "valid": usable[usable.fold == valid_fold],
        "test": usable[usable.fold == args.fold],
        "field": usable[usable.group == "field"],
    }
    assert not set(parts["train"].participant_id) & set(parts["test"].participant_id)
    data = {name: Batches(arrays, part, device) for name, part in parts.items()}

    train = data["train"]
    weights = torch.from_numpy(inverse_bin_weights(train.hb, train.hb, args.reweight_power).astype(np.float32)).to(device)
    center = float(train.hb.mean())
    model = HbModel(args.subset, train.metadata.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = nn.HuberLoss(delta=args.huber_delta, reduction="none")
    targets = torch.from_numpy(train.hb).to(device)

    run = Path(args.out) / args.pipeline / args.subset / f"fold{args.fold}"
    run.mkdir(parents=True, exist_ok=True)
    best_mae, best_epoch, started = float("inf"), -1, time.time()
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train.rows))
        train_errors = []
        for start in range(0, len(order) - args.batch_size + 1, args.batch_size):
            positions = order[start:start + args.batch_size]
            index = torch.from_numpy(positions).to(device)
            metadata = torch.from_numpy(train.metadata[positions]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                prediction = model(train.images(positions, True), metadata) + center
                loss = (loss_fn(prediction, targets[index]) * weights[index]).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_errors.append((prediction.detach() - targets[index]).abs().mean().item())
        schedule.step()

        valid_mae = float(np.abs(predict(model, data["valid"], 64, amp) + center - data["valid"].hb).mean())
        if valid_mae < best_mae:
            best_mae, best_epoch = valid_mae, epoch
            torch.save(model.state_dict(), run / "best.pt")
        print(f"epoch {epoch + 1}/{args.epochs} train MAE {np.mean(train_errors):.3f} valid MAE {valid_mae:.3f} best {best_mae:.3f} "
              f"({(time.time() - started) / (epoch + 1):.1f} s/epoch)", flush=True)

    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    bias = float((data["valid"].hb - (predict(model, data["valid"], 64, amp) + center)).mean())
    frames = []
    for name in ("valid", "test", "field"):
        frame = parts[name][["patient_uuid", "participant_id", "group", "fold", "hb"]].copy()
        frame["split"] = name
        frame["pred"] = predict(model, data[name], 64, amp) + center + bias
        frames.append(frame)
    predictions = pd.concat(frames)
    predictions.to_csv(run / "predictions.csv", index=False)
    test = predictions[predictions.split == "test"]
    (run / "summary.json").write_text(json.dumps({
        **vars(args), "best_epoch": best_epoch, "valid_mae": best_mae, "bias": bias,
        "test_mae": float((test.pred - test.hb).abs().mean()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.time() - started,
    }, indent=2))
    print(f"Done. test MAE {(test.pred - test.hb).abs().mean():.3f}, saved to {run}")


if __name__ == "__main__":
    main()
