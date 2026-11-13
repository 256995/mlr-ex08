"""Task 4 - make the leaf classifier better on a phone camera, one change at a time.

The notebook's rule for Task 4: fix a baseline, change **one** thing per variant, train
every variant on the same pictures with a fixed seed, repeat the important rows with
several seeds, and measure not only the validation accuracy but also a *camera-like*
validation set (requirement 2: the validation images randomly cropped, rotated and with
changed colours, generated once with a fixed seed).

    uv run train_leaves.py --variant baseline --seed 0        # one run, ~3 minutes
    uv run train_leaves.py --variant colour --seeds 3         # seeds 0, 1, 2
    uv run train_leaves.py --variant combined --seed 0 --export
    uv run train_leaves.py --summarise                        # the table from runs/*.json

Everything is written into ``_dev/runs/<variant>-seed<k>.json`` so that runs can go in
parallel and nothing overwrites anything.

The baseline is the recipe of Task 2 - the one the phone page runs today:
pretrained MobileNetV3-Small, its last layer replaced, the whole network fine-tuned
with Adam (lr 1e-3) for 15 epochs, batches of 32, plain ToTensor + Normalize.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = 96
BATCH = 32

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "_dev" / "leaves_data" / "leaves"  # last year's photos: 175 train + 30 val per class
RUNS = HERE.parent / "_dev" / "runs"

# The augmentation of Task 3 (geometric) and the colour one Task 4 suggests on top of it.
GEOMETRY = [
    transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
]
COLOUR = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

# The baseline recipe, then every variant as "baseline plus these settings".
BASE = dict(
    network="mobilenet",
    augment=False,
    colour=False,
    frozen=False,
    optimizer="adam",
    lr=1e-3,
    weight_decay=0.0,
    epochs=15,
    scheduler="",
    label_smoothing=0.0,
)
VARIANTS: dict[str, dict] = {
    "baseline": {},  # what the phone runs today
    "scratch": dict(network="leafnet", augment=True, epochs=30),  # Task 3 network, from scratch
    "augment": dict(augment=True),  # + geometric augmentation
    "colour": dict(augment=True, colour=True),  # … and colour jitter on top
    "frozen": dict(frozen=True),  # train only the new last layer
    "cosine": dict(scheduler="cosine", epochs=30),  # longer, learning rate decaying
    "adamw": dict(optimizer="adamw", weight_decay=1e-4),
    "smooth": dict(label_smoothing=0.05),
    # the combinations, assembled only from the single changes that measured well
    "combined": dict(augment=True, colour=True, scheduler="cosine", epochs=30),
    "augment_cosine": dict(augment=True, scheduler="cosine", epochs=30),  # combined, minus colour jitter
    "full": dict(augment=True, colour=True, scheduler="cosine", epochs=30, label_smoothing=0.05),
}


def transform(augment: bool = False, colour: bool = False) -> transforms.Compose:
    """The picture -> tensor pipeline: Task 2's two steps, optionally preceded by augmentation."""
    steps = list(GEOMETRY) if augment else []
    if colour:
        steps.append(COLOUR)
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


class LeafNet(nn.Module):
    """The Task 3 network (three conv blocks + two linear layers), kept for the "from scratch" row."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(x)
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))


def build_network(variant: dict) -> nn.Module:
    """The network of the variant; for MobileNetV3 optionally with everything but the last layer frozen."""
    if variant["network"] == "leafnet":
        return LeafNet()
    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    if variant["frozen"]:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.classifier[3].parameters():
            parameter.requires_grad_(True)
    return model


def camera_like_set(val_dir: Path, out_dir: Path, seed: int = 0) -> Path:
    """Build the camera-like validation set once and reuse it.

    One fixed "what a phone camera really hands the model" version of every validation
    picture: a random crop of 50-100 % of it, rotated by up to 25 degrees, colours
    changed, resized back to 96x96. Fixed seed, so every variant - and every later rerun -
    is measured on exactly the same pictures. No model is ever chosen using this set.
    """
    if out_dir.exists():
        return out_dir
    out_dir.mkdir(parents=True)
    for class_folder in sorted(val_dir.iterdir()):
        (out_dir / class_folder.name).mkdir()
        for index, image_file in enumerate(sorted(class_folder.iterdir())):
            torch.manual_seed(seed * 100_003 + index)  # the same picture every run, whatever else changes
            picture = Image.open(image_file).convert("RGB")
            picture = transforms.RandomResizedCrop(SIZE, scale=(0.5, 1.0))(picture)
            picture = transforms.RandomRotation(25)(picture)
            picture = transforms.ColorJitter(0.4, 0.4, 0.4, 0.15)(picture)
            picture.save(out_dir / class_folder.name / image_file.name, quality=92)
    return out_dir


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, flip_averaging: bool = False) -> float:
    """Fraction of correct top-1 predictions; flip_averaging = average the picture and its mirror."""
    model.eval()
    correct = 0
    for images, labels in loader:
        probabilities = model(images).softmax(dim=1)
        if flip_averaging:
            probabilities = (probabilities + model(torch.flip(images, dims=(3,))).softmax(dim=1)) / 2
        correct += (probabilities.argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


@torch.no_grad()
def validation_loss(model: nn.Module, loader: DataLoader) -> float:
    """Mean cross entropy on the validation set - what decides the best epoch."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = sum(criterion(model(images), labels).item() * len(labels) for images, labels in loader)
    return total / len(loader.dataset)


def train(variant_name: str, seed: int, epochs: int | None = None, quiet: bool = False) -> dict:
    """Train one variant with one seed and return everything that was measured."""
    variant = BASE | VARIANTS[variant_name]
    if epochs:
        variant["epochs"] = epochs  # only for a quick smoke test, never for a reported row
    torch.manual_seed(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        datasets.ImageFolder(DATA / "train", transform=transform(variant["augment"], variant["colour"])),
        batch_size=BATCH,
        shuffle=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(datasets.ImageFolder(DATA / "val", transform=transform()), batch_size=BATCH)
    camera_loader = DataLoader(
        datasets.ImageFolder(
            camera_like_set(DATA / "val", DATA.parent / "leaves_cam"), transform=transform()
        ),
        batch_size=BATCH,
    )

    model = build_network(variant)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if variant["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=variant["lr"], weight_decay=variant["weight_decay"])
    else:
        optimizer = torch.optim.Adam(trainable, lr=variant["lr"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=variant["epochs"])
        if variant["scheduler"] == "cosine"
        else None
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=variant["label_smoothing"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, criterion = model.to(device), criterion.to(device)

    best = {"loss": float("inf"), "epoch": 0, "state": None}
    started = time.time()
    for epoch in range(variant["epochs"]):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        if scheduler:
            scheduler.step()

        val = accuracy(model, val_loader)
        camera = accuracy(model, camera_loader)
        val_loss = validation_loss(model, val_loader)
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "epoch": epoch + 1, "state": copy.deepcopy(model.state_dict())}
        if not quiet:
            print(
                f"epoch {epoch + 1:2d}  train loss {loss_sum / len(train_loader.dataset):.4f}"
                f"  val {val:.3f}  camera-like {camera:.3f}",
                flush=True,
            )

    # Report both the last and the best epoch. "Best epoch" is chosen on the validation set,
    # so the honest column is always the camera-like accuracy of the LAST epoch.
    last = {
        "val": accuracy(model, val_loader),
        "camera": accuracy(model, camera_loader),
        "camera_tta": accuracy(model, camera_loader, flip_averaging=True),
    }
    last_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best["state"])
    best_of = {
        "val": accuracy(model, val_loader),
        "camera": accuracy(model, camera_loader),
        "camera_tta": accuracy(model, camera_loader, flip_averaging=True),
    }

    result = {
        "variant": variant_name,
        "seed": seed,
        "settings": {k: v for k, v in variant.items()},
        "trainable_parameters": sum(p.numel() for p in trainable),
        "epochs": variant["epochs"],
        "seconds": round(time.time() - started, 1),
        "last": last,
        "best": best_of,
        "best_epoch": best["epoch"],
    }
    if torch.cuda.is_available():
        result["device"] = "cuda"
    RUNS.mkdir(parents=True, exist_ok=True)
    # Every seed keeps its own file; a smoke test (--epochs) is marked in the name too, so it
    # can never end up in the table.
    name = f"{variant_name}-seed{seed}" + (f"-SMOKE-{epochs}ep" if epochs else "")
    (RUNS / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    torch.save(last_state, RUNS / f"{name}.pt")  # weights of the last epoch
    torch.save(best["state"], RUNS / f"{name}-best.pt")  # epoch with the lowest validation loss
    return result


def export(variant_name: str, seed: int, weights: str = "last") -> Path:
    """Export the trained model the way 2e does, plus the four files the page needs."""
    variant = BASE | VARIANTS[variant_name]
    model = build_network(variant)
    model.load_state_dict(
        torch.load(
            RUNS / f"{variant_name}-seed{seed}{'' if weights == 'last' else '-best'}.pt", map_location="cpu"
        )
    )
    model.eval()

    import onnx  # only the export needs onnx, so it is imported here

    onnx_path = HERE / "leaf_model_v2.onnx"
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, SIZE, SIZE),),
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        external_data=False,
    )
    model_info = {
        # the same class -> index order the training used, taken from the folder itself
        "classes": datasets.ImageFolder(DATA / "train").classes,
        "input_size": SIZE,
        "mean": MEAN,
        "std": STD,
    }
    onnx_model = onnx.load(onnx_path)
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
    onnx.save(onnx_model, onnx_path)

    # A check picture the whole comparison is worth something on: the first camera-like
    # image whose probabilities are NOT one-vs-three (a 0.99998 answer checks nothing).
    plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    chosen: tuple[Path, dict] | None = None
    for image_file in sorted((DATA.parent / "leaves_cam").rglob("*.jpg")):
        with torch.no_grad():
            probabilities = model(plain(Image.open(image_file).convert("RGB")).unsqueeze(0)).softmax(1)[0]
        if 0.5 <= probabilities.max().item() < 0.95:
            chosen = (image_file, dict(zip(model_info["classes"], probabilities.tolist(), strict=True)))
            break
    if chosen is None:
        raise SystemExit("no check image with a non-decisive answer was found")
    image_file, expected = chosen
    shutil.copy(image_file, HERE / "check_image.jpg")
    (HERE / "check_expected.json").write_text(json.dumps(expected, indent=1), encoding="utf-8")
    (HERE / "model_info.json").write_text(json.dumps(model_info), encoding="utf-8")
    print(
        f"exported {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB); "
        f"check image {image_file.name}: {expected}"
    )
    return onnx_path


def summarise() -> None:
    """Print the table of every run recorded so far, one row per variant, seeds averaged."""
    # Only a properly named run counts: one file per variant and seed, never a smoke test.
    paths = [path for path in sorted(RUNS.glob("*-seed*.json")) if "SMOKE" not in path.name]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not rows:
        raise SystemExit(f"no runs recorded in {RUNS} yet")

    def cell(values: list[float]) -> str:
        """Mean of the seeds, with +- half the spread when there is more than one of them."""
        average = statistics.mean(values)
        if len(values) < 2:
            return f"{average:.3f}      "
        return f"{average:.3f} +-{(max(values) - min(values)) / 2:.3f}"

    print(
        f"{'variant':14s} {'runs':>4s}  {'validation':>14s}  {'camera-like':>14s}  "
        f"{'cam + flip':>10s}  {'s/epoch':>7s}"
    )
    for variant_name in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant_name]
        print(
            f"{variant_name:14s} {len(group):>4d}  "
            f"{cell([row['last']['val'] for row in group]):>14s}  "
            f"{cell([row['last']['camera'] for row in group]):>14s}  "
            f"{statistics.mean([row['last']['camera_tta'] for row in group]):>10.3f}  "
            f"{statistics.mean(row['seconds'] / row['epochs'] for row in group):>7.1f}"
        )
    print("\nvalidation  = the 120 fixed validation photos, last epoch.")
    print("camera-like = the same 120 photos cropped, rotated and recoloured once with seed 0, last epoch.")
    print(
        "cam + flip  = camera-like with the mirrored picture averaged in, i.e. what the page can add for free."
    )
    print("One validation photo is worth 1/120 = 0.008, so differences of ~0.02 or less are noise.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--seed", type=int, default=0, help="the one seed to train (or to export)")
    parser.add_argument("--seeds", type=int, default=1, help="run seeds 0..seeds-1 instead of one seed")
    parser.add_argument("--epochs", type=int, help="override the epochs of the variant (smoke tests only)")
    parser.add_argument("--export", action="store_true", help="write the ONNX + page files of this run")
    parser.add_argument(
        "--weights",
        choices=["last", "best"],
        default="last",
        help="for --export: last epoch (what the camera-like column measures) or best epoch on val",
    )
    parser.add_argument("--summarise", action="store_true", help="print the table of all recorded runs")
    arguments = parser.parse_args()

    if arguments.summarise:
        summarise()
        return
    if not arguments.variant:
        parser.error("--variant is needed (or --summarise)")
    if arguments.export:  # exports a run that was trained before: --variant X --seed 0
        export(arguments.variant, arguments.seed, arguments.weights)
        return
    # --seeds N runs seeds 0..N-1; alone, --seed K runs exactly that one seed.
    seeds = list(range(arguments.seeds)) if arguments.seeds > 1 else [arguments.seed]
    for seed in seeds:
        result = train(arguments.variant, seed, epochs=arguments.epochs)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("variant", "seed", "epochs", "last", "best", "best_epoch", "seconds")
                }
            )
        )


if __name__ == "__main__":
    main()
