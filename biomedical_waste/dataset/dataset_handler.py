"""Dataset validation and optional 70/20/10 split from a flat image+label folder."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

import yaml

from biomedical_waste.config import CLASS_NAMES


def validate_yolo_dataset_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load and sanity-check a YOLOv8 data.yaml (Roboflow export)."""
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "names" not in data:
        raise ValueError("data.yaml must contain 'names'")
    names = data["names"]
    if isinstance(names, dict):
        vals = [names[k] for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else x)]
    else:
        vals = list(names)
    if set(vals) != set(CLASS_NAMES) or len(vals) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected classes {CLASS_NAMES} (unique), got {vals}"
        )

    base = yaml_path.parent / data.get("path", ".")
    base = base.resolve()

    for key in ("train", "val"):
        rel = data.get(key)
        if not rel:
            raise ValueError(f"data.yaml missing '{key}'")
        split_dir = (base / rel).resolve()
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split folder not found: {split_dir}\n"
                "Place your Roboflow YOLOv8 export under data/dataset/ or update data.yaml."
            )

    test_rel = data.get("test")
    if test_rel:
        test_dir = (base / test_rel).resolve()
        if not test_dir.is_dir():
            raise FileNotFoundError(f"Test folder not found: {test_dir}")

    return data


class DatasetHandler:
    """Utilities for YOLO biomedical waste dataset."""

    def __init__(self, data_yaml: Path | None = None) -> None:
        from biomedical_waste.config import DEFAULT_DATA_YAML

        self.data_yaml = Path(data_yaml or DEFAULT_DATA_YAML).resolve()

    def validate(self) -> dict[str, Any]:
        return validate_yolo_dataset_yaml(self.data_yaml)

    @staticmethod
    def split_into_train_val_test(
        images_dir: Path,
        labels_dir: Path,
        output_root: Path,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Split image/label pairs into train/valid/test (default 70/20/10).
        Expects matching stem names in images_dir and labels_dir (.txt for YOLO).
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        output_root = Path(output_root)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        stems: list[str] = []
        for img in images_dir.iterdir():
            if img.suffix.lower() in exts:
                lab = labels_dir / f"{img.stem}.txt"
                if lab.is_file():
                    stems.append(img.stem)

        if not stems:
            raise FileNotFoundError(
                f"No matching image/label pairs under {images_dir} / {labels_dir}"
            )

        rng = random.Random(seed)
        rng.shuffle(stems)
        n = len(stems)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {
            "train": stems[:n_train],
            "valid": stems[n_train : n_train + n_val],
            "test": stems[n_train + n_val :],
        }

        for split_name, names in splits.items():
            im_out = output_root / split_name / "images"
            lb_out = output_root / split_name / "labels"
            im_out.mkdir(parents=True, exist_ok=True)
            lb_out.mkdir(parents=True, exist_ok=True)
            for stem in names:
                for ext in exts:
                    src = images_dir / f"{stem}{ext}"
                    if src.is_file():
                        shutil.copy2(src, im_out / src.name)
                        break
                shutil.copy2(labels_dir / f"{stem}.txt", lb_out / f"{stem}.txt")

        # Write data.yaml inside output_root (self-contained dataset package)
        data_yaml = output_root / "data.yaml"
        rel_train = "train/images"
        rel_val = "valid/images"
        rel_test = "test/images"
        content = {
            "path": str(output_root.resolve()),
            "train": rel_train,
            "val": rel_val,
            "test": rel_test,
            "names": {i: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))},
        }
        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)
