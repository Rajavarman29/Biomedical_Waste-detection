"""Optional: split flat images+labels into 70/20/10 train/valid/test YOLO layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from biomedical_waste.dataset import DatasetHandler


def main() -> None:
    p = argparse.ArgumentParser(description="Split dataset into train/valid/test (70/20/10)")
    p.add_argument("--images", type=Path, required=True, help="Folder with images")
    p.add_argument("--labels", type=Path, required=True, help="Folder with YOLO .txt labels")
    p.add_argument("--out", type=Path, required=True, help="Output dataset root (creates train/valid/test)")
    args = p.parse_args()

    DatasetHandler.split_into_train_val_test(
        images_dir=args.images,
        labels_dir=args.labels,
        output_root=args.out,
    )
    print(f"Done. Wrote {args.out / 'data.yaml'} — pass this file to training: --data {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
