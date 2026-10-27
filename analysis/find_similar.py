
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

QUERY = Path(
    "/home/a.dorizza/GITHUB/Medical_Robustness_Lipschitz_v5/"
    "analysis/image.png"
)

DATASET = Path("/home/datasets/medical/kvasir-dataset-v2")

EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}

SIZE = (256, 256)


def load_image(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def main():
    query = load_image(QUERY)
    query_hash = imagehash.phash(query)

    query_gray = np.asarray(
        query.resize(SIZE).convert("L")
    )

    results = []

    for path in DATASET.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        try:
            candidate = load_image(path)

            # Perceptual hash: distanza minore = più simile
            candidate_hash = imagehash.phash(candidate)
            hash_distance = query_hash - candidate_hash

            # SSIM su immagini ridimensionate
            candidate_gray = np.asarray(
                candidate.resize(SIZE).convert("L")
            )

            score = ssim(
                query_gray,
                candidate_gray,
                data_range=255
            )

            results.append((hash_distance, score, path))

        except Exception as exc:
            print(f"Impossibile leggere {path}: {exc}")

    results.sort(key=lambda x: (x[0], -x[1]))

    print(f"\nImmagine di riferimento: {QUERY}")
    print(f"Immagini analizzate: {len(results)}")
    print("\nTop 20 risultati (pHash crescente, SSIM decrescente):\n")

    for rank, (distance, score, path) in enumerate(results[:20], 1):
        print(
            f"{rank:2d}. pHash={distance:3d} "
            f"SSIM={score:.4f}  {path}"
        )


if __name__ == "__main__":
    main()
