import os
import cv2
import numpy as np
import pandas as pd

def image_to_thumbnail(img_bgr: np.ndarray, side: int = 32) -> np.ndarray:
    """Converti in scala di grigi in [0, 1] e crea una miniatura side x side."""
    # Converti BGR a RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = img_rgb.astype(np.float32).mean(-1) / 255.0
    k = gray.shape[0] // side
    if k < 1:
        gray = cv2.resize(gray, (side, side))
        return gray.ravel()
    return gray[: k * side, : k * side].reshape(side, k, side, k).mean((1, 3)).ravel()

def green_box_fraction(img_bgr: np.ndarray, corner: float = 0.35) -> float:
    """Quota di pixel verde brillante nell'angolo in basso a sinistra (ScopeGuide)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    patch = img_rgb[int(h * (1 - corner)):, :int(w * corner)].astype(np.int16)
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    green = (g > 90) & (g > r + 40) & (g > b + 40)
    return float(green.mean())

def clarity(img_bgr: np.ndarray) -> dict:
    """Luminosità, saturazione e nitidezza (varianza del Laplaciano) al centro."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = img_rgb.astype(np.float32).mean(-1) / 255.0
    h, w = gray.shape
    centre = gray[h // 6: -h // 6, w // 6: -w // 6]
    lap = (-4 * centre[1:-1, 1:-1] + centre[:-2, 1:-1] + centre[2:, 1:-1]
           + centre[1:-1, :-2] + centre[1:-1, 2:])
    return {
        "brightness": float(centre.mean()),
        "saturated": float((img_rgb.max(-1) >= 250).mean()),
        "sharpness": float(lap.var())
    }

def find_similar_images(target_path: str, dataset_dir: str, match_tol: float = 0.01):
    # Carica immagine target
    target_img = cv2.imread(target_path)
    if target_img is None:
        raise ValueError(f"Impossibile caricare l'immagine: {target_path}")
    
    target_thumb = image_to_thumbnail(target_img)
    
    results = []
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            filepath = os.path.join(dataset_dir, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
            
            thumb = image_to_thumbnail(img)
            # Calcolo della differenza media assoluta tra miniature
            dist = float(np.abs(target_thumb - thumb).mean())
            
            # Parametri di qualità e pulizia
            g_box = green_box_fraction(img)
            clar = clarity(img)
            
            results.append({
                "filename": filename,
                "distance": dist,
                "is_match": dist <= match_tol,
                "has_green_box": g_box > 0.01,
                "sharpness": clar["sharpness"],
                "saturated": clar["saturated"],
                "brightness": clar["brightness"]
            })
            
    df = pd.DataFrame(results).sort_values("distance")
    return df

# Esempio di utilizzo:
if __name__ == "__main__":
    target_image = "/home/a.dorizza/GITHUB/Medical_Robustness_Lipschitz_v5/analysis/image.png"
    dataset_folder = "/home/datasets/medical/kvasir-dataset-v2/normal-pylorus"
    
    df_results = find_similar_images(target_image, dataset_folder)
    print(df_results.head(10).to_string(index=False))