#!/usr/bin/env python3
"""
OCR des bandeaux incrustés dans des vidéos.

Parcourt récursivement un dossier, extrait une image toutes les N frames de
chaque fichier .mp4, lit par OCR (EasyOCR) les champs du bandeau incrusté en
bas de l'image, puis exporte tous les résultats dans un fichier Excel unique.

Dépendances :
    pip install easyocr opencv-python pandas openpyxl

Usage :
    python ocr_videos_folder.py /chemin/vers/videos
    python ocr_videos_folder.py /chemin/vers/videos -o ./sorties -i 1500 --gpu

Les zones à lire (SCANNING_AREAS) sont définies en pixels pour une vidéo
1920x1080 : adaptez-les à vos propres vidéos.
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
import pandas as pd

# --- Configuration ---

OCR_LANGUAGES = ["fr"]
DEFAULT_INTERVAL_FRAMES = 750  # à 25 images/s : 1500 = une image par minute
BINARY_THRESHOLD = 120
UPSCALE_FACTOR = 2
MAX_FRAMES_WITHOUT_BANNER = 4  # arrête le scan d'une vidéo après N images sans bandeau
KEY_FIELDS = ["DEVICE_ID", "DATE", "HEURE", "TC_PRINTED", "GPS_LONG"]

# Zones du bandeau, en pixels : (gauche, droite), entre BANNER_TOP et BANNER_BOTTOM.
# Le bandeau existe en deux variantes : quand le champ GPS_LAT commence par
# "PR", les champs suivants sont décalés et on utilise les zones PR_*.
BANNER_TOP, BANNER_BOTTOM = 1015, 1054
SCANNING_AREAS = {
    "DEVICE_ID":      (19, 165),
    "DATE":           (183, 415),
    "HEURE":          (436, 623),
    "TC_PRINTED":     (733, 900),
    "GPS_LAT":        (918, 1153),
    "GPS_LONG":       (1171, 1383),
    "OFFICER_ID":     (1400, 1566),
    "PR_GPS_LAT":     (984, 1226),
    "PR_GPS_LONG":    (1237, 1459),
    "PR_OFFICER_ID":  (1459, 1639),
}

warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)


# --- Fonctions utilitaires ---

def ensure_last_n(s):
    """Ajoute le 'N' final d'une latitude si l'OCR l'a perdu."""
    if s and not s.endswith("N"):
        s += "N"
    return s


def clean_time(s):
    """Corrige les confusions OCR courantes dans une heure (hh:mm:ss)."""
    s = s.replace("l", "1")
    for c in ",.;":
        s = s.replace(c, ":")
    return s.replace("::", ":")


def format_timecode(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# --- OCR ---

def scan_area(reader, img, area_name):
    """Lit le texte d'une zone de l'image (en niveaux de gris)."""
    left, right = SCANNING_AREAS[area_name]
    crop = img[BANNER_TOP:BANNER_BOTTOM, left:right]

    crop = cv2.resize(crop, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
                      interpolation=cv2.INTER_CUBIC)
    _, crop = cv2.threshold(crop, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    results = reader.readtext(crop, detail=0, paragraph=True)
    txt = "".join(results)
    for c in '\n "%*':
        txt = txt.replace(c, "")
    return txt


def scan_frame(reader, img, timestamp_sec, video_path, id_video):
    """Lit tous les champs du bandeau d'une image."""
    gps_lat = scan_area(reader, img, "GPS_LAT")
    prefix = "PR_" if gps_lat.startswith("PR") else ""
    if prefix:
        gps_lat = scan_area(reader, img, "PR_GPS_LAT")

    return {
        "ID_VIDEO": id_video,
        "DEVICE_ID": scan_area(reader, img, "DEVICE_ID"),
        "DATE": scan_area(reader, img, "DATE"),
        "HEURE": clean_time(scan_area(reader, img, "HEURE")),
        "TC_PRINTED": scan_area(reader, img, "TC_PRINTED"),
        "TC_IN_VIDEO": format_timecode(timestamp_sec),
        "GPS_LAT": ensure_last_n(gps_lat),
        "GPS_LONG": scan_area(reader, img, prefix + "GPS_LONG"),
        "OFFICER_ID": scan_area(reader, img, prefix + "OFFICER_ID"),
        "FILENAME": video_path.stem,
        "URL_FILE": str(video_path),
    }


def scan_video(reader, video_path, interval_frames, id_video):
    """Scanne une vidéo et renvoie la liste des lignes valides."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError("impossible d'ouvrir la vidéo")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        cap.release()
        raise OSError("FPS illisible")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []
    frames_without_banner = 0
    try:
        for frame_number in range(0, total_frames, interval_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            row = scan_frame(reader, gray, frame_number / fps, video_path, id_video)

            # Pas de DEVICE_ID lisible : probablement pas de bandeau
            if len(row["DEVICE_ID"]) < 2:
                frames_without_banner += 1
                if frames_without_banner >= MAX_FRAMES_WITHOUT_BANNER:
                    print(f"     ⚠️ {MAX_FRAMES_WITHOUT_BANNER} images sans bandeau. Stop scanning.")
                    break
            else:
                frames_without_banner = 0

            if all(len(row[f]) >= 2 for f in KEY_FIELDS):
                rows.append(row)
    finally:
        cap.release()

    return rows


def scan_all_videos(reader, root_folder, interval_frames):
    """Scanne tous les .mp4 du dossier et de ses sous-dossiers."""
    start = datetime.now()
    mp4_files = sorted(Path(root_folder).rglob("*.mp4"))
    print(f"{len(mp4_files)} fichiers .mp4 trouvés dans {root_folder}\n")

    all_rows, failed = [], []
    for i, video_path in enumerate(mp4_files, start=1):
        elapsed = format_timecode((datetime.now() - start).total_seconds())
        print(f"\n{elapsed} >>> Vidéo {i}/{len(mp4_files)}: {video_path.name}")
        try:
            all_rows.extend(scan_video(reader, video_path, interval_frames, i))
        except Exception as e:
            print(f"❌ ERREUR sur la vidéo : {video_path}\n   {e}")
            failed.append(video_path)

    return all_rows, failed


def export_excel(rows, output_dir):
    if not rows:
        print("Aucune donnée à exporter.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / f"SCAN_VIDEOS_{datetime.now():%Y%m%d-%H%M%S}.xlsx"
    pd.DataFrame(rows).astype(str).to_excel(excel_path, index=False)
    print(f"\nFichier Excel final exporté : {excel_path}")


# --- Exécution ---

def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("folder", type=Path, help="dossier contenant les vidéos (.mp4)")
    parser.add_argument("-o", "--output", type=Path, default=Path("."),
                        help="dossier de sortie du fichier Excel (défaut : dossier courant)")
    parser.add_argument("-i", "--interval", type=int, default=DEFAULT_INTERVAL_FRAMES,
                        help=f"nombre de frames entre deux captures (défaut : {DEFAULT_INTERVAL_FRAMES})")
    parser.add_argument("--gpu", action="store_true", help="utiliser le GPU (CUDA) pour l'OCR")
    args = parser.parse_args()

    reader = easyocr.Reader(OCR_LANGUAGES, gpu=args.gpu)
    rows, failed = scan_all_videos(reader, args.folder, args.interval)
    export_excel(rows, args.output)

    print("\n==============================")
    print("VIDÉOS EN ERREUR")
    print("==============================")
    for v in failed:
        print(v)
    if not failed:
        print("Aucune 🎉")


if __name__ == "__main__":
    main()
