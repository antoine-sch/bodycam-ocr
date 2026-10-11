#!/usr/bin/env python3
"""
Carte animée des positions GPS issues de l'OCR des bandeaux vidéo.

Lit le fichier Excel produit par ocr_videos_folder.py, garde les points d'une
date et d'un ou plusieurs créneaux horaires, puis génère pour chaque créneau
une carte HTML (Folium) avec une frise temporelle : un point par minute et par
agent, une couleur par agent.

Dépendances :
    pip install -r requirements.txt

Usage :
    python create_interactive_map.py exemple_paris.xlsx --date 2024-06-15 --slot 12-14
    python create_interactive_map.py SCAN_VIDEOS.xlsx --date 2024-06-15 --slot 12-13 --slot 13-14 -o ./cartes
"""

import argparse
from datetime import datetime
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import TimestampedGeoJson

# --- Configuration ---

PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33",
    "#a65628", "#f781bf", "#17becf", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854", "#ffd92f", "#b3b3b3",
]
POINT_RADIUS = 7
POINT_OPACITY = 0.7


# --- Préparation des données ---

def parse_coordinate(series, positive, negative):
    """Convertit '48.85N' / '2.35W' en nombre signé (48.85 / -2.35)."""
    s = series.astype(str).str.strip().str.upper()
    sign = s.str.endswith(negative).map({True: -1, False: 1})
    value = pd.to_numeric(s.str.rstrip(positive + negative), errors="coerce")
    return value * sign


def load_data(excel_path):
    """Charge l'export OCR et garde les lignes avec GPS et horodatage valides."""
    df = pd.read_excel(excel_path)

    df["GPS_LAT"] = parse_coordinate(df["GPS_LAT"], "N", "S")
    df["GPS_LONG"] = parse_coordinate(df["GPS_LONG"], "E", "W")
    df = df[df["GPS_LAT"].between(-90, 90) & df["GPS_LONG"].between(-180, 180)]

    df["timestamp"] = pd.to_datetime(
        df["DATE"].astype(str) + " " + df["HEURE"].astype(str),
        errors="coerce",
        dayfirst=True,
    )
    df = df.dropna(subset=["timestamp"])

    df["OFFICER_ID"] = df["OFFICER_ID"].astype(str).str.strip()
    return df


def filter_slot(df, date, start_hour, end_hour):
    """Garde les points du créneau, un seul par agent et par minute (le dernier)."""
    start = pd.Timestamp(f"{date} {start_hour:02d}:00:00")
    end = pd.Timestamp(f"{date} {end_hour:02d}:00:00")

    slot = df[df["timestamp"].between(start, end)].copy()
    slot["time_bin"] = slot["timestamp"].dt.floor("min")
    slot = slot.sort_values("timestamp")
    return slot.drop_duplicates(subset=["OFFICER_ID", "time_bin"], keep="last")


# --- Carte ---

def popup_html(row, show_path):
    lines = [
        f"<li><b>N° AGENT :</b> {row['OFFICER_ID']}</li>",
        f"<li><b>HEURE :</b> {row['HEURE']}</li>",
        f"<li><b>FICHIER VIDEO :</b> {row['FILENAME']}</li>",
        f"<li><b>TIMECODE :</b> {row['TC_IN_VIDEO']}</li>",
    ]
    if show_path:
        lines.append(f"<li><b>CHEMIN :</b> {row['URL_FILE']}</li>")
    return ('<ul style="padding-left:15px; margin:0; font-size:15px;">'
            + "".join(lines) + "</ul>")


def build_map(slot, duration_minutes, show_path):
    colors = {
        officer: PALETTE[i % len(PALETTE)]
        for i, officer in enumerate(slot["OFFICER_ID"].unique())
    }

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["GPS_LONG"], row["GPS_LAT"]],
            },
            "properties": {
                "time": row["time_bin"].isoformat(),
                "popup": popup_html(row, show_path),
                "icon": "circle",
                "iconstyle": {
                    "fillColor": colors[row["OFFICER_ID"]],
                    "fillOpacity": POINT_OPACITY,
                    "stroke": False,
                    "radius": POINT_RADIUS,
                },
            },
        }
        for _, row in slot.iterrows()
    ]

    m = folium.Map(
        location=[slot["GPS_LAT"].mean(), slot["GPS_LONG"].mean()],
        min_zoom=7,
        max_zoom=20,
        tiles="Esri.WorldImagery",
        attr="Esri",
    )
    # Cadre la carte sur l'ensemble des points du créneau
    m.fit_bounds([
        [slot["GPS_LAT"].min(), slot["GPS_LONG"].min()],
        [slot["GPS_LAT"].max(), slot["GPS_LONG"].max()],
    ], max_zoom=18)
    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="PT1M",
        add_last_point=False,
        auto_play=False,
        loop=False,
        duration=f"PT{duration_minutes}M",
    ).add_to(m)
    return m


# --- Exécution ---

def parse_slot(value):
    """'12-14' -> (12, 14)"""
    try:
        start, end = (int(h) for h in value.split("-"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"créneau invalide : {value!r} (attendu : 12-14)") from None
    if not 0 <= start < end <= 23:
        raise argparse.ArgumentTypeError(f"créneau invalide : {value!r} (heures entre 0 et 23)")
    return start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("excel", type=Path, help="fichier Excel produit par ocr_videos_folder.py")
    parser.add_argument("--date", required=True, help="jour à cartographier (AAAA-MM-JJ)")
    parser.add_argument("--slot", type=parse_slot, action="append", required=True,
                        help="créneau horaire, ex. 12-14 (option répétable)")
    parser.add_argument("--duration", type=int, default=1,
                        help="durée d'affichage d'un point sur la carte, en minutes (défaut : 1)")
    parser.add_argument("-o", "--output", type=Path, default=Path("_output"),
                        help="dossier de sortie des cartes (défaut : _output)")
    parser.add_argument("--show-path", action="store_true",
                        help="afficher le chemin complet de la vidéo dans les popups")
    args = parser.parse_args()

    df = load_data(args.excel)
    args.output.mkdir(parents=True, exist_ok=True)
    run_stamp = f"{datetime.now():%Y%m%d-%H%M%S}"

    for start_hour, end_hour in args.slot:
        slot = filter_slot(df, args.date, start_hour, end_hour)
        if slot.empty:
            print(f"Aucun point pour {start_hour}h-{end_hour}h")
            continue

        output_file = args.output / f"map_{start_hour}h-{end_hour}h_{run_stamp}.html"
        build_map(slot, args.duration, args.show_path).save(output_file)
        print(f"Carte créée : {output_file}")


if __name__ == "__main__":
    main()
