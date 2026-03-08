"""
Configuration centrale du projet Football Tactical Intelligence.
Toutes les constantes et paramètres sont centralisés ici.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List


# ─── Chemins ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Créer les dossiers s'ils n'existent pas
for d in [MODELS_DIR, DATA_DIR, OUTPUT_DIR, DATA_DIR / "sample_videos"]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Dimensions réelles du terrain (en mètres, norme FIFA) ──
PITCH_LENGTH = 105.0   # longueur en mètres
PITCH_WIDTH = 68.0     # largeur en mètres


# ─── Détection YOLO ─────────────────────────────────────────
@dataclass
class DetectionConfig:
    model_path: str = "yolo11n.pt"          # Modèle YOLO (auto-download)
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.5
    device: str = "cpu"                     # "cpu", "cuda", "mps", "auto"
    imgsz: int = 1280                       # Résolution d'entrée
    # Classes COCO pertinentes : 0=person, 32=sports ball
    target_classes: List[int] = field(default_factory=lambda: [0, 32])
    ball_class: int = 32
    person_class: int = 0


# ─── Tracking ────────────────────────────────────────────────
@dataclass
class TrackingConfig:
    tracker_type: str = "bytetrack"         # "bytetrack" ou "botsort"
    track_thresh: float = 0.25
    track_buffer: int = 30                  # Frames avant de perdre un ID
    match_thresh: float = 0.8


# ─── Identification des équipes ──────────────────────────────
@dataclass
class TeamClassificationConfig:
    n_clusters: int = 3                     # Équipe A, Équipe B, Arbitre/Gardien
    color_space: str = "hsv"                # "rgb", "hsv", "lab"
    sample_region: Tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.9)
    # Fraction du bbox à utiliser pour l'échantillonnage couleur (top, left, bottom, right)
    min_pixels: int = 50                    # Pixels minimum pour classifier


# ─── Homographie ─────────────────────────────────────────────
@dataclass
class HomographyConfig:
    min_points: int = 4                     # Points minimum pour calculer H
    ransac_threshold: float = 5.0
    # Points de référence du terrain 2D (en mètres)
    pitch_corners_2d: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.0),                          # Coin supérieur gauche
        (PITCH_LENGTH, 0.0),                 # Coin supérieur droit
        (PITCH_LENGTH, PITCH_WIDTH),         # Coin inférieur droit
        (0.0, PITCH_WIDTH),                  # Coin inférieur gauche
    ])


# ─── Analyse Tactique ───────────────────────────────────────
@dataclass
class TacticalConfig:
    # Seuils de hauteur de bloc (en % de la longueur du terrain)
    high_block_threshold: float = 0.60      # > 60% = pressing haut
    mid_block_threshold: float = 0.40       # 40-60% = bloc médian
    # low < 40% = bloc bas

    # Pressing
    pressing_distance_m: float = 10.0       # Distance de pressing en mètres
    pressing_min_players: int = 3           # Nb min de joueurs pour "pressing intense"

    # Compacité
    compact_threshold_m: float = 30.0       # Distance inter-lignes max pour être "compact"

    # Seuil pour la possession territoriale
    possession_zones: int = 3               # Diviser le terrain en 3 zones (Défense, Milieu, Attaque)


# ─── Visualisation ───────────────────────────────────────────
@dataclass
class VisualizationConfig:
    # Couleurs des équipes (BGR pour OpenCV)
    team_a_color: Tuple[int, int, int] = (255, 50, 50)      # Bleu
    team_b_color: Tuple[int, int, int] = (50, 50, 255)      # Rouge
    referee_color: Tuple[int, int, int] = (0, 255, 255)     # Jaune
    ball_color: Tuple[int, int, int] = (0, 255, 0)          # Vert
    unknown_color: Tuple[int, int, int] = (200, 200, 200)   # Gris

    # Minimap
    minimap_width: int = 350
    minimap_height: int = 230
    minimap_position: str = "bottom-right"   # Position sur la vidéo
    minimap_opacity: float = 0.85

    # Annotations
    show_ids: bool = True
    show_speed: bool = True
    show_trails: bool = True
    trail_length: int = 30                   # Frames de trace

    # Output vidéo
    output_fps: int = 30
    output_codec: str = "mp4v"


# ─── Instanciation des configs ──────────────────────────────
DETECTION = DetectionConfig()
TRACKING = TrackingConfig()
TEAM_CLASSIFICATION = TeamClassificationConfig()
HOMOGRAPHY = HomographyConfig()
TACTICAL = TacticalConfig()
VISUALIZATION = VisualizationConfig()
