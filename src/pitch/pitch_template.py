"""
Template du terrain de football 2D (vue de dessus).
Définit toutes les coordonnées clés du terrain en mètres (norme FIFA).
"""

import numpy as np
from typing import Dict, List, Tuple
import cv2


# ─── Dimensions FIFA standard ──────────────────────────────
PITCH_LENGTH = 105.0    # mètres
PITCH_WIDTH = 68.0      # mètres
GOAL_WIDTH = 7.32       # mètres
GOAL_DEPTH = 2.44       # mètres (hauteur, pas utilisé en 2D)
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.32
CENTER_CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DISTANCE = 11.0
CORNER_ARC_RADIUS = 1.0


class PitchTemplate:
    """
    Modèle 2D du terrain de football avec toutes les lignes et points clés.
    
    Coordonnées en mètres, origine (0, 0) = coin supérieur gauche.
    Axe X = longueur (0 -> 105m), Axe Y = largeur (0 -> 68m).
    """

    def __init__(self, length: float = PITCH_LENGTH, width: float = PITCH_WIDTH):
        self.length = length
        self.width = width
        self._build_keypoints()

    def _build_keypoints(self):
        """Construit tous les points clés du terrain."""
        L, W = self.length, self.width
        half_W = W / 2
        half_L = L / 2

        # ─── Les 4 coins du terrain ────────────────────
        self.corners = {
            "top_left": (0.0, 0.0),
            "top_right": (L, 0.0),
            "bottom_right": (L, W),
            "bottom_left": (0.0, W),
        }

        # ─── Ligne médiane ─────────────────────────────
        self.halfway_line = [
            (half_L, 0.0),
            (half_L, W),
        ]
        self.center_spot = (half_L, half_W)

        # ─── Surface de réparation GAUCHE ──────────────
        pa_top = half_W - PENALTY_AREA_WIDTH / 2
        pa_bot = half_W + PENALTY_AREA_WIDTH / 2
        self.penalty_area_left = [
            (0.0, pa_top),
            (PENALTY_AREA_LENGTH, pa_top),
            (PENALTY_AREA_LENGTH, pa_bot),
            (0.0, pa_bot),
        ]
        self.penalty_spot_left = (PENALTY_SPOT_DISTANCE, half_W)

        # ─── Surface de but GAUCHE ─────────────────────
        ga_top = half_W - GOAL_AREA_WIDTH / 2
        ga_bot = half_W + GOAL_AREA_WIDTH / 2
        self.goal_area_left = [
            (0.0, ga_top),
            (GOAL_AREA_LENGTH, ga_top),
            (GOAL_AREA_LENGTH, ga_bot),
            (0.0, ga_bot),
        ]

        # ─── Surface de réparation DROITE ──────────────
        self.penalty_area_right = [
            (L, pa_top),
            (L - PENALTY_AREA_LENGTH, pa_top),
            (L - PENALTY_AREA_LENGTH, pa_bot),
            (L, pa_bot),
        ]
        self.penalty_spot_right = (L - PENALTY_SPOT_DISTANCE, half_W)

        # ─── Surface de but DROITE ─────────────────────
        self.goal_area_right = [
            (L, ga_top),
            (L - GOAL_AREA_LENGTH, ga_top),
            (L - GOAL_AREA_LENGTH, ga_bot),
            (L, ga_bot),
        ]

        # ─── But GAUCHE ───────────────────────────────
        goal_top = half_W - GOAL_WIDTH / 2
        goal_bot = half_W + GOAL_WIDTH / 2
        self.goal_left = [
            (0.0, goal_top),
            (0.0, goal_bot),
        ]

        # ─── But DROITE ───────────────────────────────
        self.goal_right = [
            (L, goal_top),
            (L, goal_bot),
        ]

    def get_all_keypoints(self) -> Dict[str, Tuple[float, float]]:
        """
        Retourne tous les points de repère nommés du terrain.
        Utile pour la calibration manuelle de l'homographie.
        """
        keypoints = {
            # Coins
            "corner_TL": self.corners["top_left"],
            "corner_TR": self.corners["top_right"],
            "corner_BR": self.corners["bottom_right"],
            "corner_BL": self.corners["bottom_left"],

            # Ligne médiane
            "halfway_T": self.halfway_line[0],
            "halfway_B": self.halfway_line[1],
            "center_spot": self.center_spot,

            # Surface penalty gauche
            "penalty_L_TL": self.penalty_area_left[0],
            "penalty_L_TR": self.penalty_area_left[1],
            "penalty_L_BR": self.penalty_area_left[2],
            "penalty_L_BL": self.penalty_area_left[3],
            "penalty_spot_L": self.penalty_spot_left,

            # Surface penalty droite
            "penalty_R_TR": self.penalty_area_right[0],
            "penalty_R_TL": self.penalty_area_right[1],
            "penalty_R_BL": self.penalty_area_right[2],
            "penalty_R_BR": self.penalty_area_right[3],
            "penalty_spot_R": self.penalty_spot_right,

            # Surface de but gauche
            "goal_area_L_TL": self.goal_area_left[0],
            "goal_area_L_TR": self.goal_area_left[1],
            "goal_area_L_BR": self.goal_area_left[2],
            "goal_area_L_BL": self.goal_area_left[3],

            # Surface de but droite
            "goal_area_R_TR": self.goal_area_right[0],
            "goal_area_R_TL": self.goal_area_right[1],
            "goal_area_R_BL": self.goal_area_right[2],
            "goal_area_R_BR": self.goal_area_right[3],
        }
        return keypoints

    def render(self, width_px: int = 1050, height_px: int = 680,
               bg_color: Tuple[int, int, int] = (34, 139, 34),
               line_color: Tuple[int, int, int] = (255, 255, 255),
               line_thickness: int = 2) -> np.ndarray:
        """
        Rend une image 2D du terrain.
        
        Args:
            width_px, height_px: Dimensions de l'image en pixels
            bg_color: Couleur de fond (BGR)
            line_color: Couleur des lignes (BGR)
            
        Returns:
            Image numpy BGR du terrain
        """
        img = np.full((height_px, width_px, 3), bg_color, dtype=np.uint8)

        def to_px(point: Tuple[float, float]) -> Tuple[int, int]:
            """Convertit mètres -> pixels."""
            x = int(point[0] / self.length * width_px)
            y = int(point[1] / self.width * height_px)
            return (x, y)

        # Contour du terrain
        pts = [to_px(self.corners[k]) for k in ["top_left", "top_right", "bottom_right", "bottom_left"]]
        cv2.polylines(img, [np.array(pts)], True, line_color, line_thickness)

        # Ligne médiane
        cv2.line(img, to_px(self.halfway_line[0]), to_px(self.halfway_line[1]),
                 line_color, line_thickness)

        # Cercle central
        center_px = to_px(self.center_spot)
        radius_px = int(CENTER_CIRCLE_RADIUS / self.length * width_px)
        cv2.circle(img, center_px, radius_px, line_color, line_thickness)
        cv2.circle(img, center_px, 4, line_color, -1)  # Point central

        # Surface de réparation gauche
        pts = [to_px(p) for p in self.penalty_area_left]
        cv2.polylines(img, [np.array(pts)], False, line_color, line_thickness)
        # Fermer du côté de la ligne de but
        cv2.line(img, to_px(self.penalty_area_left[0]),
                 to_px(self.penalty_area_left[3]), line_color, line_thickness)

        # Surface de réparation droite
        pts = [to_px(p) for p in self.penalty_area_right]
        cv2.polylines(img, [np.array(pts)], False, line_color, line_thickness)
        cv2.line(img, to_px(self.penalty_area_right[0]),
                 to_px(self.penalty_area_right[3]), line_color, line_thickness)

        # Surface de but gauche
        pts = [to_px(p) for p in self.goal_area_left]
        cv2.polylines(img, [np.array(pts)], False, line_color, line_thickness)
        cv2.line(img, to_px(self.goal_area_left[0]),
                 to_px(self.goal_area_left[3]), line_color, line_thickness)

        # Surface de but droite
        pts = [to_px(p) for p in self.goal_area_right]
        cv2.polylines(img, [np.array(pts)], False, line_color, line_thickness)
        cv2.line(img, to_px(self.goal_area_right[0]),
                 to_px(self.goal_area_right[3]), line_color, line_thickness)

        # Points de penalty
        cv2.circle(img, to_px(self.penalty_spot_left), 4, line_color, -1)
        cv2.circle(img, to_px(self.penalty_spot_right), 4, line_color, -1)

        # Arcs de corner (quarts de cercle)
        corner_r_px = int(CORNER_ARC_RADIUS / self.length * width_px)
        # Coin haut-gauche
        cv2.ellipse(img, to_px((0, 0)), (corner_r_px, corner_r_px),
                     0, 0, 90, line_color, line_thickness)
        # Coin haut-droit
        cv2.ellipse(img, to_px((self.length, 0)), (corner_r_px, corner_r_px),
                     0, 90, 180, line_color, line_thickness)
        # Coin bas-droit
        cv2.ellipse(img, to_px((self.length, self.width)), (corner_r_px, corner_r_px),
                     0, 180, 270, line_color, line_thickness)
        # Coin bas-gauche
        cv2.ellipse(img, to_px((0, self.width)), (corner_r_px, corner_r_px),
                     0, 270, 360, line_color, line_thickness)

        return img

    def render_minimap(self, width_px: int = 350, height_px: int = 230) -> np.ndarray:
        """Version miniature pour l'overlay sur la vidéo."""
        return self.render(width_px, height_px, line_thickness=1)
