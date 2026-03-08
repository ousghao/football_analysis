"""
Segmentation du terrain de football.
Détecte la pelouse et les lignes blanches pour la calibration.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class PitchSegmenter:
    """
    Segmentation de la pelouse par seuillage couleur HSV.
    Détecte les lignes du terrain par détection de contours.
    """

    def __init__(
        self,
        green_lower: Tuple[int, int, int] = (30, 40, 40),
        green_upper: Tuple[int, int, int] = (80, 255, 255),
        white_lower: Tuple[int, int, int] = (0, 0, 180),
        white_upper: Tuple[int, int, int] = (180, 40, 255),
    ):
        self.green_lower = np.array(green_lower)
        self.green_upper = np.array(green_upper)
        self.white_lower = np.array(white_lower)
        self.white_upper = np.array(white_upper)

    def segment_grass(self, frame: np.ndarray) -> np.ndarray:
        """
        Crée un masque binaire de la pelouse.
        
        Args:
            frame: Image BGR
            
        Returns:
            Masque binaire (255 = pelouse)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        # Nettoyage morphologique
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        return mask

    def detect_lines(self, frame: np.ndarray,
                     grass_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Détecte les lignes blanches du terrain.
        
        Args:
            frame: Image BGR
            grass_mask: Masque de la pelouse (optionnel)
            
        Returns:
            Masque binaire des lignes
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)

        # Si on a un masque de pelouse, n'extraire les lignes que sur la pelouse
        if grass_mask is not None:
            white_mask = cv2.bitwise_and(white_mask, grass_mask)

        # Affiner avec Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Combiner seuillage couleur + edges
        combined = cv2.bitwise_and(white_mask, edges)

        # Dilater légèrement pour connecter les morceaux
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.dilate(combined, kernel, iterations=1)

        return combined

    def detect_hough_lines(
        self, line_mask: np.ndarray,
        rho: float = 1, theta: float = np.pi / 180,
        threshold: int = 100, min_length: int = 80, max_gap: int = 30
    ) -> List[np.ndarray]:
        """
        Détecte les lignes droites via transformée de Hough.
        
        Returns:
            Liste de lignes [(x1, y1, x2, y2), ...]
        """
        lines = cv2.HoughLinesP(
            line_mask, rho, theta, threshold,
            minLineLength=min_length,
            maxLineGap=max_gap
        )

        if lines is None:
            return []

        return [line[0] for line in lines]

    def find_line_intersections(
        self, lines: List[np.ndarray]
    ) -> List[Tuple[float, float]]:
        """
        Trouve les intersections entre les lignes détectées.
        Ces intersections sont des candidats pour les points clés du terrain.
        
        Returns:
            Liste de points (x, y)
        """
        intersections = []

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self._line_intersection(lines[i], lines[j])
                if pt is not None:
                    intersections.append(pt)

        return intersections

    @staticmethod
    def _line_intersection(
        line1: np.ndarray, line2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Calcule l'intersection de deux segments."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None  # Lignes parallèles

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return (px, py)

    def get_pitch_mask_convex(self, frame: np.ndarray) -> np.ndarray:
        """
        Crée un masque convexe de la zone de jeu visible.
        Utile pour filtrer les détections hors terrain.
        """
        grass = self.segment_grass(frame)

        # Trouver le plus grand contour (= le terrain)
        contours, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return grass

        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)

        mask = np.zeros_like(grass)
        cv2.fillPoly(mask, [hull], 255)

        return mask

    def is_on_pitch(self, point: Tuple[float, float],
                    pitch_mask: np.ndarray) -> bool:
        """Vérifie si un point est sur le terrain."""
        x, y = int(point[0]), int(point[1])
        h, w = pitch_mask.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return pitch_mask[y, x] > 0
        return False
