"""
Global Motion Compensation (GMC) — Compensation du mouvement de caméra.

Quand le cameraman pivote (pan/tilt/zoom), le déplacement des pixels est 
interprété comme un déplacement physique des joueurs. Ce module calcule 
le mouvement global de la caméra entre deux frames et le soustrait.

Formule : Vitesse_réelle = Vitesse_détectée - Vitesse_caméra

Technique : Optical Flow sur des points d'arrière-plan (publicités, lignes, tribunes)
puis estimation d'une transformation affine globale.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class GlobalMotionCompensator:
    """
    Estime le mouvement de caméra entre deux frames consécutives
    via Optical Flow (Lucas-Kanade) sur des feature points du décor.
    
    Retourne une matrice affine 2x3 qui représente le déplacement
    global de la caméra. On peut l'appliquer pour corriger les
    positions pixel des joueurs avant la projection homographique.
    """

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 30,
        block_size: int = 3,
        win_size: Tuple[int, int] = (21, 21),
        max_level: int = 3,
    ):
        # Paramètres pour goodFeaturesToTrack
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )

        # Paramètres pour calcOpticalFlowPyrLK
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30, 0.01
            ),
        )

        self.prev_gray: Optional[np.ndarray] = None
        self.camera_motion: np.ndarray = np.eye(2, 3, dtype=np.float64)
        # Cumulative camera displacement in pixels (dx, dy)
        self.cumulative_dx: float = 0.0
        self.cumulative_dy: float = 0.0

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Calcule le mouvement de la caméra entre la frame précédente et l'actuelle.
        
        Args:
            frame: Image BGR courante
            
        Returns:
            Matrice affine 2x3 du mouvement caméra (identité si 1ère frame)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.camera_motion = np.eye(2, 3, dtype=np.float64)
            return self.camera_motion

        # Détecter des feature points sur la frame précédente
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)

        if prev_pts is None or len(prev_pts) < 10:
            self.prev_gray = gray
            self.camera_motion = np.eye(2, 3, dtype=np.float64)
            return self.camera_motion

        # Suivre ces points dans la frame courante via Optical Flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        # Garder seulement les points bien suivis
        status = status.flatten()
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 6:
            self.prev_gray = gray
            self.camera_motion = np.eye(2, 3, dtype=np.float64)
            return self.camera_motion

        # Estimer la transformation affine partielle (translation + rotation + scale)
        # RANSAC pour rejeter les outliers (= les joueurs qui bougent ≠ décor fixe)
        transform, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        if transform is not None:
            self.camera_motion = transform
            # Extraire le déplacement de la caméra (translation)
            self.cumulative_dx += transform[0, 2]
            self.cumulative_dy += transform[1, 2]
        else:
            self.camera_motion = np.eye(2, 3, dtype=np.float64)

        self.prev_gray = gray
        return self.camera_motion

    def compensate_point(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Compense un point pixel en soustrayant le mouvement de caméra.
        
        Le point est ramené à la position qu'il aurait si la caméra était fixe.
        
        Args:
            point: (x_pixel, y_pixel)
            
        Returns:
            (x_compensé, y_compensé)
        """
        # La matrice affine est :
        # [a  b  tx]
        # [c  d  ty]
        # On veut l'inverse : ramener le point au référentiel fixe
        M = self.camera_motion
        # Construire la matrice 3x3 complète
        M_full = np.eye(3, dtype=np.float64)
        M_full[:2, :] = M

        try:
            M_inv = np.linalg.inv(M_full)
        except np.linalg.LinAlgError:
            return point

        pt = np.array([point[0], point[1], 1.0])
        compensated = M_inv @ pt

        return (float(compensated[0]), float(compensated[1]))

    def get_camera_displacement(self) -> Tuple[float, float]:
        """Retourne le déplacement de caméra de la dernière frame (dx, dy en pixels)."""
        return (float(self.camera_motion[0, 2]), float(self.camera_motion[1, 2]))

    def get_camera_speed_px(self) -> float:
        """Vitesse de la caméra en pixels/frame."""
        dx, dy = self.get_camera_displacement()
        return float(np.sqrt(dx**2 + dy**2))

    def is_camera_moving(self, threshold_px: float = 1.5) -> bool:
        """Détecte si la caméra est en mouvement significatif.
        
        Seuil abaissé à 1.5px (vs 3.0) pour activer la compensation
        plus tôt et réduire les artefacts de vitesse dus au panoramique.
        """
        return self.get_camera_speed_px() > threshold_px
