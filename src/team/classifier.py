"""
Classification des équipes par analyse de couleur (K-Means clustering).
Différencie Équipe A / Équipe B / Arbitre / Gardien.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import TEAM_CLASSIFICATION, TeamClassificationConfig
from src.detection.detector import Detection, FrameDetections


@dataclass
class TeamAssignment:
    """Résultat de la classification d'un joueur."""
    track_id: Optional[int]
    team_id: int        # 0=Équipe A, 1=Équipe B, 2=Arbitre/Autre
    confidence: float
    dominant_color: Tuple[int, int, int]  # BGR


class TeamClassifier:
    """
    Classifie les joueurs en équipes via K-Means sur les couleurs de maillot.
    
    Pipeline :
    1. Extraire le crop du joueur (partie haute du bbox = maillot)
    2. Filtrer les pixels de pelouse (vert)
    3. Calculer l'histogramme de couleur dominant
    4. K-Means clustering pour trouver les groupes
    """

    def __init__(self, config: TeamClassificationConfig = None):
        self.config = config or TEAM_CLASSIFICATION
        self.kmeans: Optional[KMeans] = None
        self.team_colors: Dict[int, np.ndarray] = {}   # team_id -> BGR moyen
        self.is_fitted: bool = False

        # Cache pour éviter de reclassifier à chaque frame
        self._track_team_votes: Dict[int, List[int]] = {}

    def extract_player_color(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[np.ndarray]:
        """
        Extrait la couleur dominante du maillot d'un joueur.
        
        Stratégie : prendre la moitié supérieure du bbox (maillot)
        et filtrer les pixels verts (pelouse visible).
        """
        bbox = detection.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        h_img, w_img = frame.shape[:2]

        # Clamp aux limites de l'image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        if x2 - x1 < 5 or y2 - y1 < 10:
            return None

        # Prendre la partie maillot (30-80% de la hauteur)
        shirt_top = y1 + int((y2 - y1) * 0.15)
        shirt_bottom = y1 + int((y2 - y1) * 0.65)
        shirt_left = x1 + int((x2 - x1) * 0.15)
        shirt_right = x2 - int((x2 - x1) * 0.15)

        crop = frame[shirt_top:shirt_bottom, shirt_left:shirt_right]

        if crop.size == 0:
            return None

        # Filtrer les pixels verts (pelouse)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (30, 40, 40), (80, 255, 255))
        non_green_mask = cv2.bitwise_not(green_mask)

        # Extraire les pixels non-verts
        pixels = crop[non_green_mask > 0]

        if len(pixels) < self.config.min_pixels:
            return None

        # Convertir dans l'espace couleur configuré
        if self.config.color_space == "hsv":
            pixels_converted = cv2.cvtColor(
                pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV
            ).reshape(-1, 3)
        elif self.config.color_space == "lab":
            pixels_converted = cv2.cvtColor(
                pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
            ).reshape(-1, 3)
        else:
            pixels_converted = pixels

        # Calculer la couleur dominante via mini K-Means local
        try:
            mini_km = KMeans(n_clusters=2, n_init=3, random_state=42)
            mini_km.fit(pixels_converted)
            # Prendre le cluster le plus fréquent
            labels = mini_km.labels_
            dominant_label = Counter(labels).most_common(1)[0][0]
            # Retourner la couleur en BGR
            dominant_idx = labels == dominant_label
            dominant_color = pixels[dominant_idx].mean(axis=0)
            return dominant_color
        except Exception:
            return pixels.mean(axis=0)

    def fit(
        self, frame: np.ndarray, detections: FrameDetections
    ) -> None:
        """
        Entraîne le classifieur sur une frame représentative.
        Devrait être appelé sur une frame où les deux équipes sont bien visibles.
        
        Args:
            frame: Image BGR
            detections: Détections de la frame
        """
        colors = []
        valid_detections = []
        frame_h = frame.shape[0]

        for det in detections.players:
            # Exclure les personnes dans les 12% inférieurs du cadre :
            # ramasseurs de balles et spectateurs derrière les panneaux publicitaires.
            # Leur maillot ne représente pas une équipe en jeu.
            if det.bbox[3] > frame_h * 0.88:
                continue
            color = self.extract_player_color(frame, det)
            if color is not None:
                colors.append(color)
                valid_detections.append(det)

        if len(colors) < self.config.n_clusters:
            print(f"[Équipes] Pas assez de joueurs détectés ({len(colors)}) "
                  f"pour {self.config.n_clusters} clusters.")
            return

        colors_array = np.array(colors, dtype=np.float32)

        # K-Means global
        self.kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            n_init=10,
            random_state=42
        )
        self.kmeans.fit(colors_array)

        # Stocker les couleurs des clusters
        for i in range(self.config.n_clusters):
            cluster_mask = self.kmeans.labels_ == i
            cluster_colors = colors_array[cluster_mask]
            if len(cluster_colors) > 0:
                self.team_colors[i] = cluster_colors.mean(axis=0).astype(int)

        self.is_fitted = True

        # Identifier quel cluster est l'arbitre (le plus petit cluster)
        label_counts = Counter(self.kmeans.labels_)
        sorted_labels = label_counts.most_common()

        print(f"[Équipes] Clustering terminé:")
        for label, count in sorted_labels:
            color = self.team_colors[label]
            print(f"  Cluster {label}: {count} joueurs, "
                  f"couleur BGR=({color[0]:.0f}, {color[1]:.0f}, {color[2]:.0f})")

        # Les deux plus gros clusters = les deux équipes
        # Le plus petit = arbitre/gardien
        if len(sorted_labels) >= 3:
            self._referee_cluster = sorted_labels[-1][0]
        else:
            self._referee_cluster = -1

    def classify(
        self, frame: np.ndarray, detection: Detection
    ) -> TeamAssignment:
        """
        Classifie un joueur dans une équipe.
        
        Args:
            frame: Image BGR
            detection: Détection du joueur
            
        Returns:
            TeamAssignment avec l'ID d'équipe
        """
        if not self.is_fitted or self.kmeans is None:
            return TeamAssignment(
                track_id=detection.track_id,
                team_id=-1,
                confidence=0.0,
                dominant_color=(128, 128, 128),
            )

        color = self.extract_player_color(frame, detection)
        if color is None:
            return TeamAssignment(
                track_id=detection.track_id,
                team_id=-1,
                confidence=0.0,
                dominant_color=(128, 128, 128),
            )

        # Prédiction
        color_array = color.reshape(1, -1).astype(np.float32)
        cluster = self.kmeans.predict(color_array)[0]

        # Distance au centroïde (pour la confiance)
        centroid = self.kmeans.cluster_centers_[cluster]
        distance = np.linalg.norm(color_array - centroid)
        max_dist = 200.0  # Distance max attendue
        confidence = max(0.0, 1.0 - distance / max_dist)

        # Mapper le cluster à un team_id
        team_id = int(cluster)
        if cluster == getattr(self, "_referee_cluster", -1):
            team_id = 2  # Arbitre

        # Vote temporel pour stabiliser la classification
        if detection.track_id is not None:
            tid = detection.track_id
            if tid not in self._track_team_votes:
                self._track_team_votes[tid] = []
            self._track_team_votes[tid].append(team_id)

            # Garder les 15 derniers votes (fenêtre plus courte = correction plus rapide)
            self._track_team_votes[tid] = self._track_team_votes[tid][-15:]

            # Décision par vote majoritaire
            team_id = Counter(self._track_team_votes[tid]).most_common(1)[0][0]

        return TeamAssignment(
            track_id=detection.track_id,
            team_id=team_id,
            confidence=confidence,
            dominant_color=tuple(color.astype(int)),
        )

    def classify_frame(
        self, frame: np.ndarray, detections: FrameDetections
    ) -> FrameDetections:
        """
        Classifie tous les joueurs d'une frame et met à jour les détections.
        
        Args:
            frame: Image BGR
            detections: Détections à classifier
            
        Returns:
            FrameDetections avec team_id assignés
        """
        classified_players = []
        referees = []

        for det in detections.players:
            assignment = self.classify(frame, det)
            det.team_id = assignment.team_id

            if assignment.team_id == 2:
                referees.append(det)
            else:
                classified_players.append(det)

        detections.players = classified_players
        detections.referees = referees

        return detections

    def get_team_color_bgr(self, team_id: int) -> Tuple[int, int, int]:
        """Retourne la couleur BGR moyenne d'une équipe."""
        if team_id in self.team_colors:
            c = self.team_colors[team_id]
            return (int(c[0]), int(c[1]), int(c[2]))
        return (128, 128, 128)
