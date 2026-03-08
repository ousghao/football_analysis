"""
Module de détection d'objets avec YOLOv11/v8.
Détecte les joueurs, arbitre et ballon sur chaque frame.
"""

import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import sys
import os

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DETECTION, DetectionConfig


@dataclass
class Detection:
    """Représente une détection unique sur une frame."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    team_id: Optional[int] = None      # 0=Équipe A, 1=Équipe B, 2=Arbitre
    foot_point: Optional[Tuple[float, float]] = None  # Point au sol (bas du bbox)
    world_pos: Optional[Tuple[float, float]] = None     # Position en mètres sur le terrain

    @property
    def center(self) -> Tuple[float, float]:
        """Centre du bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Point bas-centre du bbox (point de contact au sol)."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            self.bbox[3]
        )

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    def get_crop(self, frame: np.ndarray, padding: float = 0.0) -> np.ndarray:
        """Extrait le crop du joueur depuis la frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.bbox.astype(int)

        # Ajouter le padding
        pad_w = int(self.width * padding)
        pad_h = int(self.height * padding)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        return frame[y1:y2, x1:x2]


@dataclass
class FrameDetections:
    """Toutes les détections d'une frame."""
    frame_idx: int
    players: List[Detection] = field(default_factory=list)
    ball: Optional[Detection] = None
    referees: List[Detection] = field(default_factory=list)
    goalkeepers: List[Detection] = field(default_factory=list)

    @property
    def all_persons(self) -> List[Detection]:
        """Toutes les personnes détectées."""
        return self.players + self.referees + self.goalkeepers

    @property
    def all_detections(self) -> List[Detection]:
        """Toutes les détections incluant le ballon."""
        dets = self.all_persons
        if self.ball:
            dets.append(self.ball)
        return dets


class PlayerDetector:
    """
    Détecteur de joueurs, arbitre et ballon basé sur YOLO.
    
    Utilise YOLOv11 (ou v8) pour la détection en temps réel
    avec filtrage par classe et confiance.
    """

    def __init__(self, config: DetectionConfig = None):
        self.config = config or DETECTION
        self._load_model()

    def _load_model(self):
        """Charge le modèle YOLO."""
        print(f"[Détection] Chargement du modèle: {self.config.model_path}")
        self.model = YOLO(self.config.model_path)

        # Déterminer le device
        if self.config.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

        print(f"[Détection] Device: {self.device}")

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> FrameDetections:
        """
        Détecte les objets sur une frame.
        
        Args:
            frame: Image BGR (OpenCV)
            frame_idx: Index de la frame
            
        Returns:
            FrameDetections avec toutes les détections classifiées
        """
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False
        )[0]

        frame_dets = FrameDetections(frame_idx=frame_idx)

        if results.boxes is None or len(results.boxes) == 0:
            return frame_dets

        boxes = results.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].cpu().numpy()
            class_name = self.model.names[class_id]

            # Filtrer les classes non pertinentes
            if class_id not in self.config.target_classes:
                continue

            det = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                foot_point=((bbox[0] + bbox[2]) / 2, bbox[3])
            )

            if class_id == self.config.ball_class:
                # Garder la détection de ballon la plus confiante
                if frame_dets.ball is None or confidence > frame_dets.ball.confidence:
                    frame_dets.ball = det
            elif class_id == self.config.person_class:
                # Filtrer les détections trop petites (bruit) ou trop grandes (foule)
                if det.area > 500 and det.height > 30:
                    frame_dets.players.append(det)

        return frame_dets

    def detect_with_tracking(
        self, frame: np.ndarray, frame_idx: int = 0, persist: bool = True
    ) -> FrameDetections:
        """
        Détection + Tracking intégré via YOLO.
        Assigne un ID unique à chaque joueur pour le suivi temporel.
        
        Args:
            frame: Image BGR
            frame_idx: Index de la frame
            persist: Maintenir les IDs entre les frames
            
        Returns:
            FrameDetections avec track_id assignés
        """
        results = self.model.track(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.device,
            persist=persist,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        frame_dets = FrameDetections(frame_idx=frame_idx)

        if results.boxes is None or len(results.boxes) == 0:
            return frame_dets

        boxes = results.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].cpu().numpy()
            class_name = self.model.names[class_id]

            # ID de tracking
            track_id = None
            if boxes.id is not None:
                track_id = int(boxes.id[i].item())

            if class_id not in self.config.target_classes:
                continue

            det = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                track_id=track_id,
                foot_point=((bbox[0] + bbox[2]) / 2, bbox[3])
            )

            if class_id == self.config.ball_class:
                if frame_dets.ball is None or confidence > frame_dets.ball.confidence:
                    frame_dets.ball = det
            elif class_id == self.config.person_class:
                if det.area > 500 and det.height > 30:
                    frame_dets.players.append(det)

        return frame_dets


class BallTracker:
    """
    Traqueur spécialisé pour le ballon avec interpolation.
    Le ballon est souvent perdu (occlusion, flou), donc on interpole sa position.
    """

    def __init__(self, max_gap: int = 15):
        self.max_gap = max_gap          # Gap max avant de perdre le ballon
        self.history: List[Optional[Tuple[float, float]]] = []
        self.last_seen_idx: int = -1

    def update(self, ball_detection: Optional[Detection], frame_idx: int):
        """Met à jour la position du ballon."""
        # Remplir les gaps
        while len(self.history) < frame_idx:
            self.history.append(None)

        if ball_detection is not None:
            pos = ball_detection.center
            self.history.append(pos)
            self.last_seen_idx = frame_idx
        else:
            self.history.append(None)

    def get_interpolated_positions(self) -> List[Optional[Tuple[float, float]]]:
        """
        Interpole les positions manquantes du ballon.
        Utilise une interpolation linéaire entre deux positions connues.
        """
        interpolated = list(self.history)
        n = len(interpolated)

        i = 0
        while i < n:
            if interpolated[i] is None:
                # Trouver le début et la fin du gap
                start = i - 1
                end = i
                while end < n and interpolated[end] is None:
                    end += 1

                # Interpoler si le gap n'est pas trop grand
                if start >= 0 and end < n and (end - start - 1) <= self.max_gap:
                    x1, y1 = interpolated[start]
                    x2, y2 = interpolated[end]
                    gap_len = end - start
                    for j in range(start + 1, end):
                        t = (j - start) / gap_len
                        interpolated[j] = (
                            x1 + t * (x2 - x1),
                            y1 + t * (y2 - y1)
                        )

                i = end
            else:
                i += 1

        return interpolated
