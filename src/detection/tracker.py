"""
Module de tracking avancé avec gestion des occlusions.
Couche supplémentaire au-dessus du tracking YOLO intégré.

Inclut :
- Savitzky-Golay smoothing sur les positions monde pour lisser les
  micro-tremblements causés par le mouvement de caméra.
- Filtrage intelligent des vitesses aberrantes.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from scipy.signal import savgol_filter

from .detector import Detection, FrameDetections

# Physiological max sprint speed (10.5 m/s ≈ 37.8 km/h — Usain Bolt level)
MAX_PLAYER_SPEED = 10.5


@dataclass
class TrackState:
    """État d'un joueur suivi dans le temps."""
    track_id: int
    positions: List[Tuple[int, float, float]] = field(default_factory=list)
    # (frame_idx, x, y) - positions au sol en pixels
    world_positions: List[Tuple[int, float, float]] = field(default_factory=list)
    # (frame_idx, X, Y) - positions en mètres
    team_id: Optional[int] = None
    speeds: List[float] = field(default_factory=list)       # m/s
    last_seen: int = 0
    is_active: bool = True

    @property
    def last_position(self) -> Optional[Tuple[float, float]]:
        if self.positions:
            return (self.positions[-1][1], self.positions[-1][2])
        return None

    @property
    def last_world_position(self) -> Optional[Tuple[float, float]]:
        if self.world_positions:
            return (self.world_positions[-1][1], self.world_positions[-1][2])
        return None

    @property
    def avg_speed(self) -> float:
        if self.speeds:
            return np.mean(self.speeds[-30:])  # Moyenne sur 30 dernières frames
        return 0.0

    @property
    def max_speed(self) -> float:
        if self.speeds:
            return max(self.speeds)
        return 0.0

    @property
    def total_distance(self) -> float:
        """Distance totale parcourue en mètres."""
        total = 0.0
        for i in range(1, len(self.world_positions)):
            _, x1, y1 = self.world_positions[i - 1]
            _, x2, y2 = self.world_positions[i]
            total += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return total


class MultiObjectTracker:
    """
    Gestionnaire de tracking multi-objets.
    
    Maintient l'état de tous les joueurs suivis avec calcul de vitesse,
    distance parcourue, et gestion des identités perdues/retrouvées.
    """

    def __init__(self, fps: float = 30.0, max_inactive_frames: int = 90):
        self.fps = fps
        self.max_inactive_frames = max_inactive_frames
        self.tracks: Dict[int, TrackState] = {}
        self.frame_count: int = 0

    def update(self, frame_detections: FrameDetections) -> Dict[int, TrackState]:
        """
        Met à jour tous les tracks avec les nouvelles détections.
        
        Two-pass process:
          Pass 1 — update positions for all detected players.
          Pass 2 — compute speeds with common-mode camera drift rejection
                   (the median world displacement across all players is
                    subtracted so camera-induced shifts don't inflate speeds).
        
        Args:
            frame_detections: Détections de la frame courante
            
        Returns:
            Dictionnaire des tracks actifs
        """
        self.frame_count = frame_detections.frame_idx
        active_ids = set()
        updated_world_tracks: list = []  # tracks that got a new world position

        # ── PASS 1 : update positions ─────────────────
        for det in frame_detections.players + frame_detections.referees:
            if det.track_id is None:
                continue

            tid = det.track_id
            active_ids.add(tid)
            foot = det.bottom_center

            if tid not in self.tracks:
                self.tracks[tid] = TrackState(track_id=tid)

            track = self.tracks[tid]
            track.positions.append((self.frame_count, foot[0], foot[1]))
            track.last_seen = self.frame_count
            track.is_active = True

            if det.team_id is not None:
                track.team_id = det.team_id

            if det.world_pos is not None:
                track.world_positions.append(
                    (self.frame_count, det.world_pos[0], det.world_pos[1])
                )
                updated_world_tracks.append(track)

        # ── Estimate camera-induced world drift via common-mode rejection ─
        # If all players shift in the same direction between frames,
        # that's camera motion not real movement — subtract the median.
        raw_displacements = []
        for track in updated_world_tracks:
            n = len(track.world_positions)
            if n >= 2:
                _, x1, y1 = track.world_positions[-2]
                _, x2, y2 = track.world_positions[-1]
                raw_displacements.append((x2 - x1, y2 - y1))

        camera_drift = (0.0, 0.0)
        if len(raw_displacements) >= 4:
            median_dx = float(np.median([d[0] for d in raw_displacements]))
            median_dy = float(np.median([d[1] for d in raw_displacements]))
            drift_magnitude = np.sqrt(median_dx ** 2 + median_dy ** 2)
            # Only apply if drift is significant (>0.05m) to avoid noise
            if drift_magnitude > 0.05:
                camera_drift = (median_dx, median_dy)

        # ── PASS 2 : compute speeds with drift correction ──
        for track in updated_world_tracks:
            speed = self._compute_smoothed_speed(track, camera_drift)
            track.speeds.append(speed)

        # Marquer les tracks inactifs
        for tid, track in self.tracks.items():
            if tid not in active_ids:
                if self.frame_count - track.last_seen > self.max_inactive_frames:
                    track.is_active = False

        return self.get_active_tracks()

    def _compute_smoothed_speed(
        self,
        track: TrackState,
        camera_drift: Tuple[float, float] = (0.0, 0.0),
    ) -> float:
        """
        Calcule la vitesse instantanée avec lissage Savitzky-Golay
        et rejet du mouvement de caméra commun.

        1. Lisse les positions monde sur une fenêtre glissante (SG).
        2. Calcule la dérivée (vitesse) sur les positions lissées.
        3. Soustrait le drift caméra commun (common-mode rejection).
        4. Plafonne à MAX_PLAYER_SPEED = 10.5 m/s (physiologique).

        Fenêtre = 21 frames max (~0.84s @ 25fps), polynôme d'ordre 2.
        """
        n = len(track.world_positions)
        if n < 2:
            return 0.0

        # Fenêtre SG : impaire, max 21 frames pour plus de lissage
        window = min(21, n)
        if window % 2 == 0:
            window -= 1
        if window < 3:
            # Moins de 3 points → calcul brut avec correction drift
            _, x1, y1 = track.world_positions[-2]
            _, x2, y2 = track.world_positions[-1]
            dx = (x2 - x1) - camera_drift[0]
            dy = (y2 - y1) - camera_drift[1]
            speed = np.sqrt(dx ** 2 + dy ** 2) * self.fps
            return min(speed, MAX_PLAYER_SPEED)

        # Extraire les positions récentes
        recent = track.world_positions[-window:]
        xs = np.array([p[1] for p in recent])
        ys = np.array([p[2] for p in recent])

        poly_order = min(2, window - 1)

        # Lisser les coordonnées
        xs_smooth = savgol_filter(xs, window, poly_order)
        ys_smooth = savgol_filter(ys, window, poly_order)

        # Vitesse = dérivée des positions lissées à la dernière position
        # Soustrait le drift caméra commun
        dx = (xs_smooth[-1] - xs_smooth[-2]) - camera_drift[0]
        dy = (ys_smooth[-1] - ys_smooth[-2]) - camera_drift[1]
        speed = np.sqrt(dx ** 2 + dy ** 2) * self.fps  # m/s

        return min(float(speed), MAX_PLAYER_SPEED)

    def get_active_tracks(self) -> Dict[int, TrackState]:
        """Retourne uniquement les tracks actifs."""
        return {tid: t for tid, t in self.tracks.items() if t.is_active}

    def get_team_tracks(self, team_id: int) -> Dict[int, TrackState]:
        """Retourne les tracks d'une équipe spécifique."""
        return {
            tid: t for tid, t in self.tracks.items()
            if t.is_active and t.team_id == team_id
        }

    def get_positions_at_frame(
        self, frame_idx: int, team_id: Optional[int] = None
    ) -> List[Tuple[int, float, float]]:
        """
        Retourne les positions de tous les joueurs à une frame donnée.
        
        Returns:
            Liste de (track_id, x_metres, y_metres)
        """
        positions = []
        for tid, track in self.tracks.items():
            if team_id is not None and track.team_id != team_id:
                continue
            for fidx, x, y in track.world_positions:
                if fidx == frame_idx:
                    positions.append((tid, x, y))
                    break
        return positions

    def get_trail(
        self, track_id: int, length: int = 30
    ) -> List[Tuple[float, float]]:
        """Retourne les N dernières positions pixel d'un track (pour tracer une traîne)."""
        if track_id not in self.tracks:
            return []
        positions = self.tracks[track_id].positions
        trail = [(x, y) for _, x, y in positions[-length:]]
        return trail

    def get_statistics(self) -> Dict:
        """Génère les statistiques globales de tracking."""
        stats = {
            "total_tracks": len(self.tracks),
            "active_tracks": len(self.get_active_tracks()),
            "frames_processed": self.frame_count,
        }

        for tid, track in self.tracks.items():
            if track.is_active and track.world_positions:
                stats[f"player_{tid}"] = {
                    "team": track.team_id,
                    "distance_m": round(track.total_distance, 1),
                    "avg_speed_kmh": round(track.avg_speed * 3.6, 1),
                    "max_speed_kmh": round(track.max_speed * 3.6, 1),
                    "appearances": len(track.positions),
                }

        return stats
