"""
Annotateur de frames vidéo.
Dessine les bounding boxes, IDs, vitesses, informations tactiques sur chaque frame.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import VISUALIZATION, VisualizationConfig
from src.detection.detector import Detection, FrameDetections
from src.analysis.tactical import TacticalSnapshot, GamePhase
from src.analysis.advanced_tactical import AdvancedTacticalResult


class FrameAnnotator:
    """
    Dessine les annotations sur les frames vidéo :
    - Bounding boxes colorés par équipe
    - IDs de tracking
    - Vitesse en km/h
    - Traîne de mouvement
    - Informations tactiques (phase, bloc, etc.)
    """

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VISUALIZATION
        self.trails: Dict[int, List[Tuple[int, int]]] = {}

    def get_team_color(self, team_id: Optional[int]) -> Tuple[int, int, int]:
        """Retourne la couleur BGR pour une équipe."""
        if team_id == 0:
            return self.config.team_a_color
        elif team_id == 1:
            return self.config.team_b_color
        elif team_id == 2:
            return self.config.referee_color
        return self.config.unknown_color

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
        snapshot: Optional[TacticalSnapshot] = None,
        speeds: Optional[Dict[int, float]] = None,
        advanced_result: Optional[AdvancedTacticalResult] = None,
    ) -> np.ndarray:
        """
        Annote une frame avec toutes les informations visuelles.
        
        Args:
            frame: Image BGR
            detections: Détections de la frame
            snapshot: Snapshot tactique (optionnel)
            speeds: Dict {track_id: speed_kmh}
            advanced_result: Résultat de l'analyse avancée (optionnel)
            
        Returns:
            Frame annotée
        """
        annotated = frame.copy()

        # ─── Dessiner les joueurs ────────────────────────
        for det in detections.all_persons:
            self._draw_player(annotated, det, speeds)

        # ─── Dessiner le ballon ──────────────────────────
        if detections.ball:
            self._draw_ball(annotated, detections.ball)

        # ─── Traînes de mouvement ───────────────────────
        if self.config.show_trails:
            self._draw_trails(annotated, detections)

        # ─── HUD tactique ───────────────────────────────
        if snapshot:
            self._draw_tactical_hud(annotated, snapshot, advanced_result)

        return annotated

    def _draw_player(
        self, frame: np.ndarray, det: Detection,
        speeds: Optional[Dict[int, float]] = None
    ):
        """Dessine un joueur avec bbox, ID et vitesse."""
        color = self.get_team_color(det.team_id)
        x1, y1, x2, y2 = det.bbox.astype(int)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Étiquette
        label_parts = []
        if self.config.show_ids and det.track_id is not None:
            label_parts.append(f"#{det.track_id}")
        if self.config.show_speed and speeds and det.track_id in speeds:
            speed = speeds[det.track_id]
            label_parts.append(f"{speed:.1f}km/h")

        if label_parts:
            label = " ".join(label_parts)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Fond de l'étiquette
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0] + 6, y1),
                color, -1
            )

            # Texte
            cv2.putText(
                frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA
            )

        # Point au sol
        foot = det.bottom_center
        cv2.circle(frame, (int(foot[0]), int(foot[1])), 4, color, -1)

    def _draw_ball(self, frame: np.ndarray, det: Detection):
        """Dessine le ballon avec un cercle et un marqueur."""
        color = self.config.ball_color
        cx, cy = int(det.center[0]), int(det.center[1])
        radius = max(8, int(det.width / 2))

        # Cercle avec glow
        cv2.circle(frame, (cx, cy), radius + 4, (0, 100, 0), 2)
        cv2.circle(frame, (cx, cy), radius, color, -1)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1)

        # Triangle indicateur
        pts = np.array([
            [cx, cy - radius - 12],
            [cx - 6, cy - radius - 6],
            [cx + 6, cy - radius - 6],
        ], np.int32)
        cv2.fillPoly(frame, [pts], color)

    def _draw_trails(self, frame: np.ndarray, detections: FrameDetections):
        """Dessine les traînes de mouvement des joueurs."""
        for det in detections.all_persons:
            if det.track_id is None:
                continue

            tid = det.track_id
            foot = (int(det.bottom_center[0]), int(det.bottom_center[1]))

            if tid not in self.trails:
                self.trails[tid] = []
            self.trails[tid].append(foot)

            # Limiter la longueur
            if len(self.trails[tid]) > self.config.trail_length:
                self.trails[tid] = self.trails[tid][-self.config.trail_length:]

            # Dessiner la traîne avec dégradé d'opacité
            trail = self.trails[tid]
            color = self.get_team_color(det.team_id)

            for i in range(1, len(trail)):
                alpha = i / len(trail)  # 0 -> 1 (ancien -> récent)
                thickness = max(1, int(alpha * 3))
                faded_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, trail[i - 1], trail[i], faded_color, thickness,
                         cv2.LINE_AA)

    def _draw_tactical_hud(
        self, frame: np.ndarray, snapshot: TacticalSnapshot,
        advanced_result: Optional[AdvancedTacticalResult] = None,
    ):
        """Dessine le HUD tactique en haut de l'écran."""
        h, w = frame.shape[:2]

        # ─── Fond semi-transparent ──────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # ─── Phase de jeu ──────────────────────────────
        phase_a = snapshot.team_a_phase.value if snapshot.team_a_phase else "?"
        phase_b = snapshot.team_b_phase.value if snapshot.team_b_phase else "?"

        # Couleur selon la phase
        phase_color_a = self._phase_color(snapshot.team_a_phase)
        phase_color_b = self._phase_color(snapshot.team_b_phase)

        # Équipe A (gauche)
        cv2.putText(frame, f"Equipe A: {phase_a}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color_a, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Bloc: {snapshot.team_a_block_height_pct:.0f}%", (10, 55),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Territoire: {snapshot.team_a_territory_pct:.0f}%", (10, 78),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Équipe B (droite)
        cv2.putText(frame, f"Equipe B: {phase_b}", (w - 350, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color_b, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Bloc: {snapshot.team_b_block_height_pct:.0f}%", (w - 350, 55),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Territoire: {snapshot.team_b_territory_pct:.0f}%", (w - 350, 78),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Timer (centre)
        minutes = int(snapshot.timestamp_sec // 60)
        seconds = int(snapshot.timestamp_sec % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        text_size = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        cx = (w - text_size[0]) // 2
        cv2.putText(frame, time_str, (cx, 45),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 136), 2, cv2.LINE_AA)

        # Barre de pressing
        press_a = snapshot.pressing_intensity_a
        press_b = snapshot.pressing_intensity_b
        if press_a >= 3 or press_b >= 3:
            press_text = ""
            if press_a >= 3:
                press_text += f"PRESSING A: {press_a} joueurs  "
            if press_b >= 3:
                press_text += f"PRESSING B: {press_b} joueurs"
            cv2.putText(frame, press_text, (cx - 50, 80),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # ─── Bandeau avancé (bas de l'écran) ────────────
        if advanced_result is not None:
            self._draw_advanced_hud(frame, advanced_result)

    def _draw_advanced_hud(
        self, frame: np.ndarray, adv: AdvancedTacticalResult
    ):
        """Dessine un bandeau en bas avec Pass Availability, Offside, Space Control."""
        h, w = frame.shape[:2]
        bar_h = 32
        y0 = h - bar_h

        # Fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Pass Availability
        pa = adv.pass_lanes.pass_availability_pct
        pa_color = (0, 255, 136) if pa >= 50 else (0, 200, 255) if pa >= 30 else (0, 80, 255)
        cv2.putText(frame, f"Pass Avail: {pa:.0f}%", (10, y0 + 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, pa_color, 1, cv2.LINE_AA)

        open_n = adv.pass_lanes.open_lanes
        blocked_n = adv.pass_lanes.blocked_lanes
        cv2.putText(frame, f"({open_n} open / {blocked_n} blocked)", (200, y0 + 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Offside line indicator
        off_a = adv.offside_line
        if off_a.is_offside_position_a or off_a.is_offside_position_b:
            cv2.putText(frame, "OFFSIDE!", (w // 2 - 40, y0 + 22),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            stab_a = off_a.line_stability_a
            stab_b = off_a.line_stability_b
            cv2.putText(frame, f"Line Stab: A={stab_a:.1f}m B={stab_b:.1f}m",
                         (w // 2 - 100, y0 + 22),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Space Control
        sc_a = adv.space_control.team_a_pct
        sc_b = adv.space_control.team_b_pct
        sc_text = f"Space: A {sc_a:.0f}% | B {sc_b:.0f}%"
        cv2.putText(frame, sc_text, (w - 280, y0 + 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    @staticmethod
    def _phase_color(phase: GamePhase) -> Tuple[int, int, int]:
        """Retourne la couleur associée à une phase de jeu."""
        colors = {
            GamePhase.PRESSING_HAUT: (0, 0, 255),      # Rouge
            GamePhase.BLOC_HAUT: (0, 165, 255),         # Orange
            GamePhase.BLOC_MEDIAN: (0, 255, 255),       # Jaune
            GamePhase.BLOC_BAS: (0, 255, 0),            # Vert
            GamePhase.CONTRE_ATTAQUE: (255, 0, 255),    # Magenta
            GamePhase.POSSESSION: (255, 255, 0),        # Cyan
        }
        return colors.get(phase, (200, 200, 200))
