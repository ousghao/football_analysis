"""
Minimap 2D (Bird's Eye View).
L'effet "Wow" : affiche une carte 2D en temps réel avec les positions des joueurs.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import VISUALIZATION, VisualizationConfig
from src.pitch.pitch_template import PitchTemplate, PITCH_LENGTH, PITCH_WIDTH
from src.analysis.tactical import TacticalSnapshot
from src.analysis.spatial import VoronoiResult
from src.analysis.advanced_tactical import AdvancedTacticalResult


class Minimap:
    """
    Rendu de la minimap 2D (vue de dessus) avec positions des joueurs.
    
    Fonctionnalités :
    - Positions des joueurs en temps réel (dots colorés par équipe)
    - Ballon
    - Convex Hull des deux équipes
    - Diagramme de Voronoi (optionnel)
    - Lignes défensives
    - Zones de pressing
    """

    def __init__(self, config: VisualizationConfig = None,
                 pitch: PitchTemplate = None):
        self.config = config or VISUALIZATION
        self.pitch = pitch or PitchTemplate()

        self.width = self.config.minimap_width
        self.height = self.config.minimap_height

        # Image de base du terrain (pré-rendue)
        self._base_pitch = self.pitch.render_minimap(self.width, self.height)

    def _to_px(self, world_point: Tuple[float, float]) -> Tuple[int, int]:
        """Convertit des coordonnées terrain (mètres) en pixels minimap."""
        x = int(world_point[0] / PITCH_LENGTH * self.width)
        y = int(world_point[1] / PITCH_WIDTH * self.height)
        return (max(0, min(self.width - 1, x)),
                max(0, min(self.height - 1, y)))

    def render(
        self,
        snapshot: TacticalSnapshot,
        show_hull: bool = True,
        show_voronoi: bool = False,
        show_defensive_lines: bool = True,
        voronoi_result: Optional[VoronoiResult] = None,
        advanced_result: Optional[AdvancedTacticalResult] = None,
    ) -> np.ndarray:
        """
        Rend la minimap pour un snapshot tactique donné.
        
        Args:
            snapshot: État tactique courant
            show_hull: Afficher le Convex Hull
            show_voronoi: Afficher les zones Voronoi
            show_defensive_lines: Afficher les lignes défensives
            
        Returns:
            Image BGR de la minimap
        """
        minimap = self._base_pitch.copy()

        # ─── Voronoi (en arrière-plan) ──────────────────
        if show_voronoi and voronoi_result:
            self._draw_voronoi(minimap, snapshot, voronoi_result)

        # ─── Convex Hull ────────────────────────────────
        if show_hull:
            self._draw_hull(minimap, snapshot.team_a_positions, self.config.team_a_color)
            self._draw_hull(minimap, snapshot.team_b_positions, self.config.team_b_color)

        # ─── Lignes défensives ──────────────────────────
        if show_defensive_lines:
            self._draw_defensive_line(minimap, snapshot.team_a_defensive_line,
                                       self.config.team_a_color)
            self._draw_defensive_line(minimap, snapshot.team_b_defensive_line,
                                       self.config.team_b_color)

        # ─── Joueurs Équipe A ───────────────────────────
        for tid, x, y in snapshot.team_a_positions:
            px = self._to_px((x, y))
            cv2.circle(minimap, px, 5, self.config.team_a_color, -1)
            cv2.circle(minimap, px, 5, (255, 255, 255), 1)
            if self.config.show_ids:
                cv2.putText(minimap, str(tid), (px[0] + 6, px[1] - 3),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # ─── Joueurs Équipe B ───────────────────────────
        for tid, x, y in snapshot.team_b_positions:
            px = self._to_px((x, y))
            cv2.circle(minimap, px, 5, self.config.team_b_color, -1)
            cv2.circle(minimap, px, 5, (255, 255, 255), 1)
            if self.config.show_ids:
                cv2.putText(minimap, str(tid), (px[0] + 6, px[1] - 3),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # ─── Ballon ─────────────────────────────────────
        if snapshot.ball_position:
            px = self._to_px(snapshot.ball_position)
            cv2.circle(minimap, px, 4, self.config.ball_color, -1)
            cv2.circle(minimap, px, 6, (255, 255, 255), 1)

        # ─── Centroïdes ─────────────────────────────────
        if snapshot.team_a_centroid != (0, 0):
            px = self._to_px(snapshot.team_a_centroid)
            cv2.drawMarker(minimap, px, self.config.team_a_color,
                           cv2.MARKER_CROSS, 10, 2)

        if snapshot.team_b_centroid != (0, 0):
            px = self._to_px(snapshot.team_b_centroid)
            cv2.drawMarker(minimap, px, self.config.team_b_color,
                           cv2.MARKER_CROSS, 10, 2)
        # ─── Analyse avancée (Offside + Pass lanes) ───
        if advanced_result is not None:
            self._draw_offside_lines(minimap, advanced_result)
            self._draw_pass_lanes(minimap, advanced_result)
        # ─── Bordure ────────────────────────────────────
        cv2.rectangle(minimap, (0, 0), (self.width - 1, self.height - 1),
                       (100, 100, 100), 1)

        return minimap

    def _draw_hull(
        self, minimap: np.ndarray,
        positions: List[Tuple[int, float, float]],
        color: Tuple[int, int, int],
    ):
        """Dessine le Convex Hull d'une équipe."""
        if len(positions) < 3:
            return

        from scipy.spatial import ConvexHull

        points = np.array([(x, y) for _, x, y in positions])
        try:
            hull = ConvexHull(points)
            hull_pts = [self._to_px((points[v, 0], points[v, 1]))
                        for v in hull.vertices]
            hull_pts = np.array(hull_pts, dtype=np.int32)

            # Fond semi-transparent
            overlay = minimap.copy()
            cv2.fillPoly(overlay, [hull_pts], color)
            cv2.addWeighted(overlay, 0.15, minimap, 0.85, 0, minimap)

            # Contour
            cv2.polylines(minimap, [hull_pts], True, color, 1, cv2.LINE_AA)
        except Exception:
            pass

    def _draw_defensive_line(
        self, minimap: np.ndarray,
        line_x_meters: float,
        color: Tuple[int, int, int],
    ):
        """Dessine une ligne défensive horizontale."""
        if line_x_meters <= 0:
            return

        px_x = int(line_x_meters / PITCH_LENGTH * self.width)
        cv2.line(minimap, (px_x, 0), (px_x, self.height),
                 color, 1, cv2.LINE_AA)

    def _draw_voronoi(
        self, minimap: np.ndarray,
        snapshot: TacticalSnapshot,
        voronoi_result: VoronoiResult,
    ):
        """Dessine les zones de Voronoi en arrière-plan."""
        all_positions = snapshot.team_a_positions + snapshot.team_b_positions
        if len(all_positions) < 3:
            return

        # Grille de pixels
        from scipy.spatial import cKDTree

        player_points = np.array([(x, y) for _, x, y in all_positions])
        team_labels = ([0] * len(snapshot.team_a_positions) +
                       [1] * len(snapshot.team_b_positions))

        # Pour chaque pixel de la minimap
        x_grid = np.linspace(0, PITCH_LENGTH, self.width)
        y_grid = np.linspace(0, PITCH_WIDTH, self.height)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

        tree = cKDTree(player_points)
        _, nearest = tree.query(grid_pts)

        team_map = np.array([team_labels[i] for i in nearest]).reshape(
            self.height, self.width
        )

        # Overlay couleur
        overlay = minimap.copy()
        color_a = np.array(self.config.team_a_color)
        color_b = np.array(self.config.team_b_color)

        mask_a = team_map == 0
        mask_b = team_map == 1
        overlay[mask_a] = (overlay[mask_a] * 0.7 + color_a * 0.3).astype(np.uint8)
        overlay[mask_b] = (overlay[mask_b] * 0.7 + color_b * 0.3).astype(np.uint8)

        cv2.addWeighted(overlay, 0.5, minimap, 0.5, 0, minimap)

    def _draw_offside_lines(
        self, minimap: np.ndarray, adv: AdvancedTacticalResult
    ):
        """Dessine les lignes de hors-jeu en pointillés sur la minimap."""
        offside = adv.offside_line

        # Ligne offside Équipe A (défend à gauche)
        if offside.team_a_offside_x > 0:
            px_x = int(offside.team_a_offside_x / PITCH_LENGTH * self.width)
            # Pointillés bleus
            for y in range(0, self.height, 8):
                y_end = min(y + 4, self.height)
                cv2.line(minimap, (px_x, y), (px_x, y_end),
                         self.config.team_a_color, 2, cv2.LINE_AA)

        # Ligne offside Équipe B (défend à droite)
        if offside.team_b_offside_x > 0:
            px_x = int(offside.team_b_offside_x / PITCH_LENGTH * self.width)
            for y in range(0, self.height, 8):
                y_end = min(y + 4, self.height)
                cv2.line(minimap, (px_x, y), (px_x, y_end),
                         self.config.team_b_color, 2, cv2.LINE_AA)

        # Indicateur OFFSIDE si joueur en position de hors-jeu
        if offside.is_offside_position_a or offside.is_offside_position_b:
            cv2.putText(minimap, "OFFSIDE", (self.width // 2 - 25, 12),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

    def _draw_pass_lanes(
        self, minimap: np.ndarray, adv: AdvancedTacticalResult
    ):
        """Dessine les lignes de passe (vert = ouvert, rouge = bloqué)."""
        pl = adv.pass_lanes
        if not pl.lanes:
            return

        for lane in pl.lanes:
            pt1 = self._to_px(lane.from_pos)
            pt2 = self._to_px(lane.to_pos)
            color = (0, 200, 0) if lane.is_open else (0, 0, 200)
            thickness = 2 if lane.is_open else 1
            cv2.line(minimap, pt1, pt2, color, thickness, cv2.LINE_AA)

            # Petite flèche au milieu pour la direction
            mx = (pt1[0] + pt2[0]) // 2
            my = (pt1[1] + pt2[1]) // 2
            cv2.circle(minimap, (mx, my), 2, color, -1)

    def overlay_on_frame(
        self, frame: np.ndarray, minimap: np.ndarray,
        position: str = "bottom-right", margin: int = 15,
    ) -> np.ndarray:
        """
        Superpose la minimap sur la frame vidéo.
        
        Args:
            frame: Image BGR de la frame principale
            minimap: Image BGR de la minimap
            position: "bottom-right", "bottom-left", "top-right", "top-left"
            margin: Marge en pixels
            
        Returns:
            Frame avec minimap
        """
        fh, fw = frame.shape[:2]
        mh, mw = minimap.shape[:2]

        # Position
        if position == "bottom-right":
            x = fw - mw - margin
            y = fh - mh - margin
        elif position == "bottom-left":
            x = margin
            y = fh - mh - margin
        elif position == "top-right":
            x = fw - mw - margin
            y = margin
        elif position == "top-left":
            x = margin
            y = margin
        else:
            x = fw - mw - margin
            y = fh - mh - margin

        result = frame.copy()

        # Fond semi-transparent pour lisibilité
        overlay = result.copy()
        cv2.rectangle(overlay, (x - 2, y - 2),
                       (x + mw + 2, y + mh + 2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.config.minimap_opacity,
                         result, 1 - self.config.minimap_opacity, 0, result)

        # Placer la minimap
        result[y:y + mh, x:x + mw] = minimap

        # Bordure
        cv2.rectangle(result, (x - 1, y - 1),
                       (x + mw, y + mh), (0, 255, 136), 1)

        return result
