"""
Analyse tactique : métriques de match, hauteur de bloc, phases de jeu.
Transforme les données brutes en intelligence tactique.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import TACTICAL, TacticalConfig
from ..pitch.pitch_template import PITCH_LENGTH, PITCH_WIDTH


class GamePhase(Enum):
    """Phases de jeu identifiables."""
    PRESSING_HAUT = "Pressing Haut"
    BLOC_HAUT = "Bloc Haut"
    BLOC_MEDIAN = "Bloc Médian"
    BLOC_BAS = "Bloc Bas"
    CONTRE_ATTAQUE = "Contre-Attaque"
    POSSESSION = "Possession"
    TRANSITION_OFF = "Transition Offensive"
    TRANSITION_DEF = "Transition Défensive"
    UNKNOWN = "Inconnu"


@dataclass
class TacticalSnapshot:
    """État tactique à un instant donné."""
    frame_idx: int
    timestamp_sec: float

    # Positions
    team_a_positions: List[Tuple[int, float, float]] = field(default_factory=list)
    team_b_positions: List[Tuple[int, float, float]] = field(default_factory=list)
    ball_position: Optional[Tuple[float, float]] = None

    # Métriques Équipe A
    team_a_centroid: Tuple[float, float] = (0.0, 0.0)
    team_a_hull_area: float = 0.0
    team_a_defensive_line: float = 0.0
    team_a_block_height_pct: float = 0.0
    team_a_spread: float = 0.0
    team_a_width: float = 0.0

    # Métriques Équipe B
    team_b_centroid: Tuple[float, float] = (0.0, 0.0)
    team_b_hull_area: float = 0.0
    team_b_defensive_line: float = 0.0
    team_b_block_height_pct: float = 0.0
    team_b_spread: float = 0.0
    team_b_width: float = 0.0

    # Voronoi
    team_a_territory_pct: float = 50.0
    team_b_territory_pct: float = 50.0

    # Phases
    team_a_phase: GamePhase = GamePhase.UNKNOWN
    team_b_phase: GamePhase = GamePhase.UNKNOWN

    # Pressing
    pressing_intensity_a: int = 0    # Nb joueurs A en pressing
    pressing_intensity_b: int = 0


class TacticalAnalyzer:
    """
    Le "Cerveau" du système.
    
    Analyse chaque frame pour produire des métriques tactiques :
    - Hauteur de bloc (Haut/Médian/Bas)
    - Phase de jeu (Pressing, Possession, Transition...)
    - Compacité et organisation défensive
    - Intensité du pressing
    - Domination territoriale
    """

    def __init__(self, config: TacticalConfig = None, fps: float = 30.0):
        self.config = config or TACTICAL
        self.fps = fps
        self.history: List[TacticalSnapshot] = []
        self._phase_buffer_a: deque = deque(maxlen=int(fps * 2))  # 2 sec buffer
        self._phase_buffer_b: deque = deque(maxlen=int(fps * 2))

    def analyze_frame(
        self,
        frame_idx: int,
        team_a_positions: List[Tuple[int, float, float]],
        team_b_positions: List[Tuple[int, float, float]],
        ball_position: Optional[Tuple[float, float]] = None,
        spatial_analyzer=None,
    ) -> TacticalSnapshot:
        """
        Analyse tactique complète d'une frame.
        
        Args:
            frame_idx: Index de la frame
            team_a_positions: [(track_id, x_m, y_m), ...] Équipe A
            team_b_positions: [(track_id, x_m, y_m), ...] Équipe B
            ball_position: Position du ballon en mètres
            spatial_analyzer: Instance de SpatialAnalyzer
            
        Returns:
            TacticalSnapshot avec toutes les métriques
        """
        snapshot = TacticalSnapshot(
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / self.fps,
            team_a_positions=team_a_positions,
            team_b_positions=team_b_positions,
            ball_position=ball_position,
        )

        # ─── Métriques spatiales ────────────────────────
        if spatial_analyzer:
            pos_a = [(x, y) for _, x, y in team_a_positions]
            pos_b = [(x, y) for _, x, y in team_b_positions]

            if len(pos_a) >= 3:
                metrics_a = spatial_analyzer.compute_team_metrics(pos_a, attack_direction=1)
                snapshot.team_a_centroid = metrics_a.centroid
                snapshot.team_a_hull_area = metrics_a.hull_area
                snapshot.team_a_defensive_line = metrics_a.defensive_line_y
                snapshot.team_a_spread = metrics_a.spread
                snapshot.team_a_width = metrics_a.max_width

            if len(pos_b) >= 3:
                metrics_b = spatial_analyzer.compute_team_metrics(pos_b, attack_direction=-1)
                snapshot.team_b_centroid = metrics_b.centroid
                snapshot.team_b_hull_area = metrics_b.hull_area
                snapshot.team_b_defensive_line = metrics_b.defensive_line_y
                snapshot.team_b_spread = metrics_b.spread
                snapshot.team_b_width = metrics_b.max_width

            # Voronoi
            if len(team_a_positions) >= 2 and len(team_b_positions) >= 2:
                voronoi = spatial_analyzer.compute_voronoi(
                    team_a_positions, team_b_positions
                )
                snapshot.team_a_territory_pct = voronoi.team_a_control
                snapshot.team_b_territory_pct = voronoi.team_b_control

        # ─── Hauteur de bloc ────────────────────────────
        snapshot.team_a_block_height_pct = self._compute_block_height(
            team_a_positions, attack_direction=1
        )
        snapshot.team_b_block_height_pct = self._compute_block_height(
            team_b_positions, attack_direction=-1
        )

        # ─── Pressing ──────────────────────────────────
        if ball_position and spatial_analyzer:
            pos_a = [(x, y) for _, x, y in team_a_positions]
            pos_b = [(x, y) for _, x, y in team_b_positions]

            n_a, _ = spatial_analyzer.compute_pressing_intensity(
                pos_a, ball_position, self.config.pressing_distance_m
            )
            n_b, _ = spatial_analyzer.compute_pressing_intensity(
                pos_b, ball_position, self.config.pressing_distance_m
            )
            snapshot.pressing_intensity_a = n_a
            snapshot.pressing_intensity_b = n_b

        # ─── Phases de jeu ─────────────────────────────
        snapshot.team_a_phase = self._detect_phase(
            snapshot.team_a_block_height_pct,
            snapshot.pressing_intensity_a,
            is_team_a=True,
        )
        snapshot.team_b_phase = self._detect_phase(
            snapshot.team_b_block_height_pct,
            snapshot.pressing_intensity_b,
            is_team_a=False,
        )

        self.history.append(snapshot)
        return snapshot

    def _compute_block_height(
        self,
        positions: List[Tuple[int, float, float]],
        attack_direction: int = 1,
    ) -> float:
        """
        Calcule la hauteur du bloc en pourcentage du terrain.
        
        0% = dans les buts propres, 100% = dans les buts adverses
        """
        if len(positions) < 2:
            return 50.0

        x_positions = [x for _, x, _ in positions]
        centroid_x = np.mean(x_positions)

        if attack_direction == 1:
            # Attaque vers la droite: 0=buts propres, 105=buts adverses
            return float(centroid_x / PITCH_LENGTH * 100)
        else:
            # Attaque vers la gauche: 105=buts propres, 0=buts adverses
            return float((1 - centroid_x / PITCH_LENGTH) * 100)

    def _detect_phase(
        self,
        block_height_pct: float,
        pressing_count: int,
        is_team_a: bool,
    ) -> GamePhase:
        """
        Détecte la phase de jeu actuelle d'une équipe.
        """
        buffer = self._phase_buffer_a if is_team_a else self._phase_buffer_b

        # Logique de détection
        if block_height_pct > self.config.high_block_threshold * 100:
            if pressing_count >= self.config.pressing_min_players:
                phase = GamePhase.PRESSING_HAUT
            else:
                phase = GamePhase.BLOC_HAUT
        elif block_height_pct > self.config.mid_block_threshold * 100:
            phase = GamePhase.BLOC_MEDIAN
        else:
            phase = GamePhase.BLOC_BAS

        buffer.append(phase)

        # Stabilisation : phase majoritaire sur le buffer
        if len(buffer) > 0:
            from collections import Counter
            phase_counts = Counter(buffer)
            phase = phase_counts.most_common(1)[0][0]

        return phase

    def get_period_summary(
        self, start_frame: int = 0, end_frame: int = -1
    ) -> Dict:
        """
        Génère un résumé tactique pour une période donnée.
        
        C'est le rapport automatique : "L'équipe a passé 65% du temps en bloc médian..."
        """
        if end_frame == -1:
            end_frame = len(self.history)

        period = [s for s in self.history if start_frame <= s.frame_idx < end_frame]

        if not period:
            return {}

        # ─── Distribution des phases ────────────────────
        from collections import Counter
        phases_a = Counter(s.team_a_phase for s in period)
        phases_b = Counter(s.team_b_phase for s in period)
        total = len(period)

        phase_dist_a = {
            phase.value: round(count / total * 100, 1)
            for phase, count in phases_a.most_common()
        }
        phase_dist_b = {
            phase.value: round(count / total * 100, 1)
            for phase, count in phases_b.most_common()
        }

        # ─── Moyennes ──────────────────────────────────
        avg_hull_a = np.mean([s.team_a_hull_area for s in period if s.team_a_hull_area > 0])
        avg_hull_b = np.mean([s.team_b_hull_area for s in period if s.team_b_hull_area > 0])
        avg_territory_a = np.mean([s.team_a_territory_pct for s in period])
        avg_territory_b = np.mean([s.team_b_territory_pct for s in period])
        avg_block_a = np.mean([s.team_a_block_height_pct for s in period])
        avg_block_b = np.mean([s.team_b_block_height_pct for s in period])
        avg_spread_a = np.mean([s.team_a_spread for s in period if s.team_a_spread > 0])
        avg_spread_b = np.mean([s.team_b_spread for s in period if s.team_b_spread > 0])

        # ─── Pressing ──────────────────────────────────
        intense_pressing_a = sum(
            1 for s in period
            if s.pressing_intensity_a >= self.config.pressing_min_players
        )
        intense_pressing_b = sum(
            1 for s in period
            if s.pressing_intensity_b >= self.config.pressing_min_players
        )

        duration_sec = len(period) / self.fps

        summary = {
            "duration_seconds": round(duration_sec, 1),
            "duration_minutes": round(duration_sec / 60, 1),
            "frames_analyzed": len(period),

            "team_a": {
                "phase_distribution": phase_dist_a,
                "avg_block_height_pct": round(float(avg_block_a), 1),
                "avg_hull_area_m2": round(float(avg_hull_a), 1),
                "avg_territory_pct": round(float(avg_territory_a), 1),
                "avg_spread_m": round(float(avg_spread_a), 1),
                "intense_pressing_frames": intense_pressing_a,
                "intense_pressing_pct": round(intense_pressing_a / total * 100, 1),
            },

            "team_b": {
                "phase_distribution": phase_dist_b,
                "avg_block_height_pct": round(float(avg_block_b), 1),
                "avg_hull_area_m2": round(float(avg_hull_b), 1),
                "avg_territory_pct": round(float(avg_territory_b), 1),
                "avg_spread_m": round(float(avg_spread_b), 1),
                "intense_pressing_frames": intense_pressing_b,
                "intense_pressing_pct": round(intense_pressing_b / total * 100, 1),
            },
        }

        return summary
