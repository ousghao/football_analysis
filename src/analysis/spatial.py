"""
Analyse spatiale avancée : Voronoi, Convex Hull, contrôle de l'espace.
Ce module est la mine d'or pour l'évaluation de la performance sans ballon.
"""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
from scipy.ndimage import uniform_filter
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from ..pitch.pitch_template import PITCH_LENGTH, PITCH_WIDTH


@dataclass
class SpatialMetrics:
    """Métriques spatiales d'une équipe à un instant t."""
    # Convex Hull
    hull_area: float = 0.0            # Surface couverte par le bloc (m²)
    hull_vertices: List[Tuple[float, float]] = field(default_factory=list)

    # Centroid
    centroid: Tuple[float, float] = (0.0, 0.0)   # Barycentre de l'équipe

    # Compacité
    spread: float = 0.0              # Écart-type des distances au centroïde
    max_width: float = 0.0           # Largeur max du bloc
    max_depth: float = 0.0           # Profondeur max du bloc
    inter_line_distance: float = 0.0  # Distance entre les lignes

    # Lignes
    defensive_line_y: float = 0.0    # Position de la ligne défensive (X en mètres)
    offensive_line_y: float = 0.0    # Position de la ligne offensive


@dataclass
class VoronoiResult:
    """Résultat de l'analyse de Voronoi."""
    team_a_control: float = 0.0      # % du terrain contrôlé par équipe A
    team_b_control: float = 0.0      # % du terrain contrôlé par équipe B
    player_areas: Dict[int, float] = field(default_factory=dict)  # track_id -> m²
    regions: Optional[Voronoi] = None


class SpatialAnalyzer:
    """
    Moteur d'analyse spatiale.
    
    Calcule les métriques de contrôle territorial, compacité,
    et évaluation des espaces libres via diagrammes de Voronoi.
    """

    def __init__(self, pitch_length: float = PITCH_LENGTH,
                 pitch_width: float = PITCH_WIDTH):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.pitch_area = pitch_length * pitch_width

    def compute_convex_hull(
        self, positions: List[Tuple[float, float]]
    ) -> Optional[Tuple[float, List[Tuple[float, float]]]]:
        """
        Calcule le Convex Hull (enveloppe convexe) d'un groupe de joueurs.
        
        C'est la surface couverte par le "bloc" de l'équipe.
        
        Args:
            positions: Liste de (x, y) en mètres
            
        Returns:
            (aire_m2, vertices) ou None si pas assez de points
        """
        if len(positions) < 3:
            return None

        points = np.array(positions)

        try:
            hull = ConvexHull(points)
            area = hull.volume  # En 2D, volume = aire
            vertices = [(points[v, 0], points[v, 1]) for v in hull.vertices]
            return (area, vertices)
        except Exception:
            return None

    def compute_team_metrics(
        self, positions: List[Tuple[float, float]],
        attack_direction: int = 1  # 1 = attaque vers la droite, -1 = gauche
    ) -> SpatialMetrics:
        """
        Calcule toutes les métriques spatiales d'une équipe.
        
        Args:
            positions: Positions des joueurs en mètres [(x, y), ...]
            attack_direction: Direction d'attaque (1=droite, -1=gauche)
            
        Returns:
            SpatialMetrics complètes
        """
        metrics = SpatialMetrics()

        if len(positions) < 2:
            return metrics

        points = np.array(positions)

        # ─── Centroïde ──────────────────────────────────
        metrics.centroid = (float(points[:, 0].mean()), float(points[:, 1].mean()))

        # ─── Convex Hull ────────────────────────────────
        hull_result = self.compute_convex_hull(positions)
        if hull_result:
            metrics.hull_area, metrics.hull_vertices = hull_result

        # ─── Compacité (spread) ─────────────────────────
        cx, cy = metrics.centroid
        distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        metrics.spread = float(distances.std())

        # ─── Dimensions du bloc ─────────────────────────
        metrics.max_width = float(points[:, 1].max() - points[:, 1].min())
        metrics.max_depth = float(points[:, 0].max() - points[:, 0].min())

        # ─── Lignes défensive et offensive ──────────────
        # La ligne défensive est la position X du défenseur le plus reculé
        if attack_direction == 1:
            metrics.defensive_line_y = float(points[:, 0].min())
            metrics.offensive_line_y = float(points[:, 0].max())
        else:
            metrics.defensive_line_y = float(points[:, 0].max())
            metrics.offensive_line_y = float(points[:, 0].min())

        # ─── Distance inter-lignes ──────────────────────
        # Diviser les joueurs en 3 lignes par position X
        sorted_x = np.sort(points[:, 0])
        n = len(sorted_x)
        if n >= 6:
            # Ligne défensive = 1/3 le plus bas, milieu, attaque = 1/3 le plus haut
            third = n // 3
            def_line = sorted_x[:third].mean()
            mid_line = sorted_x[third:2 * third].mean()
            att_line = sorted_x[2 * third:].mean()
            metrics.inter_line_distance = float(att_line - def_line)
        else:
            metrics.inter_line_distance = metrics.max_depth

        return metrics

    def compute_voronoi(
        self,
        team_a_positions: List[Tuple[int, float, float]],
        team_b_positions: List[Tuple[int, float, float]],
        grid_resolution: int = 100,
    ) -> VoronoiResult:
        """
        Calcule les diagrammes de Voronoi pour évaluer le contrôle territorial.
        
        Chaque point du terrain est attribué au joueur le plus proche.
        Le pourcentage du terrain contrôlé par chaque équipe est calculé.
        
        Args:
            team_a_positions: [(track_id, x, y), ...] en mètres
            team_b_positions: [(track_id, x, y), ...] en mètres
            grid_resolution: Résolution de la grille (plus haut = plus précis)
            
        Returns:
            VoronoiResult avec les pourcentages de contrôle
        """
        result = VoronoiResult()

        all_positions = team_a_positions + team_b_positions
        if len(all_positions) < 3:
            return result

        # Créer une grille de points sur le terrain
        x_grid = np.linspace(0, self.pitch_length, grid_resolution)
        y_grid = np.linspace(0, self.pitch_width, grid_resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Positions de tous les joueurs
        player_points = np.array([(x, y) for _, x, y in all_positions])
        player_ids = [tid for tid, _, _ in all_positions]
        team_labels = ([0] * len(team_a_positions) +
                       [1] * len(team_b_positions))

        # Pour chaque point de la grille, trouver le joueur le plus proche
        from scipy.spatial import cKDTree
        tree = cKDTree(player_points)
        _, nearest = tree.query(grid_points)

        # Compter les pixels contrôlés par chaque équipe
        team_control = np.array([team_labels[i] for i in nearest])
        total_points = len(grid_points)

        result.team_a_control = float(np.sum(team_control == 0) / total_points * 100)
        result.team_b_control = float(np.sum(team_control == 1) / total_points * 100)

        # Surface par joueur
        for idx, (tid, _, _) in enumerate(all_positions):
            player_mask = nearest == idx
            area_fraction = np.sum(player_mask) / total_points
            result.player_areas[tid] = float(area_fraction * self.pitch_area)

        # Calculer le Voronoi géométrique (pour visualisation)
        try:
            # Ajouter des points miroir aux bords pour limiter les régions
            mirror_points = self._add_mirror_points(player_points)
            all_pts = np.vstack([player_points, mirror_points])
            result.regions = Voronoi(all_pts)
        except Exception:
            pass

        return result

    def _add_mirror_points(self, points: np.ndarray) -> np.ndarray:
        """Ajoute des points miroir autour du terrain pour borner les régions Voronoi."""
        mirrors = []
        for p in points:
            mirrors.append([-p[0], p[1]])           # Miroir gauche
            mirrors.append([2 * self.pitch_length - p[0], p[1]])  # Miroir droit
            mirrors.append([p[0], -p[1]])            # Miroir haut
            mirrors.append([p[0], 2 * self.pitch_width - p[1]])   # Miroir bas
        return np.array(mirrors)

    def find_free_spaces(
        self,
        all_positions: List[Tuple[float, float]],
        grid_resolution: int = 50,
        min_distance: float = 10.0,
    ) -> List[Tuple[float, float, float]]:
        """
        Trouve les espaces libres sur le terrain (zones éloignées de tout joueur).
        
        Utile pour évaluer les appels de balle et les espaces exploitables.
        
        Returns:
            Liste de (x, y, distance_au_joueur_le_plus_proche) en mètres
        """
        if not all_positions:
            return []

        player_points = np.array(all_positions)

        x_grid = np.linspace(0, self.pitch_length, grid_resolution)
        y_grid = np.linspace(0, self.pitch_width, grid_resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        from scipy.spatial import cKDTree
        tree = cKDTree(player_points)
        distances, _ = tree.query(grid_points)

        # Garder les points avec distance > seuil
        free_mask = distances > min_distance
        free_points = grid_points[free_mask]
        free_distances = distances[free_mask]

        # Trier par distance décroissante (plus grand espace en premier)
        sort_idx = np.argsort(-free_distances)
        results = [
            (float(free_points[i, 0]), float(free_points[i, 1]),
             float(free_distances[i]))
            for i in sort_idx[:20]  # Top 20 espaces
        ]

        return results

    def compute_pressing_intensity(
        self,
        defending_positions: List[Tuple[float, float]],
        ball_position: Tuple[float, float],
        threshold_m: float = 10.0,
    ) -> Tuple[int, float]:
        """
        Calcule l'intensité du pressing autour du ballon.
        
        Args:
            defending_positions: Positions des défenseurs
            ball_position: Position du ballon en mètres
            threshold_m: Rayon de pressing en mètres
            
        Returns:
            (nombre_de_joueurs_en_pressing, distance_moyenne)
        """
        if not defending_positions:
            return (0, 0.0)

        ball = np.array(ball_position)
        positions = np.array(defending_positions)

        distances = np.sqrt(np.sum((positions - ball) ** 2, axis=1))
        pressing_mask = distances < threshold_m

        n_pressing = int(pressing_mask.sum())
        avg_dist = float(distances[pressing_mask].mean()) if n_pressing > 0 else 0.0

        return (n_pressing, avg_dist)
