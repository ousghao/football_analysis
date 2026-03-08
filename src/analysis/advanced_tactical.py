"""
Analyse Tactique Avancée — Space Control, Offside Line, Pass Lanes.

Trois modules d'intelligence tactique de niveau professionnel :

1. SpaceControlAnalyzer  — Voronoi par joueur avec surface m², gaps défensifs
2. OffsideLineTracker    — Ligne de hors-jeu dynamique, stabilité, profondeur
3. PassLaneAnalyzer      — Lignes de passe ouvertes/bloquées, Pass Availability %

Ces métriques sont celles utilisées par les départements d'analyse
de clubs comme le Barça, Liverpool ou le Real Madrid.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque

from scipy.spatial import cKDTree

from ..pitch.pitch_template import PITCH_LENGTH, PITCH_WIDTH


# ═══════════════════════════════════════════════════════════════
#  1. SPACE CONTROL — Voronoi par joueur
# ═══════════════════════════════════════════════════════════════

@dataclass
class PlayerSpaceControl:
    """Contrôle d'espace d'un joueur individuel."""
    track_id: int
    team_id: int                       # 0=A, 1=B
    area_m2: float = 0.0              # Surface Voronoi en m²
    area_pct: float = 0.0             # % du terrain total
    position: Tuple[float, float] = (0.0, 0.0)


@dataclass
class SpaceControlResult:
    """Résultat complet du Space Control."""
    # Par joueur
    player_spaces: List[PlayerSpaceControl] = field(default_factory=list)

    # Par équipe
    team_a_total_m2: float = 0.0
    team_b_total_m2: float = 0.0
    team_a_pct: float = 50.0
    team_b_pct: float = 50.0

    # Top/Bottom joueurs
    biggest_space_player: Optional[int] = None       # track_id
    biggest_space_area: float = 0.0
    smallest_space_player: Optional[int] = None
    smallest_space_area: float = float("inf")

    # Gaps défensifs (zones > seuil sans défenseur)
    defensive_gaps_a: List[Tuple[float, float, float]] = field(default_factory=list)
    defensive_gaps_b: List[Tuple[float, float, float]] = field(default_factory=list)

    # Contrôle de la surface de réparation
    penalty_area_control_a: float = 0.0   # % of own penalty area controlled
    penalty_area_control_b: float = 0.0

    # Grille Voronoi (pour visualisation)
    control_grid: Optional[np.ndarray] = None   # shape (H, W), values = player_index


class SpaceControlAnalyzer:
    """
    Calcule le contrôle d'espace par joueur via Voronoi pondéré.

    Chaque point du terrain est attribué au joueur qui peut l'atteindre
    le plus rapidement (distance euclidienne simple pour l'instant).

    Génère une grille de contrôle utilisable pour la visualisation
    et des métriques par joueur (surface, % du terrain).
    """

    def __init__(self, grid_resolution: int = 80):
        self.grid_res = grid_resolution
        self.pitch_area = PITCH_LENGTH * PITCH_WIDTH

        # Surface de réparation (coordonnées FIFA)
        # Gauche : x ∈ [0, 16.5], y ∈ [13.84, 54.16]
        self.penalty_left = (0, 16.5, 13.84, 54.16)
        # Droite : x ∈ [88.5, 105], y ∈ [13.84, 54.16]
        self.penalty_right = (88.5, 105.0, 13.84, 54.16)

    def analyze(
        self,
        team_a: List[Tuple[int, float, float]],
        team_b: List[Tuple[int, float, float]],
    ) -> SpaceControlResult:
        """
        Analyse complète du space control.

        Args:
            team_a: [(track_id, x_m, y_m), ...]
            team_b: [(track_id, x_m, y_m), ...]

        Returns:
            SpaceControlResult avec métriques par joueur et par équipe
        """
        result = SpaceControlResult()
        all_players = team_a + team_b
        if len(all_players) < 2:
            return result

        # Labels : 0=team_a, 1=team_b
        team_labels = [0] * len(team_a) + [1] * len(team_b)
        positions = np.array([(x, y) for _, x, y in all_players])
        track_ids = [tid for tid, _, _ in all_players]

        # Grille du terrain
        x_grid = np.linspace(0, PITCH_LENGTH, self.grid_res)
        y_grid = np.linspace(0, PITCH_WIDTH, self.grid_res)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

        # KDTree → joueur le plus proche de chaque point
        tree = cKDTree(positions)
        _, nearest_idx = tree.query(grid_pts)

        total_pts = len(grid_pts)
        cell_area = self.pitch_area / total_pts

        # Grille de contrôle pour visualisation
        result.control_grid = nearest_idx.reshape(len(y_grid), len(x_grid))

        # Surface par joueur
        for i, (tid, x, y) in enumerate(all_players):
            count = np.sum(nearest_idx == i)
            area = count * cell_area
            pct = count / total_pts * 100

            ps = PlayerSpaceControl(
                track_id=tid,
                team_id=team_labels[i],
                area_m2=area,
                area_pct=pct,
                position=(x, y),
            )
            result.player_spaces.append(ps)

            # Top / Bottom
            if area > result.biggest_space_area:
                result.biggest_space_area = area
                result.biggest_space_player = tid
            if area < result.smallest_space_area:
                result.smallest_space_area = area
                result.smallest_space_player = tid

        # Totaux par équipe
        result.team_a_total_m2 = sum(
            ps.area_m2 for ps in result.player_spaces if ps.team_id == 0
        )
        result.team_b_total_m2 = sum(
            ps.area_m2 for ps in result.player_spaces if ps.team_id == 1
        )
        result.team_a_pct = result.team_a_total_m2 / self.pitch_area * 100
        result.team_b_pct = result.team_b_total_m2 / self.pitch_area * 100

        # Contrôle de la surface de réparation
        result.penalty_area_control_a = self._penalty_area_control(
            grid_pts, nearest_idx, team_labels, self.penalty_left, target_team=0
        )
        result.penalty_area_control_b = self._penalty_area_control(
            grid_pts, nearest_idx, team_labels, self.penalty_right, target_team=1
        )

        # Gaps défensifs (zones vides > 12m du défenseur le plus proche)
        result.defensive_gaps_a = self._find_gaps(team_a, half="left")
        result.defensive_gaps_b = self._find_gaps(team_b, half="right")

        return result

    def _penalty_area_control(
        self, grid_pts, nearest_idx, team_labels, rect, target_team: int
    ) -> float:
        """% de la surface de réparation contrôlée par l'équipe target."""
        x_min, x_max, y_min, y_max = rect
        mask = (
            (grid_pts[:, 0] >= x_min) & (grid_pts[:, 0] <= x_max) &
            (grid_pts[:, 1] >= y_min) & (grid_pts[:, 1] <= y_max)
        )
        if mask.sum() == 0:
            return 0.0
        in_rect = nearest_idx[mask]
        controlled = sum(1 for idx in in_rect if team_labels[idx] == target_team)
        return controlled / mask.sum() * 100

    def _find_gaps(
        self,
        team_positions: List[Tuple[int, float, float]],
        half: str = "left",
        gap_threshold: float = 12.0,
    ) -> List[Tuple[float, float, float]]:
        """Trouve les trous dans le dispositif défensif."""
        if len(team_positions) < 3:
            return []

        positions = np.array([(x, y) for _, x, y in team_positions])

        # Ne scanner que la moitié défensive
        if half == "left":
            x_range = np.linspace(0, PITCH_LENGTH / 2, 30)
        else:
            x_range = np.linspace(PITCH_LENGTH / 2, PITCH_LENGTH, 30)
        y_range = np.linspace(5, PITCH_WIDTH - 5, 20)

        xx, yy = np.meshgrid(x_range, y_range)
        scan_pts = np.column_stack([xx.ravel(), yy.ravel()])

        tree = cKDTree(positions)
        distances, _ = tree.query(scan_pts)

        gap_mask = distances > gap_threshold
        gaps = []
        for pt, dist in zip(scan_pts[gap_mask], distances[gap_mask]):
            gaps.append((float(pt[0]), float(pt[1]), float(dist)))

        # Trier par distance décroissante, top 5
        gaps.sort(key=lambda g: -g[2])
        return gaps[:5]


# ═══════════════════════════════════════════════════════════════
#  2. OFFSIDE LINE TRACKER — Ligne de hors-jeu dynamique
# ═══════════════════════════════════════════════════════════════

@dataclass
class OffsideLineResult:
    """Résultat de l'analyse de la ligne de hors-jeu."""
    # Lignes défensives (position X en mètres du dernier défenseur)
    team_a_offside_x: float = 0.0      # Ligne de hors-jeu imposée par A
    team_b_offside_x: float = 0.0      # Ligne de hors-jeu imposée par B

    # Profondeur : distance entre ligne et porteur de balle adverse
    depth_vs_ball_a: float = 0.0       # Distance (m) ligne A ↔ balle
    depth_vs_ball_b: float = 0.0

    # Stabilité de la ligne (écart-type sur N frames)
    line_stability_a: float = 0.0      # Faible = bien organisé
    line_stability_b: float = 0.0      # Élevé = désorganisé

    # Nb de joueurs sur la même ligne (±2m)
    defenders_online_a: int = 0
    defenders_online_b: int = 0

    # Joueur le plus avancé (potentiellement hors-jeu)
    furthest_attacker_a: Optional[int] = None  # track_id du joueur A le + avancé
    furthest_attacker_b: Optional[int] = None
    is_offside_position_a: bool = False         # Ce joueur est-il devant la ligne B?
    is_offside_position_b: bool = False


class OffsideLineTracker:
    """
    Suit la ligne de hors-jeu en temps réel.

    La ligne de hors-jeu = position X de l'avant-dernier défenseur
    (le dernier étant le gardien, qu'on exclut ici en prenant le 2ème
    joueur le plus reculé — mais comme on ne détecte pas le gardien
    de manière fiable, on prend le dernier joueur de champ).

    Calcule aussi :
    - La stabilité de la ligne (Line Height Stability)
    - La profondeur entre la ligne et le ballon
    - La détection de situations de hors-jeu
    """

    def __init__(self, fps: float = 25.0, history_window: float = 3.0):
        self.fps = fps
        window_size = max(3, int(fps * history_window))
        self.line_history_a: deque = deque(maxlen=window_size)
        self.line_history_b: deque = deque(maxlen=window_size)

    def analyze(
        self,
        team_a: List[Tuple[int, float, float]],
        team_b: List[Tuple[int, float, float]],
        ball_position: Optional[Tuple[float, float]] = None,
    ) -> OffsideLineResult:
        """
        Analyse la ligne de hors-jeu pour les deux équipes.

        Convention : Équipe A attaque vers la droite (X croissant),
                     Équipe B attaque vers la gauche (X décroissant).
        """
        result = OffsideLineResult()

        # ── Équipe A : défend à gauche, dernier défenseur = X min
        if len(team_a) >= 2:
            xs_a = sorted([(x, tid) for tid, x, y in team_a])
            # Dernier défenseur = X le plus bas (hors gardien)
            # On prend le 2ème plus bas si possible (le 1er est peut-être le GK)
            if len(xs_a) >= 2:
                result.team_a_offside_x = xs_a[1][0]  # 2ème plus reculé
            else:
                result.team_a_offside_x = xs_a[0][0]

            # Compteur de défenseurs alignés (même X ± 2m)
            line_x = result.team_a_offside_x
            result.defenders_online_a = sum(
                1 for tid, x, y in team_a if abs(x - line_x) < 2.0
            )

        # ── Équipe B : défend à droite, dernier défenseur = X max
        if len(team_b) >= 2:
            xs_b = sorted([(x, tid) for tid, x, y in team_b], reverse=True)
            if len(xs_b) >= 2:
                result.team_b_offside_x = xs_b[1][0]
            else:
                result.team_b_offside_x = xs_b[0][0]

            line_x = result.team_b_offside_x
            result.defenders_online_b = sum(
                1 for tid, x, y in team_b if abs(x - line_x) < 2.0
            )

        # ── Profondeur vs ballon ────────────────────────
        if ball_position:
            bx = ball_position[0]
            result.depth_vs_ball_a = abs(bx - result.team_a_offside_x)
            result.depth_vs_ball_b = abs(bx - result.team_b_offside_x)

        # ── Joueur le plus avancé (hors-jeu potentiel) ──
        if team_a:
            furthest_a = max(team_a, key=lambda p: p[1])  # X max = le + avancé vers la droite
            result.furthest_attacker_a = furthest_a[0]
            # Est-il devant la ligne défensive de B ?
            if result.team_b_offside_x > 0:
                result.is_offside_position_a = furthest_a[1] > result.team_b_offside_x

        if team_b:
            furthest_b = min(team_b, key=lambda p: p[1])  # X min = le + avancé vers la gauche
            result.furthest_attacker_b = furthest_b[0]
            if result.team_a_offside_x > 0:
                result.is_offside_position_b = furthest_b[1] < result.team_a_offside_x

        # ── Stabilité de la ligne (écart-type sur fenêtre glissante) ──
        self.line_history_a.append(result.team_a_offside_x)
        self.line_history_b.append(result.team_b_offside_x)

        if len(self.line_history_a) >= 3:
            result.line_stability_a = float(np.std(list(self.line_history_a)))
        if len(self.line_history_b) >= 3:
            result.line_stability_b = float(np.std(list(self.line_history_b)))

        return result


# ═══════════════════════════════════════════════════════════════
#  3. PASS LANES — Lignes de passe ouvertes / bloquées
# ═══════════════════════════════════════════════════════════════

@dataclass
class PassLane:
    """Une ligne de passe entre le porteur du ballon et un coéquipier."""
    from_id: int                          # track_id du porteur
    to_id: int                            # track_id du receveur
    from_pos: Tuple[float, float] = (0.0, 0.0)
    to_pos: Tuple[float, float] = (0.0, 0.0)
    distance: float = 0.0                # Distance en mètres
    is_open: bool = True                  # Ligne dégagée ?
    blocking_player: Optional[int] = None # track_id du bloqueur (si fermée)
    min_clearance: float = float("inf")  # Dist min au défenseur le + proche du segment


@dataclass
class PassLaneResult:
    """Résultat complet de l'analyse des lignes de passe."""
    ball_carrier: Optional[int] = None      # track_id
    ball_carrier_team: Optional[int] = None # 0=A, 1=B

    lanes: List[PassLane] = field(default_factory=list)

    # Métriques agrégées
    total_lanes: int = 0
    open_lanes: int = 0
    blocked_lanes: int = 0
    pass_availability_pct: float = 0.0      # % de lignes ouvertes

    # Joueurs les mieux / pire situés
    best_receiver: Optional[int] = None     # Meilleur coéquipier disponible
    most_isolated: Optional[int] = None     # Coéquipier le + bloqué/isolé


class PassLaneAnalyzer:
    """
    Analyse les lignes de passe entre le porteur du ballon et ses coéquipiers.

    Pour chaque coéquipier, trace un segment porteur → receveur et vérifie
    si un adversaire se trouve dans un couloir autour de ce segment.

    Largeur du couloir : 1.5m (un joueur qui est à < 1.5m de la ligne
    de passe la bloque visuellement/physiquement).
    """

    def __init__(self, blocking_radius: float = 1.5, max_pass_distance: float = 40.0):
        self.blocking_radius = blocking_radius
        self.max_pass_distance = max_pass_distance

    def analyze(
        self,
        team_a: List[Tuple[int, float, float]],
        team_b: List[Tuple[int, float, float]],
        ball_position: Optional[Tuple[float, float]] = None,
    ) -> PassLaneResult:
        """
        Analyse toutes les lignes de passe depuis le porteur du ballon.

        Le porteur est le joueur le plus proche du ballon.
        Les receveurs sont ses coéquipiers.
        Les bloqueurs sont les adversaires.
        """
        result = PassLaneResult()

        if ball_position is None or (len(team_a) < 2 and len(team_b) < 2):
            return result

        # ── Identifier le porteur du ballon ─────────────
        carrier_id, carrier_team, carrier_pos = self._find_ball_carrier(
            team_a, team_b, ball_position
        )
        if carrier_id is None:
            return result

        result.ball_carrier = carrier_id
        result.ball_carrier_team = carrier_team

        # Coéquipiers et adversaires
        if carrier_team == 0:
            teammates = [(tid, x, y) for tid, x, y in team_a if tid != carrier_id]
            opponents = team_b
        else:
            teammates = [(tid, x, y) for tid, x, y in team_b if tid != carrier_id]
            opponents = team_a

        opp_positions = np.array([(x, y) for _, x, y in opponents]) if opponents else np.array([]).reshape(0, 2)

        # ── Analyser chaque ligne de passe ──────────────
        for tid, tx, ty in teammates:
            dist = np.sqrt((tx - carrier_pos[0])**2 + (ty - carrier_pos[1])**2)

            # Ignorer les passes trop longues
            if dist > self.max_pass_distance:
                continue

            lane = PassLane(
                from_id=carrier_id,
                to_id=tid,
                from_pos=carrier_pos,
                to_pos=(tx, ty),
                distance=dist,
            )

            # Vérifier si un adversaire bloque cette ligne
            if len(opp_positions) > 0:
                is_open, blocker_id, min_clearance = self._check_lane(
                    carrier_pos, (tx, ty), opponents, opp_positions
                )
                lane.is_open = is_open
                lane.blocking_player = blocker_id
                lane.min_clearance = min_clearance
            else:
                lane.is_open = True
                lane.min_clearance = float("inf")

            result.lanes.append(lane)

        # ── Métriques agrégées ──────────────────────────
        result.total_lanes = len(result.lanes)
        result.open_lanes = sum(1 for l in result.lanes if l.is_open)
        result.blocked_lanes = result.total_lanes - result.open_lanes
        result.pass_availability_pct = (
            result.open_lanes / result.total_lanes * 100
            if result.total_lanes > 0 else 0.0
        )

        # Meilleur receveur = ligne ouverte la + courte
        open_lanes = [l for l in result.lanes if l.is_open]
        if open_lanes:
            best = min(open_lanes, key=lambda l: l.distance)
            result.best_receiver = best.to_id

        # Joueur le plus isolé = toutes ses lignes sont bloquées ou il est le plus loin
        blocked_players = [l.to_id for l in result.lanes if not l.is_open]
        if blocked_players:
            from collections import Counter
            counts = Counter(blocked_players)
            result.most_isolated = counts.most_common(1)[0][0]

        return result

    def _find_ball_carrier(
        self,
        team_a: List[Tuple[int, float, float]],
        team_b: List[Tuple[int, float, float]],
        ball_pos: Tuple[float, float],
    ) -> Tuple[Optional[int], Optional[int], Tuple[float, float]]:
        """Identifie le joueur le plus proche du ballon."""
        best_id = None
        best_team = None
        best_pos = (0.0, 0.0)
        best_dist = float("inf")

        for tid, x, y in team_a:
            d = np.sqrt((x - ball_pos[0])**2 + (y - ball_pos[1])**2)
            if d < best_dist:
                best_dist = d
                best_id = tid
                best_team = 0
                best_pos = (x, y)

        for tid, x, y in team_b:
            d = np.sqrt((x - ball_pos[0])**2 + (y - ball_pos[1])**2)
            if d < best_dist:
                best_dist = d
                best_id = tid
                best_team = 1
                best_pos = (x, y)

        # Si personne n'est à < 5m du ballon, pas de porteur clair
        if best_dist > 5.0:
            return None, None, (0.0, 0.0)

        return best_id, best_team, best_pos

    def _check_lane(
        self,
        from_pos: Tuple[float, float],
        to_pos: Tuple[float, float],
        opponents: List[Tuple[int, float, float]],
        opp_positions: np.ndarray,
    ) -> Tuple[bool, Optional[int], float]:
        """
        Vérifie si une ligne de passe est bloquée par un adversaire.

        Calcule la distance perpendiculaire de chaque adversaire au segment.
        Si cette distance < blocking_radius, la ligne est bloquée.

        Returns:
            (is_open, blocking_player_id, min_clearance)
        """
        A = np.array(from_pos)
        B = np.array(to_pos)
        AB = B - A
        ab_len_sq = np.dot(AB, AB)

        if ab_len_sq < 1e-6:
            return True, None, float("inf")

        min_dist = float("inf")
        blocker_id = None

        for i, (tid, ox, oy) in enumerate(opponents):
            P = np.array([ox, oy])
            AP = P - A

            # Paramètre t de la projection sur le segment [A, B]
            t = np.dot(AP, AB) / ab_len_sq

            # Clamp t dans [0, 1] → point le + proche sur le segment
            t = max(0.0, min(1.0, t))

            # Point le plus proche sur le segment
            closest = A + t * AB
            dist = float(np.linalg.norm(P - closest))

            if dist < min_dist:
                min_dist = dist
                if dist < self.blocking_radius:
                    blocker_id = tid

        is_open = min_dist >= self.blocking_radius
        return is_open, blocker_id, min_dist


# ═══════════════════════════════════════════════════════════════
#  CONTAINER — Regroupe les 3 analyseurs
# ═══════════════════════════════════════════════════════════════

@dataclass
class AdvancedTacticalResult:
    """Résultat combiné des 3 analyses avancées."""
    space_control: SpaceControlResult = field(default_factory=SpaceControlResult)
    offside_line: OffsideLineResult = field(default_factory=OffsideLineResult)
    pass_lanes: PassLaneResult = field(default_factory=PassLaneResult)


class AdvancedTacticalAnalyzer:
    """
    Orchestrateur des 3 modules d'analyse avancée.

    Appelé une fois par frame après l'analyse tactique de base.
    """

    def __init__(self, fps: float = 25.0):
        self.space_control = SpaceControlAnalyzer()
        self.offside_tracker = OffsideLineTracker(fps=fps)
        self.pass_lanes = PassLaneAnalyzer()

        # Historique pour le rapport
        self.history: List[AdvancedTacticalResult] = []

    def analyze_frame(
        self,
        team_a: List[Tuple[int, float, float]],
        team_b: List[Tuple[int, float, float]],
        ball_position: Optional[Tuple[float, float]] = None,
    ) -> AdvancedTacticalResult:
        """Exécute les 3 analyses sur une frame."""
        result = AdvancedTacticalResult()

        if len(team_a) >= 2 or len(team_b) >= 2:
            result.space_control = self.space_control.analyze(team_a, team_b)
            result.offside_line = self.offside_tracker.analyze(
                team_a, team_b, ball_position
            )
            result.pass_lanes = self.pass_lanes.analyze(
                team_a, team_b, ball_position
            )

        self.history.append(result)
        return result

    def get_summary(self) -> Dict:
        """Génère un résumé des métriques avancées sur toute la vidéo."""
        if not self.history:
            return {}

        valid = [h for h in self.history
                 if h.space_control.team_a_pct > 0]

        if not valid:
            return {}

        # Space Control moyens
        avg_sc_a = np.mean([h.space_control.team_a_pct for h in valid])
        avg_sc_b = np.mean([h.space_control.team_b_pct for h in valid])

        # Stabilité de la ligne moyenne
        line_stab_a = [h.offside_line.line_stability_a for h in valid
                       if h.offside_line.line_stability_a > 0]
        line_stab_b = [h.offside_line.line_stability_b for h in valid
                       if h.offside_line.line_stability_b > 0]

        avg_stab_a = float(np.mean(line_stab_a)) if line_stab_a else 0.0
        avg_stab_b = float(np.mean(line_stab_b)) if line_stab_b else 0.0

        # Pass availability — globale et par équipe
        pass_avail = [h.pass_lanes.pass_availability_pct for h in valid
                      if h.pass_lanes.total_lanes > 0]
        avg_pass_avail = float(np.mean(pass_avail)) if pass_avail else 0.0

        pass_avail_a = [h.pass_lanes.pass_availability_pct for h in valid
                        if h.pass_lanes.total_lanes > 0
                        and h.pass_lanes.ball_carrier_team == 0]
        pass_avail_b = [h.pass_lanes.pass_availability_pct for h in valid
                        if h.pass_lanes.total_lanes > 0
                        and h.pass_lanes.ball_carrier_team == 1]
        avg_pass_avail_a = float(np.mean(pass_avail_a)) if pass_avail_a else 0.0
        avg_pass_avail_b = float(np.mean(pass_avail_b)) if pass_avail_b else 0.0

        # Offside positions count
        offside_a = sum(1 for h in valid if h.offside_line.is_offside_position_a)
        offside_b = sum(1 for h in valid if h.offside_line.is_offside_position_b)

        # Penalty area control
        pen_ctrl_a = np.mean([h.space_control.penalty_area_control_a for h in valid])
        pen_ctrl_b = np.mean([h.space_control.penalty_area_control_b for h in valid])

        return {
            "space_control": {
                "team_a_avg_pct": round(float(avg_sc_a), 1),
                "team_b_avg_pct": round(float(avg_sc_b), 1),
                "team_a_penalty_area_control_pct": round(float(pen_ctrl_a), 1),
                "team_b_penalty_area_control_pct": round(float(pen_ctrl_b), 1),
            },
            "defensive_line": {
                "team_a_stability_m": round(avg_stab_a, 2),
                "team_b_stability_m": round(avg_stab_b, 2),
                "team_a_offside_situations": offside_a,
                "team_b_offside_situations": offside_b,
            },
            "pass_lanes": {
                "avg_pass_availability_pct": round(avg_pass_avail, 1),
                "team_a_avg_pass_availability_pct": round(avg_pass_avail_a, 1),
                "team_b_avg_pass_availability_pct": round(avg_pass_avail_b, 1),
            },
        }
