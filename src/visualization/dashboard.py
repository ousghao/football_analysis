"""
Dashboard de résumé : génère des graphiques mplsoccer à la fin de l'analyse.
Cartes de chaleur, positions moyennes, Voronoi statique, etc.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional

try:
    from mplsoccer import Pitch, VerticalPitch
    HAS_MPLSOCCER = True
except ImportError:
    HAS_MPLSOCCER = False
    print("[Dashboard] mplsoccer non installé. pip install mplsoccer")

from ..analysis.tactical import TacticalAnalyzer, TacticalSnapshot
from ..analysis.advanced_tactical import AdvancedTacticalAnalyzer
from ..detection.tracker import MultiObjectTracker


class TacticalDashboard:
    """
    Génère des visualisations statiques de synthèse avec mplsoccer.
    
    Graphiques produits :
    - Positions moyennes des joueurs
    - Carte de chaleur par joueur/équipe
    - Diagramme de Voronoi moyen
    - Évolution de la hauteur de bloc dans le temps
    - Distribution des phases de jeu
    """

    def __init__(self, output_dir: str = "output/dashboard"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        tactical_analyzer: TacticalAnalyzer,
        tracker: MultiObjectTracker,
        team_a_name: str = "Équipe A",
        team_b_name: str = "Équipe B",
        advanced_tactical: Optional[AdvancedTacticalAnalyzer] = None,
    ) -> List[str]:
        """
        Génère tous les graphiques du dashboard.
        
        Returns:
            Liste des chemins des images générées
        """
        outputs = []

        outputs.append(self.plot_average_positions(
            tactical_analyzer, team_a_name, team_b_name
        ))
        outputs.append(self.plot_block_height_timeline(
            tactical_analyzer, team_a_name, team_b_name
        ))
        outputs.append(self.plot_territory_timeline(
            tactical_analyzer, team_a_name, team_b_name
        ))
        outputs.append(self.plot_phase_distribution(
            tactical_analyzer, team_a_name, team_b_name
        ))
        outputs.append(self.plot_distance_stats(tracker))

        if advanced_tactical is not None:
            outputs.append(self.plot_space_control_timeline(
                advanced_tactical, team_a_name, team_b_name
            ))
            outputs.append(self.plot_pass_availability_timeline(
                advanced_tactical, team_a_name, team_b_name
            ))

        return [o for o in outputs if o is not None]

    def plot_average_positions(
        self,
        analyzer: TacticalAnalyzer,
        team_a_name: str = "Équipe A",
        team_b_name: str = "Équipe B",
    ) -> Optional[str]:
        """Trace les positions moyennes des joueurs sur le terrain."""
        if not HAS_MPLSOCCER or not analyzer.history:
            return None

        # Calculer les positions moyennes par joueur
        positions_a: Dict[int, List[Tuple[float, float]]] = {}
        positions_b: Dict[int, List[Tuple[float, float]]] = {}

        for snap in analyzer.history:
            for tid, x, y in snap.team_a_positions:
                if tid not in positions_a:
                    positions_a[tid] = []
                positions_a[tid].append((x, y))
            for tid, x, y in snap.team_b_positions:
                if tid not in positions_b:
                    positions_b[tid] = []
                positions_b[tid].append((x, y))

        fig, ax = plt.subplots(figsize=(12, 8))
        pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68,
                       pitch_color="#1a472a", line_color="white")
        pitch.draw(ax=ax)

        # Positions moyennes Équipe A
        for tid, pts in positions_a.items():
            arr = np.array(pts)
            avg_x, avg_y = arr.mean(axis=0)
            ax.scatter(avg_x, avg_y, c="blue", s=200, edgecolors="white",
                       linewidths=1.5, zorder=5)
            ax.annotate(str(tid), (avg_x, avg_y), color="white",
                        fontsize=8, ha="center", va="center",
                        fontweight="bold", zorder=6)

        # Positions moyennes Équipe B
        for tid, pts in positions_b.items():
            arr = np.array(pts)
            avg_x, avg_y = arr.mean(axis=0)
            ax.scatter(avg_x, avg_y, c="red", s=200, edgecolors="white",
                       linewidths=1.5, zorder=5)
            ax.annotate(str(tid), (avg_x, avg_y), color="white",
                        fontsize=8, ha="center", va="center",
                        fontweight="bold", zorder=6)

        ax.set_title(f"Positions Moyennes — {team_a_name} (bleu) vs {team_b_name} (rouge)",
                      color="white", fontsize=14, pad=15)

        path = str(self.output_dir / "average_positions.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="#1a472a", edgecolor="none")
        plt.close(fig)

        print(f"[Dashboard] Positions moyennes: {path}")
        return path

    def plot_block_height_timeline(
        self,
        analyzer: TacticalAnalyzer,
        team_a_name: str = "Équipe A",
        team_b_name: str = "Équipe B",
    ) -> Optional[str]:
        """Graphique de l'évolution de la hauteur de bloc dans le temps."""
        if not analyzer.history:
            return None

        times = [s.timestamp_sec / 60 for s in analyzer.history]
        height_a = [s.team_a_block_height_pct for s in analyzer.history]
        height_b = [s.team_b_block_height_pct for s in analyzer.history]

        # Lissage (moyenne mobile)
        window = min(30, len(times) // 10 + 1)
        if window > 1:
            height_a = np.convolve(height_a, np.ones(window) / window, mode="same")
            height_b = np.convolve(height_b, np.ones(window) / window, mode="same")

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0a0a1a")
        ax.set_facecolor("#0a0a1a")

        ax.plot(times, height_a, color="#4444ff", linewidth=1.5,
                label=team_a_name, alpha=0.9)
        ax.plot(times, height_b, color="#ff4444", linewidth=1.5,
                label=team_b_name, alpha=0.9)

        # Zones
        ax.axhspan(60, 100, alpha=0.1, color="red", label="Zone Pressing")
        ax.axhspan(40, 60, alpha=0.1, color="yellow", label="Zone Médiane")
        ax.axhspan(0, 40, alpha=0.1, color="green", label="Zone Basse")

        ax.axhline(y=50, color="white", linestyle="--", alpha=0.3)

        ax.set_xlabel("Temps (minutes)", color="white", fontsize=11)
        ax.set_ylabel("Hauteur du bloc (%)", color="white", fontsize=11)
        ax.set_title("Évolution de la Hauteur de Bloc", color="white",
                      fontsize=14, pad=10)
        ax.legend(fontsize=9, loc="upper right", facecolor="#1a1a2e",
                  edgecolor="#333", labelcolor="white")
        ax.tick_params(colors="white")
        ax.set_ylim(0, 100)

        for spine in ax.spines.values():
            spine.set_color("#333")

        path = str(self.output_dir / "block_height_timeline.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="#0a0a1a", edgecolor="none")
        plt.close(fig)

        print(f"[Dashboard] Hauteur de bloc: {path}")
        return path

    def plot_territory_timeline(
        self,
        analyzer: TacticalAnalyzer,
        team_a_name: str = "Équipe A",
        team_b_name: str = "Équipe B",
    ) -> Optional[str]:
        """Évolution du contrôle territorial (Voronoi) dans le temps."""
        if not analyzer.history:
            return None

        times = [s.timestamp_sec / 60 for s in analyzer.history]
        terr_a = [s.team_a_territory_pct for s in analyzer.history]
        terr_b = [s.team_b_territory_pct for s in analyzer.history]

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0a0a1a")
        ax.set_facecolor("#0a0a1a")

        ax.fill_between(times, terr_a, 50, alpha=0.3, color="#4444ff",
                         where=[a > 50 for a in terr_a])
        ax.fill_between(times, terr_b, 50, alpha=0.3, color="#ff4444",
                         where=[b > 50 for b in terr_b])

        ax.plot(times, terr_a, color="#4444ff", linewidth=1, label=team_a_name)
        ax.plot(times, terr_b, color="#ff4444", linewidth=1, label=team_b_name)

        ax.axhline(y=50, color="white", linestyle="--", alpha=0.3)

        ax.set_xlabel("Temps (minutes)", color="white", fontsize=11)
        ax.set_ylabel("Contrôle territorial (%)", color="white", fontsize=11)
        ax.set_title("Domination Territoriale (Voronoi)", color="white",
                      fontsize=14, pad=10)
        ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333",
                  labelcolor="white")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_color("#333")

        path = str(self.output_dir / "territory_timeline.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="#0a0a1a", edgecolor="none")
        plt.close(fig)

        print(f"[Dashboard] Territoire: {path}")
        return path

    def plot_phase_distribution(
        self,
        analyzer: TacticalAnalyzer,
        team_a_name: str = "Équipe A",
        team_b_name: str = "Équipe B",
    ) -> Optional[str]:
        """Camembert de la distribution des phases de jeu."""
        summary = analyzer.get_period_summary()
        if not summary:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0a0a1a")

        for ax, team_key, team_name, colors in [
            (axes[0], "team_a", team_a_name, plt.cm.Blues),
            (axes[1], "team_b", team_b_name, plt.cm.Reds),
        ]:
            ax.set_facecolor("#0a0a1a")
            data = summary.get(team_key, {}).get("phase_distribution", {})

            if data:
                labels = list(data.keys())
                values = list(data.values())
                color_list = colors(np.linspace(0.3, 0.9, len(labels)))

                wedges, texts, autotexts = ax.pie(
                    values, labels=labels, autopct="%1.1f%%",
                    colors=color_list, textprops={"color": "white", "fontsize": 9}
                )
                for autotext in autotexts:
                    autotext.set_fontsize(8)
            ax.set_title(team_name, color="white", fontsize=13)

        fig.suptitle("Distribution des Phases de Jeu", color="white",
                      fontsize=15, y=1.02)

        path = str(self.output_dir / "phase_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="#0a0a1a", edgecolor="none")
        plt.close(fig)

        print(f"[Dashboard] Phases: {path}")
        return path

    def plot_distance_stats(
        self, tracker: MultiObjectTracker
    ) -> Optional[str]:
        """Graphique en barres des distances parcourues par joueur."""
        stats = tracker.get_statistics()
        player_stats = {
            k: v for k, v in stats.items()
            if isinstance(v, dict) and "distance_m" in v
        }

        if not player_stats:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#0a0a1a")
        ax.set_facecolor("#0a0a1a")

        names = []
        distances = []
        colors = []

        for player_key, pdata in sorted(
            player_stats.items(),
            key=lambda x: -x[1]["distance_m"]
        ):
            names.append(player_key.replace("player_", "#"))
            distances.append(pdata["distance_m"] / 1000)  # km
            team = pdata.get("team", -1)
            colors.append("#4444ff" if team == 0 else
                          "#ff4444" if team == 1 else "#888888")

        bars = ax.barh(names, distances, color=colors, edgecolor="#333")

        ax.set_xlabel("Distance parcourue (km)", color="white", fontsize=11)
        ax.set_title("Distance Parcourue par Joueur", color="white",
                      fontsize=14, pad=10)
        ax.tick_params(colors="white")
        ax.invert_yaxis()

        for spine in ax.spines.values():
            spine.set_color("#333")

        # Ajouter les valeurs sur les barres
        for bar, dist in zip(bars, distances):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{dist:.2f} km", va="center", color="white", fontsize=9)

        path = str(self.output_dir / "distance_stats.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="#0a0a1a", edgecolor="none")
        plt.close(fig)

        print(f"[Dashboard] Distances: {path}")
        return path

    def plot_space_control_timeline(
        self,
        advanced: AdvancedTacticalAnalyzer,
        team_a_name: str = "Équipe A",
        team_b_name: str = "Équipe B",
    ) -> Optional[str]:
        """Évolution du Space Control (Voronoi par joueur) dans le temps."""
        if not advanced.history:
            return None

        sc_a = [h.space_control.team_a_pct for h in advanced.history]
        sc_b = [h.space_control.team_b_pct for h in advanced.history]
        frames = list(range(len(advanced.history)))

        # Lissage
        window = min(30, len(frames) // 10 + 1)
        if window > 1:
            sc_a = np.convolve(sc_a, np.ones(window) / window, mode="same")
            sc_b = np.convolve(sc_b, np.ones(window) / window, mode="same")

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0a0a1a")
        ax.set_facecolor("#0a0a1a")

        ax.fill_between(frames, sc_a, 50, alpha=0.3, color="#4444ff",
                         where=[a > 50 for a in sc_a])
        ax.fill_between(frames, sc_b, 50, alpha=0.3, color="#ff4444",
                         where=[b > 50 for b in sc_b])
        ax.plot(frames, sc_a, color="#4444ff", linewidth=1.5, label=team_a_name)
        ax.plot(frames, sc_b, color="#ff4444", linewidth=1.5, label=team_b_name)
        ax.axhline(y=50, color="white", linestyle="--", alpha=0.3)

        ax.set_xlabel("Frame", color="white", fontsize=11)
        ax.set_ylabel("Space Control (%)", color="white", fontsize=11)
        ax.set_title("Space Control — Voronoi par joueur", color="white",
                      fontsize=14, pad=10)
        ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
        ax.tick_params(colors="white")
        ax.set_ylim(20, 80)
        for spine in ax.spines.values():
            spine.set_color("#333")

        path = str(self.output_dir / "space_control_timeline.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="#0a0a1a", edgecolor="none")
        plt.close(fig)
        print(f"[Dashboard] Space Control: {path}")
        return path

    def plot_pass_availability_timeline(
        self,
        advanced: AdvancedTacticalAnalyzer,
        team_a_name: str = "Équipe A",
        team_b_name: str = "Équipe B",
    ) -> Optional[str]:
        """Évolution du Pass Availability % par équipe dans le temps."""
        if not advanced.history:
            return None

        frames = list(range(len(advanced.history)))

        # Séries par équipe (None quand cette équipe n'a pas le ballon)
        raw_a = [
            h.pass_lanes.pass_availability_pct
            if h.pass_lanes.total_lanes > 0 and h.pass_lanes.ball_carrier_team == 0
            else None
            for h in advanced.history
        ]
        raw_b = [
            h.pass_lanes.pass_availability_pct
            if h.pass_lanes.total_lanes > 0 and h.pass_lanes.ball_carrier_team == 1
            else None
            for h in advanced.history
        ]

        def _smooth(values, window=30):
            arr = np.array([v if v is not None else np.nan for v in values],
                           dtype=float)
            nans = np.isnan(arr)
            if nans.all():
                return arr
            x = np.arange(len(arr))
            arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
            w = min(window, max(1, len(arr) // 10 + 1))
            if w > 1:
                arr = np.convolve(arr, np.ones(w) / w, mode="same")
            return arr

        pa_a = _smooth(raw_a)
        pa_b = _smooth(raw_b)

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0a0a1a")
        ax.set_facecolor("#0a0a1a")

        ax.fill_between(frames, pa_a, 0, alpha=0.15, color="#00ff88")
        ax.plot(frames, pa_a, color="#00ff88", linewidth=1.5, label=team_a_name)

        ax.fill_between(frames, pa_b, 0, alpha=0.15, color="#ff6633")
        ax.plot(frames, pa_b, color="#ff6633", linewidth=1.5, label=team_b_name)

        ax.axhline(y=50, color="white", linestyle="--", alpha=0.3)

        ax.set_xlabel("Frame", color="white", fontsize=11)
        ax.set_ylabel("Pass Availability (%)", color="white", fontsize=11)
        ax.set_title("Pass Availability — Lignes de passe ouvertes par équipe",
                     color="white", fontsize=14, pad=10)
        ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333",
                  labelcolor="white")
        ax.tick_params(colors="white")
        ax.set_ylim(0, 100)
        for spine in ax.spines.values():
            spine.set_color("#333")

        path = str(self.output_dir / "pass_availability_timeline.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="#0a0a1a", edgecolor="none")
        plt.close(fig)
        print(f"[Dashboard] Pass Availability: {path}")
        return path
