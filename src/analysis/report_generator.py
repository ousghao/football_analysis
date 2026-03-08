"""
Générateur de rapport automatique.
Produit un rapport HTML/texte complet à partir de l'analyse tactique.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .tactical import TacticalAnalyzer, TacticalSnapshot
from .phase_detector import PhaseDetector


class ReportGenerator:
    """
    Génère des rapports de match automatiques.
    
    Élimine le besoin de noter manuellement :
    "Le bloc était trop bas à la 20ème minute."
    
    Le rapport dit : "L'équipe a passé 65% du temps en bloc médian
    et a subi 12 pressions intenses dans sa propre surface."
    """

    def __init__(
        self,
        tactical_analyzer: TacticalAnalyzer,
        phase_detector: PhaseDetector,
        team_a_name: str = "Équipe A",
        team_b_name: str = "Équipe B",
    ):
        self.tactical = tactical_analyzer
        self.phase = phase_detector
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name

    def generate_text_report(self) -> str:
        """Génère un rapport texte complet."""
        summary = self.tactical.get_period_summary()
        timeline = self.phase.get_timeline()
        phase_summary = self.phase.get_phase_summary()

        if not summary:
            return "Pas assez de données pour générer un rapport."

        lines = []
        lines.append("=" * 70)
        lines.append("   RAPPORT D'ANALYSE TACTIQUE - FOOTBALL INTELLIGENCE")
        lines.append("=" * 70)
        lines.append(f"   Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        lines.append(f"   Durée analysée: {summary.get('duration_minutes', 0)} minutes")
        lines.append(f"   Frames analysées: {summary.get('frames_analyzed', 0)}")
        lines.append("=" * 70)
        lines.append("")

        # ─── Résumé par équipe ──────────────────────────
        for team_key, team_name in [("team_a", self.team_a_name),
                                     ("team_b", self.team_b_name)]:
            data = summary.get(team_key, {})
            lines.append(f"┌─── {team_name} {'─' * (50 - len(team_name))}")
            lines.append(f"│")
            lines.append(f"│  Hauteur de bloc moyenne: {data.get('avg_block_height_pct', 0)}%")
            lines.append(f"│  Surface couverte (Convex Hull): {data.get('avg_hull_area_m2', 0)} m²")
            lines.append(f"│  Contrôle territorial (Voronoi): {data.get('avg_territory_pct', 0)}%")
            lines.append(f"│  Compacité (spread): {data.get('avg_spread_m', 0)} m")
            lines.append(f"│")
            lines.append(f"│  Distribution des phases:")

            phases = data.get("phase_distribution", {})
            for phase, pct in sorted(phases.items(), key=lambda x: -x[1]):
                bar = "█" * int(pct / 2)
                lines.append(f"│    {phase:<20s} {pct:5.1f}% {bar}")

            lines.append(f"│")
            lines.append(f"│  Pressing intense: {data.get('intense_pressing_pct', 0)}% du temps")
            lines.append(f"│    ({data.get('intense_pressing_frames', 0)} situations)")
            lines.append(f"└{'─' * 55}")
            lines.append("")

        # ─── Timeline ───────────────────────────────────
        if timeline:
            lines.append("┌─── Timeline des transitions tactiques ───────────────┐")
            for event in timeline[:30]:  # Max 30 événements
                lines.append(
                    f"│  [{event['time']}] {event['team']}: "
                    f"{event['transition']} (durée: {event['phase_duration']})"
                )
            if len(timeline) > 30:
                lines.append(f"│  ... et {len(timeline) - 30} autres transitions")
            lines.append(f"└{'─' * 55}┘")
            lines.append("")

        # ─── Insights automatiques ──────────────────────
        lines.append("┌─── Insights Automatiques ─────────────────────────────┐")
        insights = self._generate_insights(summary)
        for insight in insights:
            lines.append(f"│  → {insight}")
        lines.append(f"└{'─' * 55}┘")

        return "\n".join(lines)

    def _generate_insights(self, summary: Dict) -> List[str]:
        """Génère des insights automatiques basés sur les données."""
        insights = []
        ta = summary.get("team_a", {})
        tb = summary.get("team_b", {})

        # Domination territoriale
        terr_a = ta.get("avg_territory_pct", 50)
        terr_b = tb.get("avg_territory_pct", 50)
        if terr_a > terr_b + 5:
            insights.append(
                f"{self.team_a_name} domine territorialement "
                f"({terr_a:.0f}% vs {terr_b:.0f}%)"
            )
        elif terr_b > terr_a + 5:
            insights.append(
                f"{self.team_b_name} domine territorialement "
                f"({terr_b:.0f}% vs {terr_a:.0f}%)"
            )
        else:
            insights.append("Domination territoriale équilibrée")

        # Hauteur de bloc
        block_a = ta.get("avg_block_height_pct", 50)
        block_b = tb.get("avg_block_height_pct", 50)
        if block_a > 60:
            insights.append(f"{self.team_a_name} joue haut (bloc à {block_a:.0f}%)")
        elif block_a < 40:
            insights.append(f"{self.team_a_name} est très reculée (bloc à {block_a:.0f}%)")

        if block_b > 60:
            insights.append(f"{self.team_b_name} joue haut (bloc à {block_b:.0f}%)")
        elif block_b < 40:
            insights.append(f"{self.team_b_name} est très reculée (bloc à {block_b:.0f}%)")

        # Compacité
        spread_a = ta.get("avg_spread_m", 0)
        spread_b = tb.get("avg_spread_m", 0)
        if spread_a > 0:
            if spread_a < 12:
                insights.append(f"{self.team_a_name} est très compacte ({spread_a:.1f}m)")
            elif spread_a > 18:
                insights.append(f"{self.team_a_name} est étirée ({spread_a:.1f}m)")

        # Pressing
        press_a = ta.get("intense_pressing_pct", 0)
        press_b = tb.get("intense_pressing_pct", 0)
        if press_a > 20:
            insights.append(
                f"{self.team_a_name} pressing intense ({press_a:.0f}% du temps)"
            )
        if press_b > 20:
            insights.append(
                f"{self.team_b_name} pressing intense ({press_b:.0f}% du temps)"
            )

        return insights

    def generate_html_report(self, output_path: str) -> str:
        """Génère un rapport HTML professionnel."""
        summary = self.tactical.get_period_summary()
        timeline = self.phase.get_timeline()

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Tactique - Football Intelligence</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
               background: #0a0a1a; color: #e0e0e0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #1a1a3e, #2d2d6b);
                   padding: 30px; border-radius: 12px; margin-bottom: 20px;
                   text-align: center; }}
        .header h1 {{ color: #00ff88; font-size: 2em; }}
        .header p {{ color: #888; margin-top: 5px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ background: #1a1a2e; border-radius: 12px; padding: 20px;
                 border: 1px solid #333; }}
        .card h2 {{ color: #00ff88; margin-bottom: 15px; font-size: 1.3em;
                    border-bottom: 1px solid #333; padding-bottom: 8px; }}
        .stat {{ display: flex; justify-content: space-between; padding: 8px 0;
                 border-bottom: 1px solid #1a1a3e; }}
        .stat-label {{ color: #888; }}
        .stat-value {{ font-weight: bold; color: #fff; }}
        .phase-bar {{ display: flex; align-items: center; margin: 5px 0; }}
        .phase-bar .label {{ width: 160px; font-size: 0.9em; }}
        .phase-bar .bar {{ flex: 1; height: 20px; background: #333;
                          border-radius: 4px; overflow: hidden; }}
        .phase-bar .fill {{ height: 100%; border-radius: 4px;
                           transition: width 0.5s; }}
        .fill-blue {{ background: linear-gradient(90deg, #4444ff, #6666ff); }}
        .fill-red {{ background: linear-gradient(90deg, #ff4444, #ff6666); }}
        .timeline {{ margin-top: 20px; }}
        .timeline-event {{ padding: 10px 15px; border-left: 3px solid #00ff88;
                          margin: 5px 0; background: #1a1a2e; border-radius: 0 8px 8px 0; }}
        .timeline-event .time {{ color: #00ff88; font-weight: bold; }}
        .insight {{ background: #1a2e1a; border-left: 3px solid #00ff88;
                   padding: 10px 15px; margin: 5px 0; border-radius: 0 8px 8px 0; }}
        .full-width {{ grid-column: 1 / -1; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>⚽ Rapport d'Analyse Tactique</h1>
        <p>{self.team_a_name} vs {self.team_b_name}</p>
        <p>{datetime.now().strftime('%d/%m/%Y %H:%M')} | Durée: {summary.get('duration_minutes', 0)} min</p>
    </div>

    <div class="grid">
"""

        # Cards pour chaque équipe
        for team_key, team_name, color_class in [
            ("team_a", self.team_a_name, "fill-blue"),
            ("team_b", self.team_b_name, "fill-red"),
        ]:
            data = summary.get(team_key, {})
            html += f"""
        <div class="card">
            <h2>{team_name}</h2>
            <div class="stat">
                <span class="stat-label">Hauteur de bloc</span>
                <span class="stat-value">{data.get('avg_block_height_pct', 0)}%</span>
            </div>
            <div class="stat">
                <span class="stat-label">Surface couverte</span>
                <span class="stat-value">{data.get('avg_hull_area_m2', 0)} m²</span>
            </div>
            <div class="stat">
                <span class="stat-label">Contrôle territorial</span>
                <span class="stat-value">{data.get('avg_territory_pct', 0)}%</span>
            </div>
            <div class="stat">
                <span class="stat-label">Compacité</span>
                <span class="stat-value">{data.get('avg_spread_m', 0)} m</span>
            </div>
            <div class="stat">
                <span class="stat-label">Pressing intense</span>
                <span class="stat-value">{data.get('intense_pressing_pct', 0)}%</span>
            </div>
            <h3 style="margin-top:15px; color:#888;">Distribution des phases</h3>
"""
            for phase, pct in sorted(
                data.get("phase_distribution", {}).items(),
                key=lambda x: -x[1]
            ):
                html += f"""
            <div class="phase-bar">
                <span class="label">{phase}</span>
                <div class="bar"><div class="fill {color_class}" style="width:{pct}%"></div></div>
                <span style="width:50px; text-align:right;">{pct}%</span>
            </div>
"""
            html += "        </div>\n"

        # Insights
        insights = self._generate_insights(summary)
        html += """
        <div class="card full-width">
            <h2>🎯 Insights Automatiques</h2>
"""
        for insight in insights:
            html += f'            <div class="insight">→ {insight}</div>\n'

        html += "        </div>\n"

        # Timeline
        if timeline:
            html += """
        <div class="card full-width">
            <h2>📋 Timeline des Transitions</h2>
            <div class="timeline">
"""
            for event in timeline[:20]:
                html += f"""
                <div class="timeline-event">
                    <span class="time">[{event['time']}]</span>
                    {event['team']}: {event['transition']}
                    <span style="color:#666;"> (durée: {event['phase_duration']})</span>
                </div>
"""
            html += "            </div>\n        </div>\n"

        html += """
    </div>
</div>
</body>
</html>"""

        # Sauvegarder
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"[Rapport] HTML généré: {output_path}")
        return output_path

    def generate_json_report(self, output_path: str) -> str:
        """Export JSON pour intégration avec d'autres outils."""
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "team_a": self.team_a_name,
                "team_b": self.team_b_name,
            },
            "summary": self.tactical.get_period_summary(),
            "timeline": self.phase.get_timeline(),
            "phase_summary": self.phase.get_phase_summary(),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[Rapport] JSON généré: {output_path}")
        return output_path
