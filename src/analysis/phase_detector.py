"""
Détecteur de phases de jeu avancé.
Analyse les transitions et les patterns tactiques complexes.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import deque, Counter
from dataclasses import dataclass
from enum import Enum

from .tactical import GamePhase, TacticalSnapshot


@dataclass
class PhaseEvent:
    """Événement de changement de phase."""
    frame_idx: int
    timestamp_sec: float
    team: str                    # "A" ou "B"
    from_phase: GamePhase
    to_phase: GamePhase
    duration_sec: float          # Durée de la phase précédente


class PhaseDetector:
    """
    Détecteur de phases de jeu avancé avec analyse des transitions.
    
    Fonctionnalités :
    - Détection de contre-attaque (changement rapide de hauteur de bloc)
    - Détection de transition (perte/gain de balle)
    - Timeline des phases (quand chaque phase commence/finit)
    """

    def __init__(self, fps: float = 30.0, smoothing_window: int = 30):
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.events: List[PhaseEvent] = []

        self._last_phase_a: Optional[GamePhase] = None
        self._last_phase_b: Optional[GamePhase] = None
        self._phase_start_a: int = 0
        self._phase_start_b: int = 0

        # Buffer pour détecter les contre-attaques
        self._centroid_history_a: deque = deque(maxlen=int(fps * 5))  # 5 sec
        self._centroid_history_b: deque = deque(maxlen=int(fps * 5))

        # Cooldown entre deux détections de contre-attaque (évite les faux positifs)
        self._last_ca_frame_a: int = -99999
        self._last_ca_frame_b: int = -99999
        self._ca_cooldown_frames: int = int(fps * 10)  # 10 secondes minimum

        # Fenêtre de vote pour stabiliser la phase (2 secondes)
        _vw = max(3, int(fps * 2))
        self._phase_vote_buffer_a: deque = deque(maxlen=_vw)
        self._phase_vote_buffer_b: deque = deque(maxlen=_vw)

    def update(self, snapshot: TacticalSnapshot):
        """
        Met à jour le détecteur avec un nouveau snapshot.
        Détecte les changements de phase et les événements.
        """
        frame_idx = snapshot.frame_idx
        ts = snapshot.timestamp_sec

        # ─── Tracking du centroïde — ignore les frames sans détection ──
        cx_a = snapshot.team_a_centroid[0]
        cx_b = snapshot.team_b_centroid[0]
        # N'ajouter que si des joueurs sont détectés (centroïde non nul)
        if cx_a != 0.0 or len(self._centroid_history_a) == 0:
            self._centroid_history_a.append(cx_a)
        if cx_b != 0.0 or len(self._centroid_history_b) == 0:
            self._centroid_history_b.append(cx_b)

        # ─── Détection de contre-attaque avec cooldown ───────────────
        ca_cooldown_ok_a = (frame_idx - self._last_ca_frame_a) > self._ca_cooldown_frames
        ca_cooldown_ok_b = (frame_idx - self._last_ca_frame_b) > self._ca_cooldown_frames

        ca_a = ca_cooldown_ok_a and self._detect_counter_attack(self._centroid_history_a)
        ca_b = ca_cooldown_ok_b and self._detect_counter_attack(self._centroid_history_b)

        phase_a = snapshot.team_a_phase
        phase_b = snapshot.team_b_phase

        if ca_a:
            phase_a = GamePhase.CONTRE_ATTAQUE
            self._last_ca_frame_a = frame_idx
        if ca_b:
            phase_b = GamePhase.CONTRE_ATTAQUE
            self._last_ca_frame_b = frame_idx

        # ─── Phase voting (fenêtre 2s pour éviter les oscillations) ──
        self._phase_vote_buffer_a.append(phase_a)
        self._phase_vote_buffer_b.append(phase_b)
        if len(self._phase_vote_buffer_a) >= int(self.fps):
            phase_a = Counter(self._phase_vote_buffer_a).most_common(1)[0][0]
        if len(self._phase_vote_buffer_b) >= int(self.fps):
            phase_b = Counter(self._phase_vote_buffer_b).most_common(1)[0][0]

        # ─── Détection des changements de phase ────────
        if self._last_phase_a is not None and phase_a != self._last_phase_a:
            duration = (frame_idx - self._phase_start_a) / self.fps
            if duration > 3.0:  # Minimum 3 secondes de phase stable
                self.events.append(PhaseEvent(
                    frame_idx=frame_idx,
                    timestamp_sec=ts,
                    team="A",
                    from_phase=self._last_phase_a,
                    to_phase=phase_a,
                    duration_sec=duration,
                ))
            self._phase_start_a = frame_idx

        if self._last_phase_b is not None and phase_b != self._last_phase_b:
            duration = (frame_idx - self._phase_start_b) / self.fps
            if duration > 3.0:
                self.events.append(PhaseEvent(
                    frame_idx=frame_idx,
                    timestamp_sec=ts,
                    team="B",
                    from_phase=self._last_phase_b,
                    to_phase=phase_b,
                    duration_sec=duration,
                ))
            self._phase_start_b = frame_idx

        self._last_phase_a = phase_a
        self._last_phase_b = phase_b

    def _detect_counter_attack(
        self, centroid_history: deque, speed_threshold: float = 15.0
    ) -> bool:
        """
        Détecte une contre-attaque basée sur la vitesse de progression du centroïde.

        Seuil relevé à 15 m/s (vs 8 m/s) pour éviter les faux positifs lors
        de pertes de détection ou de changements de formation naturels.
        """
        if len(centroid_history) < self.fps:
            return False

        recent = list(centroid_history)[-int(self.fps):]  # Dernière seconde
        progression = recent[-1] - recent[0]  # Mètres de déplacement

        return abs(progression) > speed_threshold

    def get_timeline(self) -> List[Dict]:
        """
        Génère la timeline des phases de jeu.
        Format adapté pour l'affichage ou l'export.
        """
        timeline = []
        for event in self.events:
            minutes = int(event.timestamp_sec // 60)
            seconds = int(event.timestamp_sec % 60)

            timeline.append({
                "time": f"{minutes:02d}:{seconds:02d}",
                "frame": event.frame_idx,
                "team": f"Équipe {event.team}",
                "transition": f"{event.from_phase.value} → {event.to_phase.value}",
                "phase_duration": f"{event.duration_sec:.1f}s",
            })

        return timeline

    def get_phase_summary(self) -> Dict:
        """Résumé des durées passées dans chaque phase."""
        summary = {"team_a": {}, "team_b": {}}

        for event in self.events:
            team_key = f"team_{event.team.lower()}"
            phase = event.from_phase.value
            if phase not in summary[team_key]:
                summary[team_key][phase] = 0.0
            summary[team_key][phase] += event.duration_sec

        return summary
