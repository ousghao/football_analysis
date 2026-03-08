"""
╔══════════════════════════════════════════════════════════════════════╗
║           FOOTBALL TACTICAL INTELLIGENCE - MAIN PIPELINE           ║
║                                                                    ║
║  Transforme un flux vidéo brut en Intelligence Tactique.           ║
║  Du "Video Analysis" → "Data Intelligence"                         ║
╚══════════════════════════════════════════════════════════════════════╝

Pipeline complet :
  1. Charger la vidéo
  2. Détection + Tracking (YOLO + ByteTrack)
  3. Classification des équipes (K-Means)
  4. Projection 2D (Homographie)
  5. Analyse Tactique (Voronoi, Convex Hull, Phases)
  6. Visualisation (Annotations + Minimap)
  7. Export (Vidéo annotée + Rapport HTML/JSON)

Usage:
    python main.py --video data/sample_videos/match.mp4
    python main.py --video match.mp4 --calibrate      # Calibration manuelle
    python main.py --video match.mp4 --no-display      # Sans affichage live
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ─── Imports du projet ──────────────────────────────────────
from config import (
    DETECTION, TRACKING, TEAM_CLASSIFICATION, HOMOGRAPHY,
    TACTICAL, VISUALIZATION, OUTPUT_DIR, DATA_DIR
)
from src.detection.detector import PlayerDetector, BallTracker
from src.detection.tracker import MultiObjectTracker
from src.detection.motion_compensation import GlobalMotionCompensator
from src.pitch.pitch_template import PitchTemplate
from src.pitch.segmentation import PitchSegmenter
from src.pitch.homography import HomographyEstimator, ManualCalibrator
from src.team.classifier import TeamClassifier
from src.analysis.spatial import SpatialAnalyzer
from src.analysis.tactical import TacticalAnalyzer
from src.analysis.phase_detector import PhaseDetector
from src.analysis.report_generator import ReportGenerator
from src.analysis.advanced_tactical import AdvancedTacticalAnalyzer
from src.visualization.annotator import FrameAnnotator
from src.visualization.minimap import Minimap
from src.visualization.dashboard import TacticalDashboard


class FootballAnalysisPipeline:
    """
    Pipeline principal d'analyse tactique de football.
    
    Orchestre tous les modules pour transformer une vidéo
    en rapport d'intelligence tactique complet.
    """

    def __init__(
        self,
        video_path: str,
        output_name: str = None,
        team_a_name: str = "Équipe A",
        team_b_name: str = "Équipe B",
        homography_path: str = None,
        max_frames: int = 0,            # 0 = toute la vidéo
        skip_frames: int = 0,           # Traiter 1 frame sur N
        display: bool = True,
        show_voronoi: bool = False,
    ):
        self.video_path = video_path
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
        self.display = display
        self.show_voronoi = show_voronoi
        self.max_frames = max_frames
        self.skip_frames = skip_frames

        # Nom de sortie
        if output_name is None:
            output_name = Path(video_path).stem + "_analyzed"
        self.output_name = output_name
        self.output_dir = OUTPUT_DIR / output_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ─── Ouvrir la vidéo ────────────────────────────
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir la vidéo: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n{'='*60}")
        print(f"  FOOTBALL TACTICAL INTELLIGENCE")
        print(f"{'='*60}")
        print(f"  Vidéo:       {video_path}")
        print(f"  Résolution:  {self.frame_width}x{self.frame_height}")
        print(f"  FPS:         {self.fps}")
        print(f"  Frames:      {self.total_frames}")
        print(f"  Durée:       {self.total_frames / self.fps:.1f}s")
        print(f"  Sortie:      {self.output_dir}")
        print(f"{'='*60}\n")

        # ─── Initialiser les modules ────────────────────
        self._init_modules(homography_path)

    def _init_modules(self, homography_path: str = None):
        """Initialise tous les modules du pipeline."""

        # 1. Détection
        print("[Init] Chargement du détecteur YOLO...")
        self.detector = PlayerDetector()

        # 2. Tracking
        self.tracker = MultiObjectTracker(fps=self.fps)
        self.ball_tracker = BallTracker(max_gap=15)

        # 2b. Compensation du mouvement de caméra (GMC)
        self.gmc = GlobalMotionCompensator()

        # 3. Terrain
        self.pitch = PitchTemplate()
        self.pitch_segmenter = PitchSegmenter()

        # 4. Homographie
        self.homography = HomographyEstimator(self.pitch)
        if homography_path and Path(homography_path).exists():
            self.homography.load(homography_path)
            print(f"[Init] Homographie chargée: {homography_path}")

        # 5. Classification des équipes
        self.team_classifier = TeamClassifier()

        # 6. Analyse
        self.spatial_analyzer = SpatialAnalyzer()
        self.tactical_analyzer = TacticalAnalyzer(fps=self.fps)
        self.phase_detector = PhaseDetector(fps=self.fps)

        # 6b. Analyse avancée (Space Control, Offside Line, Pass Lanes)
        self.advanced_tactical = AdvancedTacticalAnalyzer(fps=self.fps)

        # 7. Visualisation
        self.annotator = FrameAnnotator()
        self.minimap = Minimap(pitch=self.pitch)

        # 8. Rapport
        self.report_generator = ReportGenerator(
            self.tactical_analyzer, self.phase_detector,
            self.team_a_name, self.team_b_name
        )

        # 9. Writer vidéo
        output_video_path = str(self.output_dir / f"{self.output_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*VISUALIZATION.output_codec)
        self.writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        print(f"[Init] Vidéo de sortie: {output_video_path}")

    def calibrate(self, frame: np.ndarray = None):
        """
        Lance la calibration interactive de l'homographie.

        Mode vidéo (par défaut) : navigation complète dans la vidéo
        pour sélectionner les meilleures frames et marquer des points
        depuis plusieurs positions de caméra.
        
        Mode frame unique : si une frame est fournie, calibre dessus.
        """
        calibrator = ManualCalibrator(self.pitch)

        if frame is not None:
            # Mode frame unique (rétro-compatible)
            self.homography = calibrator.calibrate_interactive(frame)
        else:
            # Mode navigation vidéo complète
            self.homography = calibrator.calibrate_interactive(
                frame=None, video_path=self.video_path
            )
            # Remettre la vidéo au début pour le pipeline
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Sauvegarder
        h_path = str(self.output_dir / "homography.json")
        self.homography.save(h_path)

    def run(self, progress_callback=None):
        """
        Exécute le pipeline complet d'analyse.
        
        Args:
            progress_callback: Optional callable(frame_idx, total_frames)
                               called after each processed frame for progress tracking.
        """
        print("\n[Pipeline] Démarrage de l'analyse...\n")
        start_time = time.time()

        frame_idx = 0
        processed = 0
        calibration_done = False
        team_fitted = False

        effective_total = self.max_frames if self.max_frames > 0 else self.total_frames
        pbar = tqdm(total=effective_total, desc="Analyse tactique",
                    unit="frame", ncols=100)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.max_frames > 0 and frame_idx >= self.max_frames:
                break

            # Skip frames pour accélérer
            if self.skip_frames > 0 and frame_idx % (self.skip_frames + 1) != 0:
                frame_idx += 1
                continue

            # ═══════════════════════════════════════════════
            # ÉTAPE 1 : Détection + Tracking
            # ═══════════════════════════════════════════════
            detections = self.detector.detect_with_tracking(
                frame, frame_idx=frame_idx
            )

            # ═══════════════════════════════════════════════
            # ÉTAPE 2 : Classification des équipes
            # ═══════════════════════════════════════════════
            # Entraîner le classifieur sur la 30ème frame (quand on a assez de joueurs)
            if not team_fitted and frame_idx >= 30 and len(detections.players) >= 8:
                print(f"\n[Pipeline] Entraînement du classifieur d'équipes "
                      f"(frame {frame_idx}, {len(detections.players)} joueurs)")
                self.team_classifier.fit(frame, detections)
                team_fitted = self.team_classifier.is_fitted

            if team_fitted:
                detections = self.team_classifier.classify_frame(frame, detections)

            # ═══════════════════════════════════════════════
            # ÉTAPE 2b : Compensation du mouvement de caméra
            # ═══════════════════════════════════════════════
            # Estime le pan/tilt caméra via optical flow sur le décor
            # et compense les positions pixel des joueurs avant projection.
            self.gmc.update(frame)

            if self.gmc.is_camera_moving():
                for det in detections.all_persons:
                    cx, cy = det.bottom_center
                    cx_c, cy_c = self.gmc.compensate_point((cx, cy))
                    det.compensated_foot = (cx_c, cy_c)
                if detections.ball:
                    bx, by = detections.ball.center
                    detections.ball.compensated_center = self.gmc.compensate_point((bx, by))
            else:
                for det in detections.all_persons:
                    det.compensated_foot = det.bottom_center
                if detections.ball:
                    detections.ball.compensated_center = detections.ball.center

            # Sélection automatique de la scène d'homographie
            # Quand le cameraman zoome ou change d'angle, le système bascule
            # vers la matrice H qui projette le plus de joueurs dans le terrain.
            if (self.homography.H is not None
                    and len(self.homography.scene_homographies) > 1):
                all_feet = [
                    getattr(det, 'compensated_foot', det.bottom_center)
                    for det in detections.all_persons
                ]
                if len(all_feet) >= 3:
                    self.homography.select_best_scene(all_feet)

            # ═══════════════════════════════════════════════
            # ÉTAPE 3 : Projection 2D (Homographie)
            # ═══════════════════════════════════════════════
            if self.homography.H is not None:
                for det in detections.all_persons:
                    foot = getattr(det, 'compensated_foot', det.bottom_center)
                    world_pos = self.homography.project_point(foot)
                    if world_pos is not None:
                        det.world_pos = world_pos

                if detections.ball:
                    ball_pt = getattr(detections.ball, 'compensated_center', detections.ball.center)
                    ball_world = self.homography.project_point(ball_pt)
                    if ball_world:
                        detections.ball.world_pos = ball_world

            # Marquer comme « autre » (team_id=2) tout joueur dont les pieds
            # projettent hors du terrain : ramasseurs de balles, spectateurs
            # partiellement détectés, etc. Ils ne seront pas inclus dans
            # l'analyse tactique et seront annotés en couleur « arbitre ».
            for det in detections.players:
                if det.world_pos is None:
                    det.team_id = 2

            # ═══════════════════════════════════════════════
            # ÉTAPE 4 : Mise à jour du tracker
            # ═══════════════════════════════════════════════
            active_tracks = self.tracker.update(detections)
            self.ball_tracker.update(detections.ball, frame_idx)

            # ═══════════════════════════════════════════════
            # ÉTAPE 5 : Analyse Tactique
            # ═══════════════════════════════════════════════
            team_a_world = [
                (det.track_id, det.world_pos[0], det.world_pos[1])
                for det in detections.players
                if det.team_id == 0 and det.world_pos and det.track_id
            ]
            team_b_world = [
                (det.track_id, det.world_pos[0], det.world_pos[1])
                for det in detections.players
                if det.team_id == 1 and det.world_pos and det.track_id
            ]
            ball_world = (
                detections.ball.world_pos
                if detections.ball and detections.ball.world_pos
                else None
            )

            snapshot = self.tactical_analyzer.analyze_frame(
                frame_idx=frame_idx,
                team_a_positions=team_a_world,
                team_b_positions=team_b_world,
                ball_position=ball_world,
                spatial_analyzer=self.spatial_analyzer,
            )

            self.phase_detector.update(snapshot)

            # Analyse avancée (Space Control, Offside Line, Pass Lanes)
            advanced_result = self.advanced_tactical.analyze_frame(
                team_a_world, team_b_world, ball_world
            )

            # ═══════════════════════════════════════════════
            # ÉTAPE 6 : Visualisation
            # ═══════════════════════════════════════════════
            # Vitesses pour l'annotation
            speeds = {}
            for tid, track in active_tracks.items():
                if track.speeds:
                    speeds[tid] = track.avg_speed * 3.6  # m/s -> km/h

            # Annoter la frame
            annotated = self.annotator.annotate_frame(
                frame, detections, snapshot, speeds,
                advanced_result=advanced_result,
            )

            # Minimap 2D (Bird's Eye View)
            if self.homography.H is not None and (team_a_world or team_b_world):
                voronoi_result = None
                if self.show_voronoi and team_a_world and team_b_world:
                    voronoi_result = self.spatial_analyzer.compute_voronoi(
                        team_a_world, team_b_world
                    )

                minimap_img = self.minimap.render(
                    snapshot,
                    show_hull=True,
                    show_voronoi=self.show_voronoi,
                    show_defensive_lines=True,
                    voronoi_result=voronoi_result,
                    advanced_result=advanced_result,
                )
                annotated = self.minimap.overlay_on_frame(
                    annotated, minimap_img,
                    position=VISUALIZATION.minimap_position,
                )

            # ═══════════════════════════════════════════════
            # ÉTAPE 7 : Écriture
            # ═══════════════════════════════════════════════
            self.writer.write(annotated)

            if progress_callback is not None:
                progress_callback(frame_idx, effective_total)

            if self.display:
                display_frame = cv2.resize(annotated, (1280, 720))
                cv2.imshow("Football Tactical Intelligence", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[Pipeline] Arrêt demandé par l'utilisateur.")
                    break
                elif key == ord("v"):
                    self.show_voronoi = not self.show_voronoi
                    print(f"[Pipeline] Voronoi: {'ON' if self.show_voronoi else 'OFF'}")

            frame_idx += 1
            processed += 1
            pbar.update(1)

        pbar.close()

        # ═══════════════════════════════════════════════════
        # FINALISATION
        # ═══════════════════════════════════════════════════
        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"  ANALYSE TERMINÉE")
        print(f"{'='*60}")
        print(f"  Frames traitées:  {processed}")
        print(f"  Temps total:      {elapsed:.1f}s")
        print(f"  Vitesse:          {processed / elapsed:.1f} FPS")
        print(f"{'='*60}")

        # Libérer les ressources vidéo
        self.cap.release()
        self.writer.release()
        if self.display:
            cv2.destroyAllWindows()

        # Générer les rapports
        self._generate_reports()

        return self.output_dir

    def _generate_reports(self):
        """Génère tous les rapports de sortie."""
        print("\n[Rapports] Génération des rapports...")

        # Rapport texte
        text_report = self.report_generator.generate_text_report()
        report_path = self.output_dir / "rapport_tactique.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text_report)
        print(f"  → Rapport texte: {report_path}")
        print("\n" + text_report)

        # Rapport HTML
        html_path = str(self.output_dir / "rapport_tactique.html")
        self.report_generator.generate_html_report(html_path)

        # Rapport JSON
        json_path = str(self.output_dir / "rapport_tactique.json")
        self.report_generator.generate_json_report(json_path)

        # Dashboard mplsoccer
        print("\n[Dashboard] Génération des graphiques...")
        dashboard = TacticalDashboard(
            output_dir=str(self.output_dir / "dashboard")
        )
        dashboard.generate_all(
            self.tactical_analyzer, self.tracker,
            self.team_a_name, self.team_b_name,
            advanced_tactical=self.advanced_tactical,
        )

        # Rapport avancé (Space Control, Offside, Pass Lanes)
        advanced_summary = self.advanced_tactical.get_summary()
        if advanced_summary:
            import json as json_mod
            adv_path = self.output_dir / "advanced_tactical.json"
            with open(adv_path, "w", encoding="utf-8") as f:
                json_mod.dump(advanced_summary, f, indent=2, ensure_ascii=False)
            print(f"  → Analyse avancée: {adv_path}")

        # Statistiques de tracking
        stats = self.tracker.get_statistics()
        stats_path = self.output_dir / "tracking_stats.json"
        import json
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  → Stats tracking: {stats_path}")

        print(f"\n{'='*60}")
        print(f"  Tous les résultats sont dans: {self.output_dir}")
        print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Football Tactical Intelligence - Analyse vidéo automatisée",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py --video match.mp4
  python main.py --video match.mp4 --calibrate
  python main.py --video match.mp4 --team-a "MAS" --team-b "WAC"
  python main.py --video match.mp4 --max-frames 500 --no-display
  python main.py --video match.mp4 --voronoi --skip 2
        """
    )

    parser.add_argument("--video", "-v", required=True,
                        help="Chemin vers la vidéo du match")
    parser.add_argument("--team-a", default="Équipe A",
                        help="Nom de l'équipe A")
    parser.add_argument("--team-b", default="Équipe B",
                        help="Nom de l'équipe B")
    parser.add_argument("--calibrate", "-c", action="store_true",
                        help="Lancer la calibration manuelle de l'homographie")
    parser.add_argument("--homography", default=None,
                        help="Chemin vers un fichier homography.json existant")
    parser.add_argument("--output", "-o", default=None,
                        help="Nom du dossier de sortie")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Nombre max de frames (0 = toute la vidéo)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Traiter 1 frame sur N (0 = toutes)")
    parser.add_argument("--no-display", action="store_true",
                        help="Pas d'affichage en temps réel")
    parser.add_argument("--voronoi", action="store_true",
                        help="Activer les diagrammes de Voronoi sur la minimap")

    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = FootballAnalysisPipeline(
        video_path=args.video,
        output_name=args.output,
        team_a_name=args.team_a,
        team_b_name=args.team_b,
        homography_path=args.homography,
        max_frames=args.max_frames,
        skip_frames=args.skip,
        display=not args.no_display,
        show_voronoi=args.voronoi,
    )

    if args.calibrate:
        pipeline.calibrate()

    pipeline.run()


if __name__ == "__main__":
    main()
