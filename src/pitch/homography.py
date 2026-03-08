"""
Module d'homographie : transformation perspective → vue de dessus.
Le cœur du "Digital Twin" : convertit les coordonnées pixels en mètres réels.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
from pathlib import Path

from .pitch_template import PitchTemplate, PITCH_LENGTH, PITCH_WIDTH


class HomographyEstimator:
    """
    Calcule et applique la matrice d'homographie pour passer
    de la vue caméra à la vue de dessus (Bird's Eye View).
    
    La matrice H vérifie : point_2D = H @ point_camera
    """

    def __init__(self, pitch: PitchTemplate = None):
        self.pitch = pitch or PitchTemplate()
        self.H: Optional[np.ndarray] = None          # Matrice 3x3
        self.H_inv: Optional[np.ndarray] = None      # Inverse pour projection retour
        self.src_points: Optional[np.ndarray] = None  # Points image
        self.dst_points: Optional[np.ndarray] = None  # Points terrain 2D
        self.reprojection_error: float = float("inf")
        # Multi-scène : une H par position/zoom caméra
        self.scene_homographies: List[Dict] = []
        self._active_scene: int = 0

    def compute_from_correspondences(
        self,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]],
        method: int = cv2.RANSAC,
        ransac_threshold: float = 3.0,
    ) -> np.ndarray:
        """
        Calcule la matrice d'homographie à partir de correspondances point-à-point.

        Pipeline:
          1. RANSAC pour identifier les inliers (rejette points d'autres caméras)
          2. Raffinement par moindres carrés sur les seuls inliers
          3. Erreur de reprojection calculée uniquement sur les inliers

        Args:
            image_points: Points dans l'image caméra [(x_px, y_px), ...]
            world_points: Points correspondants sur le terrain 2D [(X_m, Y_m), ...]
            method: Méthode de calcul (cv2.RANSAC recommandé)
            ransac_threshold: Seuil RANSAC en mètres terrain (défaut 3.0m)
            
        Returns:
            Matrice H (3x3)
        """
        assert len(image_points) >= 4, "Au moins 4 correspondances requises"
        assert len(image_points) == len(world_points), "Nombre de points incohérent"

        src = np.array(image_points, dtype=np.float64)
        dst = np.array(world_points, dtype=np.float64)

        # Étape 1 : RANSAC pour trouver les inliers
        H_ransac, mask = cv2.findHomography(src, dst, method, ransac_threshold)

        if H_ransac is None:
            print("[Homographie] ⚠ RANSAC a échoué, recalcul sans RANSAC...")
            H_ransac, mask = cv2.findHomography(src, dst, 0)

        inlier_mask = mask.flatten().astype(bool)
        n_inliers = int(inlier_mask.sum())
        n_outliers = len(inlier_mask) - n_inliers

        # Rapporter les outliers (points de caméras différentes)
        if n_outliers > 0:
            print(f"[Homographie] ⚠ {n_outliers} point(s) rejeté(s) comme outliers "
                  f"(probablement depuis une autre position caméra):")
            for i, (is_inlier, sp, dp) in enumerate(zip(inlier_mask, src, dst)):
                if not is_inlier:
                    print(f"   OUTLIER #{i+1}: pixel({sp[0]:.0f},{sp[1]:.0f}) "
                          f"→ terrain({dp[0]:.1f},{dp[1]:.1f}m)")

        # Étape 2 : Raffinement par moindres carrés sur les inliers seuls
        # Donne un H plus précis que le RANSAC initial
        self.H = H_ransac  # fallback
        if n_inliers >= 4:
            H_refined, _ = cv2.findHomography(
                src[inlier_mask], dst[inlier_mask], 0  # 0 = least-squares pur
            )
            if H_refined is not None:
                self.H = H_refined
                print(f"[Homographie] Raffinement par moindres carrés sur "
                      f"{n_inliers} inliers.")
        else:
            print(f"[Homographie] ⚠ Seulement {n_inliers} inliers — résultat peu fiable.")

        self.H_inv = np.linalg.inv(self.H)
        self.src_points = src
        self.dst_points = dst

        # Étape 3 : Erreur sur les inliers uniquement (mesure réaliste)
        self._compute_reprojection_error(src, dst, inlier_mask)

        # Construire les scènes secondaires depuis les points outliers RANSAC.
        # Chaque scène = un cluster de points cohérent (même position/zoom caméra).
        if n_outliers >= 4:
            secondary = self._build_secondary_scenes(
                src[~inlier_mask], dst[~inlier_mask]
            )
        else:
            secondary = []

        main_scene = {
            "H": self.H.tolist(),
            "H_inv": self.H_inv.tolist(),
            "n_points": n_inliers,
            "reprojection_error": self.reprojection_error,
            # Empreinte calibrée : bounding box des points de calibration monde.
            # Utilisée pour le per-player override dans project_point.
            # Note : select_best_scene utilise les limites COMPLÈTES du terrain
            # pour la scène 0 afin de compter tous les joueurs dans le score global.
            "world_bounds": self._make_world_bounds(dst[inlier_mask]),
        }
        self.scene_homographies = [main_scene] + secondary
        self._active_scene = 0

        print(f"[Homographie] Matrice calculée. Erreur de reprojection (inliers): "
              f"{self.reprojection_error:.2f} mètres. "
              f"Inliers: {n_inliers}/{len(inlier_mask)}")

        if secondary:
            print(f"[Homographie] 🔄 Multi-scène: "
                  f"{len(self.scene_homographies)} position(s) caméra détectées:")
            for i, sc in enumerate(self.scene_homographies):
                print(f"   Scène {i+1}: {sc['n_points']} pts, "
                      f"erreur {sc['reprojection_error']:.2f}m")
        elif self.reprojection_error > 5.0:
            print(f"[Homographie] ⚠ L'erreur est élevée ({self.reprojection_error:.1f}m). "
                  f"Conseil: calibrez depuis UNE SEULE position caméra où "
                  f"un maximum de repères sont visibles.")

        return self.H

    def _compute_reprojection_error(
        self, src: np.ndarray, dst: np.ndarray,
        mask: Optional[np.ndarray] = None
    ):
        """
        Calcule l'erreur moyenne de reprojection.

        Utilise uniquement les inliers RANSAC (mask=True) pour donner
        une mesure réaliste de la qualité de l'homographie.
        Les outliers (points d'une autre position caméra) sont ignorés.
        """
        if self.H is None:
            return

        # Appliquer le masque d'inliers si fourni
        if mask is not None:
            bool_mask = np.array(mask, dtype=bool).flatten()
            src_eval = src[bool_mask]
            dst_eval = dst[bool_mask]
        else:
            src_eval = src
            dst_eval = dst

        projected = self.project_points([(p[0], p[1]) for p in src_eval])
        errors = []
        for proj, gt in zip(projected, dst_eval):
            if proj is not None:
                err = np.sqrt((proj[0] - gt[0]) ** 2 + (proj[1] - gt[1]) ** 2)
                errors.append(err)

        self.reprojection_error = np.mean(errors) if errors else float("inf")

    # ─────────────────────────────────────────────────────────────────────
    # MULTI-SCÈNE — Gère le zoom/dézoom et les changements d'angle caméra
    # ─────────────────────────────────────────────────────────────────────

    def _build_secondary_scenes(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        min_pts: int = 4,
        max_scenes: int = 4,
    ) -> List[Dict]:
        """
        Construit des matrices H supplémentaires depuis les points outliers RANSAC.
        Chaque cluster = une position ou un zoom caméra distinct.
        Utilise RANSAC itératif : trouver le cluster → retirer ses points → répéter.
        """
        scenes = []
        rem_src = src.copy()
        rem_dst = dst.copy()

        while len(rem_src) >= min_pts and len(scenes) < max_scenes:
            H, mask = cv2.findHomography(rem_src, rem_dst, cv2.RANSAC, 3.0)
            if H is None:
                break
            inliers = mask.flatten().astype(bool)
            n_in = int(inliers.sum())
            if n_in < min_pts:
                break

            # Raffinement moindres carrés sur les inliers
            H_ref, _ = cv2.findHomography(rem_src[inliers], rem_dst[inliers], 0)
            H_sc = H_ref if H_ref is not None else H
            H_inv = np.linalg.inv(H_sc)

            # Erreur de reprojection pour cette scène
            src_h = np.hstack([rem_src[inliers], np.ones((n_in, 1))])
            proj = (H_sc @ src_h.T).T
            w = proj[:, 2:3]
            w[np.abs(w) < 1e-10] = 1e-10
            xy = proj[:, :2] / w
            err = float(np.mean(np.sqrt(np.sum((xy - rem_dst[inliers]) ** 2, axis=1))))

            scenes.append({
                "H": H_sc.tolist(),
                "H_inv": H_inv.tolist(),
                "n_points": n_in,
                "reprojection_error": err,
                "world_bounds": self._make_world_bounds(rem_dst[inliers]),
            })
            rem_src = rem_src[~inliers]
            rem_dst = rem_dst[~inliers]

        return scenes

    def _make_world_bounds(self, dst_pts: np.ndarray, padding: float = 5.0) -> dict:
        """Bounding-box des points monde d'une scène + marge padding."""
        return {
            "x_min": float(dst_pts[:, 0].min()) - padding,
            "x_max": float(dst_pts[:, 0].max()) + padding,
            "y_min": float(dst_pts[:, 1].min()) - padding,
            "y_max": float(dst_pts[:, 1].max()) + padding,
        }

    def _activate_scene(self, idx: int) -> None:
        """Bascule vers la scène d'homographie à l'index donné."""
        if 0 <= idx < len(self.scene_homographies):
            sc = self.scene_homographies[idx]
            self._active_scene = idx
            self.H = np.array(sc["H"], dtype=np.float64)
            self.H_inv = np.array(sc["H_inv"], dtype=np.float64)
            self.reprojection_error = sc["reprojection_error"]

    def select_best_scene(
        self, image_points: List[Tuple[float, float]]
    ) -> bool:
        """
        Sélectionne automatiquement la meilleure homographie pour cette frame.

        Algorithme en deux étapes :
        1. Pour chaque scène, projeter les points avec son H et compter combien
           tombent dans les limites monde calibrées de CETTE scène (world_bounds).
        2. Appliquer un facteur d'étalement : un mauvais H peut projeter tous les
           joueurs à l'intérieur de sa zone mais ils seront tous regroupés dans
           un coin, alors qu'un bon H les étale naturellement dans toute sa zone.
        3. Hystérèse relative : seule une amélioration significative déclenche
           un changement pour éviter les oscillations lors des transitions.

        Args:
            image_points: Positions pixel (pieds) des joueurs détectés.
        Returns:
            True si la scène active a changé.
        """
        if len(self.scene_homographies) <= 1 or not image_points:
            return False

        adj_scores: List[float] = []

        for sc_idx, sc in enumerate(self.scene_homographies):
            H = np.array(sc["H"], dtype=np.float64)

            # La scène 0 (vue large) est supposée couvrir tout le terrain :
            # utiliser les limites complètes du pitch pour le scoring global.
            # Ainsi les joueurs côté gauche comptent pour la scène principale
            # et n'inflatent pas artificiellement le score d'une scène secondaire.
            # Les scènes secondaires (zoom) gardent leurs world_bounds serrés
            # pour que le switch ne se déclenche que quand vraiment justifié.
            if sc_idx == 0:
                bx0, bx1 = -2.0, PITCH_LENGTH + 2.0
                by0, by1 = -2.0, PITCH_WIDTH + 2.0
            else:
                bounds = sc.get("world_bounds")
                if bounds:
                    bx0, bx1 = bounds["x_min"], bounds["x_max"]
                    by0, by1 = bounds["y_min"], bounds["y_max"]
                else:
                    bx0, bx1 = -2.0, PITCH_LENGTH + 2.0
                    by0, by1 = -2.0, PITCH_WIDTH + 2.0

            in_pts: List[Tuple[float, float]] = []
            for px, py in image_points:
                pt = np.array([px, py, 1.0])
                proj = H @ pt
                if abs(proj[2]) < 1e-10:
                    continue
                x, y = proj[0] / proj[2], proj[1] / proj[2]
                if bx0 <= x <= bx1 and by0 <= y <= by1:
                    in_pts.append((x, y))

            n_in = len(in_pts)
            if n_in == 0:
                adj_scores.append(0.0)
                continue

            # Facteur d'étalement : pénalise les projections trop regroupées.
            # Un H incorrect projette souvent tous les joueurs dans un coin de
            # sa zone de couverture — score réduit dans ce cas.
            x_spread = max(p[0] for p in in_pts) - min(p[0] for p in in_pts)
            scene_x_range = max(1.0, bx1 - bx0)
            spread_factor = min(1.0, x_spread / (scene_x_range * 0.3))

            adj_scores.append(float(n_in) * spread_factor)

        best_idx = int(np.argmax(adj_scores))
        best_score = adj_scores[best_idx]
        cur_score = adj_scores[self._active_scene]

        # Basculer seulement si gain significatif (> 1.0 point ajusté)
        if best_idx != self._active_scene and best_score > cur_score + 1.0:
            print(f"[Homographie] 🔄 Scène {best_idx + 1} activée "
                  f"(score ajusté: {best_score:.1f} vs {cur_score:.1f})")
            self._activate_scene(best_idx)
            return True
        return False

    def project_point(
        self, image_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Projette un point de l'image vers le terrain 2D (en mètres).

        En mode multi-scène, applique une sélection de scène PER-PLAYER :
        si la projection de la scène active tombe EN DEHORS de sa zone de
        calibration (world_bounds), le système essaie les autres scènes et
        utilise celle dont les world_bounds couvrent le résultat projeté.

        Exemple typique :
          Scène 1 active (vue large, droite) — joueur côté gauche :
          → Scène 1 projette à x=44m (hors bounds [47.5–110m])
          → Scène 2 projette à x=-3m (dans bounds [-5–21.5m])
          → Retourne le résultat de la Scène 2. ✓

        Args:
            image_point: (x_pixel, y_pixel) dans l'image caméra

        Returns:
            (X_mètres, Y_mètres) sur le terrain, ou None si hors terrain
        """
        if self.H is None:
            return None

        margin = 2.0

        def _apply(H: np.ndarray) -> Optional[Tuple[float, float]]:
            pt = np.array([image_point[0], image_point[1], 1.0], dtype=np.float64)
            proj = H @ pt
            if abs(proj[2]) < 1e-10:
                return None
            return float(proj[0] / proj[2]), float(proj[1] / proj[2])

        def _in_pitch(pos: Optional[Tuple[float, float]]) -> bool:
            if pos is None:
                return False
            x, y = pos
            return (-margin <= x <= self.pitch.length + margin and
                    -margin <= y <= self.pitch.width + margin)

        def _in_scene_bounds(pos: Tuple[float, float], scene: dict) -> bool:
            b = scene.get('world_bounds')
            if not b:
                return True
            return (b['x_min'] <= pos[0] <= b['x_max'] and
                    b['y_min'] <= pos[1] <= b['y_max'])

        # ── Scène unique : comportement inchangé (pas de surcharge) ────────
        if len(self.scene_homographies) <= 1:
            pos = _apply(self.H)
            return pos if _in_pitch(pos) else None

        # ── Multi-scène : sélection par player ─────────────────────────────
        # Règle principale : seule la scène primaire (index 0, vue large) peut
        # déclencher un override par player vers une scène secondaire.
        # Si une scène secondaire est globalement active (vue zoomée), on lui
        # fait entièrement confiance — appliquer le H de la vue large sur des
        # pixels d'une frame zoomée donnerait des positions complètement fausses.
        active_sc = self.scene_homographies[self._active_scene]
        pos = _apply(self.H)

        if pos is not None:
            if _in_scene_bounds(pos, active_sc):
                # Dans la zone calibrée de la scène active → résultat fiable
                return pos if _in_pitch(pos) else None

            # Hors des world_bounds de la scène active.
            # Chercher une scène alternative seulement si on est sur la scène 0
            # (vue large). Les scènes zoomed (idx > 0) ne font pas d'override
            # croisé : hors-limites dans la vue zoomée = hors terrain ou bord de
            # la zone visible, on retombe sur le repli.
            if self._active_scene == 0:
                for idx, sc in enumerate(self.scene_homographies):
                    if idx == 0:
                        continue
                    H_sc = np.array(sc['H'], dtype=np.float64)
                    alt = _apply(H_sc)
                    # ⚠ Vérification double : dans les limites de la scène ET
                    # dans les limites du terrain — évite que les ramasseurs
                    # à x=-5m soient retournés comme joueurs valides.
                    if alt is not None and _in_scene_bounds(alt, sc) and _in_pitch(alt):
                        return alt

            # Repli : aucune scène candidate ne couvre ce pixel.
            # Vérification de fiabilité : si TOUTES les scènes secondaires projettent
            # ce pixel en dehors du terrain, la projection de la scène principale est
            # probablement une extrapolation aberrante (ramasseur de balles dans une
            # zone mal couverte). On retourne None pour exclure ce joueur.
            # Si au moins une scène secondaire projette dans le terrain, les deux
            # scènes "s'accordent" sur la présence d'un joueur → résultat fiable.
            secondary_agrees = False
            for idx, sc in enumerate(self.scene_homographies):
                if idx == self._active_scene:
                    continue
                H_sc = np.array(sc['H'], dtype=np.float64)
                alt = _apply(H_sc)
                if alt is not None and _in_pitch(alt):
                    secondary_agrees = True
                    break

            if not secondary_agrees:
                return None  # Extrapolation non validée → hors terrain probable

            return pos if _in_pitch(pos) else None

        return None

    def project_points(
        self, image_points: List[Tuple[float, float]]
    ) -> List[Optional[Tuple[float, float]]]:
        """Projette une liste de points."""
        return [self.project_point(p) for p in image_points]

    def inverse_project_point(
        self, world_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Projette un point du terrain 2D vers l'image (inverse).
        Utile pour dessiner des éléments tactiques sur la vidéo.
        """
        if self.H_inv is None:
            return None

        pt = np.array([world_point[0], world_point[1], 1.0], dtype=np.float64)
        projected = self.H_inv @ pt

        if abs(projected[2]) < 1e-10:
            return None

        x = projected[0] / projected[2]
        y = projected[1] / projected[2]

        return (float(x), float(y))

    def compute_distance_meters(
        self,
        point_a_px: Tuple[float, float],
        point_b_px: Tuple[float, float],
    ) -> Optional[float]:
        """
        Calcule la distance réelle en mètres entre deux points pixel.
        C'est la puissance du Digital Twin.
        """
        a = self.project_point(point_a_px)
        b = self.project_point(point_b_px)

        if a is None or b is None:
            return None

        return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

    def save(self, filepath: str):
        """Sauvegarde la matrice d'homographie (avec toutes les scènes multi-caméra)."""
        data = {
            "H": self.H.tolist() if self.H is not None else None,
            "src_points": self.src_points.tolist() if self.src_points is not None else None,
            "dst_points": self.dst_points.tolist() if self.dst_points is not None else None,
            "reprojection_error": self.reprojection_error,
            "scene_homographies": self.scene_homographies,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Homographie] Sauvegardée dans {filepath}")

    def load(self, filepath: str):
        """Charge une matrice d'homographie sauvegardée."""
        with open(filepath, "r") as f:
            data = json.load(f)

        if data["H"] is not None:
            self.H = np.array(data["H"], dtype=np.float64)
            self.H_inv = np.linalg.inv(self.H)
        if data["src_points"] is not None:
            self.src_points = np.array(data["src_points"], dtype=np.float64)
        if data["dst_points"] is not None:
            self.dst_points = np.array(data["dst_points"], dtype=np.float64)
        self.reprojection_error = data.get("reprojection_error", float("inf"))

        # Charger les scènes multi-caméra (rétro-compatible avec l'ancien format)
        self.scene_homographies = data.get("scene_homographies", [])
        if not self.scene_homographies and self.H is not None:
            self.scene_homographies = [{
                "H": self.H.tolist(),
                "H_inv": self.H_inv.tolist(),
                "n_points": 0,
                "reprojection_error": self.reprojection_error,
            }]
        self._active_scene = 0

        n_sc = len(self.scene_homographies)
        suffix = f" — {n_sc} scène(s)" if n_sc > 1 else ""
        print(f"[Homographie] Chargée depuis {filepath}{suffix}")


class ManualCalibrator:
    """
    Calibration interactive avec navigation vidéo.

    L'utilisateur peut :
    - Naviguer dans la vidéo (trackbar + flèches) pour trouver la meilleure frame
    - Appuyer sur 'c' pour capturer une frame et marquer des points
    - Collecter des points depuis PLUSIEURS frames (multi-keyframe)
    - Les points de toutes les keyframes sont fusionnés pour un seul calcul H

    Cela résout le problème où certains repères (corners, surfaces)
    ne sont visibles que quand la caméra pointe dans une direction donnée.
    """

    def __init__(self, pitch: PitchTemplate = None):
        self.pitch = pitch or PitchTemplate()
        self.image_points: List[Tuple[float, float]] = []
        self.world_points: List[Tuple[float, float]] = []
        self.point_sources: List[int] = []  # frame_idx de chaque point
        self.current_frame: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────
    # API PUBLIQUE
    # ─────────────────────────────────────────────────────────

    def calibrate_interactive(
        self, frame: Optional[np.ndarray] = None, video_path: str = None
    ) -> HomographyEstimator:
        """
        Lance la calibration interactive.

        Si video_path est fourni → mode navigation vidéo complète.
        Sinon → mode frame unique (rétro-compatible).
        """
        if video_path is not None:
            return self._calibrate_with_video(video_path)
        else:
            return self._calibrate_single_frame(frame)

    # ─────────────────────────────────────────────────────────
    # MODE NAVIGATION VIDÉO (nouveau)
    # ─────────────────────────────────────────────────────────

    def _calibrate_with_video(self, video_path: str) -> HomographyEstimator:
        """
        Mode complet : navigation dans la vidéo + sélection multi-keyframe.

        Phase 1 — NAVIGATION : Parcourir la vidéo pour trouver une bonne frame
        Phase 2 — MARQUAGE  : Cliquer les points visibles sur cette frame
        Phase 3 — RÉPÉTITION: Naviguer vers une autre frame pour +de points
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # État de navigation
        nav_state = {
            "frame_idx": 0,
            "paused": True,
            "need_refresh": True,
        }

        window_name = "CALIBRATION - Navigation Video"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Trackbar pour scrubber
        def on_trackbar(val):
            nav_state["frame_idx"] = val
            nav_state["need_refresh"] = True

        cv2.createTrackbar("Frame", window_name, 0, max(total_frames - 1, 1), on_trackbar)

        self._print_nav_help(total_frames, fps)

        current_frame = None
        keyframes_used = []

        while True:
            # Lire la frame si besoin
            if nav_state["need_refresh"]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, nav_state["frame_idx"])
                ret, current_frame = cap.read()
                if not ret:
                    nav_state["frame_idx"] = max(0, nav_state["frame_idx"] - 1)
                    continue
                nav_state["need_refresh"] = False

            # Affichage
            display = current_frame.copy()
            self._draw_nav_hud(display, nav_state["frame_idx"], total_frames,
                               fps, len(self.image_points), keyframes_used)

            cv2.imshow(window_name, display)
            cv2.setTrackbarPos("Frame", window_name, nav_state["frame_idx"])

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                # Quitter la calibration
                break

            elif key == ord("c"):
                # CAPTURER cette frame → passer en mode marquage
                cv2.destroyAllWindows()
                kf_idx = nav_state["frame_idx"]
                print(f"\n[Calibration] Frame {kf_idx} capturée — Marquage des points...")
                keyframes_used.append(kf_idx)
                self._mark_points_on_frame(current_frame, kf_idx)

                # Recréer la fenêtre de navigation
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.createTrackbar("Frame", window_name, 0, max(total_frames - 1, 1), on_trackbar)
                cv2.setTrackbarPos("Frame", window_name, nav_state["frame_idx"])
                nav_state["need_refresh"] = True

                print(f"\n[Calibration] Total : {len(self.image_points)} points sur "
                      f"{len(keyframes_used)} keyframe(s).")
                print("  → 'c' = capturer une autre frame | 'q' = terminer\n")

            elif key == 81 or key == 2424832 or key == ord("a"):
                # Flèche gauche / 'a' → reculer 1 frame
                nav_state["frame_idx"] = max(0, nav_state["frame_idx"] - 1)
                nav_state["need_refresh"] = True

            elif key == 83 or key == 2555904 or key == ord("d"):
                # Flèche droite / 'd' → avancer 1 frame
                nav_state["frame_idx"] = min(total_frames - 1, nav_state["frame_idx"] + 1)
                nav_state["need_refresh"] = True

            elif key == ord(","):
                # Reculer 1 seconde
                nav_state["frame_idx"] = max(0, nav_state["frame_idx"] - int(fps))
                nav_state["need_refresh"] = True

            elif key == ord("."):
                # Avancer 1 seconde
                nav_state["frame_idx"] = min(total_frames - 1, nav_state["frame_idx"] + int(fps))
                nav_state["need_refresh"] = True

            elif key == ord("["):
                # Reculer 5 secondes
                nav_state["frame_idx"] = max(0, nav_state["frame_idx"] - int(fps * 5))
                nav_state["need_refresh"] = True

            elif key == ord("]"):
                # Avancer 5 secondes
                nav_state["frame_idx"] = min(total_frames - 1, nav_state["frame_idx"] + int(fps * 5))
                nav_state["need_refresh"] = True

            elif key == ord(" "):
                # Espace → lecture/pause
                nav_state["paused"] = not nav_state["paused"]

            # Lecture auto si pas en pause
            if not nav_state["paused"]:
                nav_state["frame_idx"] = min(total_frames - 1, nav_state["frame_idx"] + 1)
                nav_state["need_refresh"] = True
                if nav_state["frame_idx"] >= total_frames - 1:
                    nav_state["paused"] = True

        cap.release()
        cv2.destroyAllWindows()

        # Calculer l'homographie finale avec tous les points collectés
        return self._compute_final_homography(keyframes_used)

    def _mark_points_on_frame(self, frame: np.ndarray, frame_idx: int):
        """
        Mode marquage : clic sur les points du terrain visibles dans la frame capturée.
        Les points déjà marqués dans des keyframes précédentes sont en orange.
        """
        keypoints = self.pitch.get_all_keypoints()
        keypoint_names = list(keypoints.keys())

        # Identifier les points déjà marqués (sur d'autres keyframes)
        already_marked = set()
        for i, wp in enumerate(self.world_points):
            for name, ref_pt in keypoints.items():
                if abs(wp[0] - ref_pt[0]) < 0.1 and abs(wp[1] - ref_pt[1]) < 0.1:
                    already_marked.add(name)
                    break

        # Filtrer : ne proposer que les points pas encore marqués
        remaining_names = [n for n in keypoint_names if n not in already_marked]
        if not remaining_names:
            print("[Calibration] Tous les points sont déjà marqués!")
            return

        current_idx = [0]
        display = frame.copy()
        local_image_pts = []
        local_world_pts = []

        window_name = f"Marquage - Frame {frame_idx}"

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if current_idx[0] < len(remaining_names):
                    name = remaining_names[current_idx[0]]
                    world_pt = keypoints[name]

                    local_image_pts.append((float(x), float(y)))
                    local_world_pts.append(world_pt)

                    cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
                    cv2.putText(display, name, (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    current_idx[0] += 1
                    print(f"  + {name} -> pixel ({x}, {y}) -> terrain {world_pt}")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_click)

        print(f"\n  Points restants ({len(remaining_names)}):")
        for i, name in enumerate(remaining_names):
            pt = keypoints[name]
            print(f"    {i+1:2d}. {name:<25s} -> ({pt[0]:.1f}, {pt[1]:.1f})")
        print(f"  Déjà marqués : {len(already_marked)}")
        print(f"  Touches : clic=marquer | 's'=skip | 'u'=undo | 'q'=terminé\n")

        while True:
            info = display.copy()

            # Dessiner les points déjà marqués (orange)
            for ip in self.image_points:
                cv2.circle(info, (int(ip[0]), int(ip[1])), 4, (0, 140, 255), -1)

            # HUD
            if current_idx[0] < len(remaining_names):
                name = remaining_names[current_idx[0]]
                wp = keypoints[name]
                cv2.putText(info,
                    f"Cliquez: {name} ({wp[0]:.0f}m, {wp[1]:.0f}m)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(info,
                    f"'s'=SKIP | 'u'=undo | 'q'=valider cette frame",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            else:
                cv2.putText(info, "Tous les points restants faits! 'q' pour valider",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            total = len(self.image_points) + len(local_image_pts)
            cv2.putText(info, f"Points cette frame: {len(local_image_pts)} | "
                        f"Total global: {total}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(window_name, info)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                if current_idx[0] < len(remaining_names):
                    skipped = remaining_names[current_idx[0]]
                    current_idx[0] += 1
                    print(f"  SKIP: {skipped}")
            elif key == ord("u") and local_image_pts:
                local_image_pts.pop()
                local_world_pts.pop()
                current_idx[0] = max(0, current_idx[0] - 1)
                display = frame.copy()
                for ip in local_image_pts:
                    cv2.circle(display, (int(ip[0]), int(ip[1])), 6, (0, 255, 0), -1)

        cv2.destroyAllWindows()

        # Ajouter les points de cette keyframe au total
        self.image_points.extend(local_image_pts)
        self.world_points.extend(local_world_pts)
        self.point_sources.extend([frame_idx] * len(local_image_pts))

        print(f"  → {len(local_image_pts)} point(s) ajouté(s) depuis frame {frame_idx}")

    def _draw_nav_hud(self, frame: np.ndarray, idx: int, total: int,
                      fps: float, n_points: int, keyframes: List[int]):
        """Dessine le HUD de navigation sur la frame."""
        h, w = frame.shape[:2]

        # Bande noire semi-transparente en haut
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Infos frame
        time_s = idx / fps if fps > 0 else 0
        total_s = total / fps if fps > 0 else 0
        cv2.putText(frame,
            f"Frame {idx}/{total} | {time_s:.1f}s / {total_s:.1f}s",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Contrôles
        cv2.putText(frame,
            "A/D=frame | ,/.=1s | [/]=5s | ESPACE=play | C=capturer | Q=fin",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        # Stats
        kf_str = ", ".join(str(k) for k in keyframes) if keyframes else "aucune"
        cv2.putText(frame,
            f"Points: {n_points} | Keyframes: [{kf_str}]",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)

        # Barre de progression visuelle en bas
        bar_y = h - 8
        bar_h = 6
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (40, 40, 40), -1)
        progress = int(w * idx / max(total - 1, 1))
        cv2.rectangle(frame, (0, bar_y), (progress, bar_y + bar_h), (0, 255, 200), -1)

        # Marqueurs des keyframes capturées sur la barre
        for kf in keyframes:
            kf_x = int(w * kf / max(total - 1, 1))
            cv2.rectangle(frame, (kf_x - 2, bar_y - 4), (kf_x + 2, bar_y + bar_h + 4),
                         (0, 140, 255), -1)

    def _print_nav_help(self, total_frames: int, fps: float):
        """Affiche l'aide console pour le mode navigation."""
        print("\n" + "=" * 65)
        print("  CALIBRATION INTERACTIVE — Navigation Vidéo")
        print("=" * 65)
        print(f"  Vidéo : {total_frames} frames | {fps:.0f} FPS | {total_frames/fps:.1f}s")
        print("-" * 65)
        print("  NAVIGATION:")
        print("    A / D         → Frame précédente / suivante")
        print("    , / .         → Reculer / avancer 1 seconde")
        print("    [ / ]         → Reculer / avancer 5 secondes")
        print("    ESPACE        → Lecture / Pause")
        print("    Trackbar      → Scrubber direct")
        print()
        print("  CALIBRATION:")
        print("    C             → Capturer cette frame pour marquer des points")
        print("    Q             → Terminer et calculer l'homographie")
        print()
        print("  WORKFLOW:")
        print("    1. Naviguez vers une frame où des repères sont bien visibles")
        print("    2. Appuyez 'C' pour capturer et marquer les points")
        print("    3. Revenez en navigation, allez sur une AUTRE frame")
        print("    4. Appuyez 'C' pour ajouter des points supplémentaires")
        print("    5. Répétez. Appuyez 'Q' quand vous avez ≥ 4 points au total")
        print("=" * 65 + "\n")

    def _compute_final_homography(
        self, keyframes_used: List[int]
    ) -> HomographyEstimator:
        """Calcule l'homographie finale à partir de tous les points collectés."""
        estimator = HomographyEstimator(self.pitch)

        print(f"\n[Calibration] Calcul de l'homographie finale...")
        print(f"  Points totaux : {len(self.image_points)}")
        print(f"  Keyframes     : {keyframes_used}")

        if len(self.image_points) >= 4:
            estimator.compute_from_correspondences(
                self.image_points, self.world_points
            )
        else:
            print("[Calibration] ERREUR: Au moins 4 points sont nécessaires!")

        return estimator

    # ─────────────────────────────────────────────────────────
    # MODE FRAME UNIQUE (rétro-compatible)
    # ─────────────────────────────────────────────────────────

    def _calibrate_single_frame(self, frame: np.ndarray) -> HomographyEstimator:
        """Mode original : calibration sur une seule frame statique."""
        self.current_frame = frame.copy()
        self.image_points = []
        self.world_points = []

        keypoints = self.pitch.get_all_keypoints()
        keypoint_names = list(keypoints.keys())
        current_idx = [0]

        display = frame.copy()
        window_name = "Calibration - Cliquer sur les points du terrain"

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if current_idx[0] < len(keypoint_names):
                    name = keypoint_names[current_idx[0]]
                    world_pt = keypoints[name]

                    self.image_points.append((float(x), float(y)))
                    self.world_points.append(world_pt)

                    cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(display, name, (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    current_idx[0] += 1
                    print(f"[Calibration] Point {name} -> image ({x}, {y}) "
                          f"-> terrain {world_pt}")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_click)

        print("\n" + "=" * 60)
        print("CALIBRATION MANUELLE DE L'HOMOGRAPHIE")
        print("=" * 60)
        print("Cliquez sur les points VISIBLES dans l'image.")
        print("  's' = SKIP (point pas visible)")
        print("  'u' = Annuler le dernier point")
        print("  'q' = Terminer (min 4 points)")
        print("-" * 60)
        print("Points disponibles :")
        for i, name in enumerate(keypoint_names):
            pt = keypoints[name]
            print(f"  {i + 1:2d}. {name:<20s} -> ({pt[0]:.1f}, {pt[1]:.1f})")
        print("=" * 60 + "\n")

        while True:
            info = display.copy()
            if current_idx[0] < len(keypoint_names):
                name = keypoint_names[current_idx[0]]
                world_pt = keypoints[name]
                cv2.putText(info, f"Cliquez sur: {name} ({world_pt[0]:.0f}m, {world_pt[1]:.0f}m)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)
                cv2.putText(info, f"'s'=SKIP si pas visible | 'q'=terminer | 'u'=undo",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            else:
                cv2.putText(info, "Tous les points faits! 'q' pour terminer",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
            cv2.putText(info, f"Points marques: {len(self.image_points)}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(window_name, info)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                if current_idx[0] < len(keypoint_names):
                    skipped = keypoint_names[current_idx[0]]
                    current_idx[0] += 1
                    print(f"[Calibration] SKIP: {skipped} (pas visible)")
            elif key == ord("u") and self.image_points:
                self.image_points.pop()
                self.world_points.pop()
                current_idx[0] -= 1
                display = frame.copy()
                for i, ip in enumerate(self.image_points):
                    cv2.circle(display, (int(ip[0]), int(ip[1])), 5, (0, 255, 0), -1)

        cv2.destroyAllWindows()

        estimator = HomographyEstimator(self.pitch)
        if len(self.image_points) >= 4:
            estimator.compute_from_correspondences(
                self.image_points, self.world_points
            )
        else:
            print("[Calibration] ERREUR: Au moins 4 points sont nécessaires!")

        return estimator


class AutoCalibrator:
    """
    Calibration automatique basée sur la détection de lignes.
    Tente de trouver automatiquement les correspondances
    terrain-image à partir des intersections de lignes.
    """

    def __init__(self, pitch: PitchTemplate = None):
        self.pitch = pitch or PitchTemplate()

    def estimate_from_lines(
        self,
        frame: np.ndarray,
        line_mask: np.ndarray,
        detected_lines: List[np.ndarray],
        intersections: List[Tuple[float, float]],
    ) -> Optional[HomographyEstimator]:
        """
        Tente une calibration automatique.
        
        Stratégie simplifiée :
        1. Classer les lignes en horizontales/verticales
        2. Identifier les lignes principales (touche, 16m, milieu)
        3. Déduire les correspondances
        
        Note: Cette méthode est approximative. Pour un résultat précis,
        utiliser la calibration manuelle ou un modèle de segmentation dédié.
        """
        h, w = frame.shape[:2]

        # Séparer lignes horizontales et verticales
        horizontal = []
        vertical = []

        for line in detected_lines:
            x1, y1, x2, y2 = line
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 30 or angle > 150:
                horizontal.append(line)
            elif 60 < angle < 120:
                vertical.append(line)

        if len(horizontal) < 2 or len(vertical) < 2:
            print("[AutoCalibration] Pas assez de lignes détectées.")
            return None

        # Trier par position
        horizontal.sort(key=lambda l: (l[1] + l[3]) / 2)
        vertical.sort(key=lambda l: (l[0] + l[2]) / 2)

        # Stratégie basique : prendre les lignes extrêmes comme bords
        top_line = horizontal[0]
        bottom_line = horizontal[-1]
        left_line = vertical[0]
        right_line = vertical[-1]

        # Calculer les 4 coins à partir des intersections
        from .segmentation import PitchSegmenter
        segmenter = PitchSegmenter()

        corners_image = []
        for h_line in [top_line, bottom_line]:
            for v_line in [left_line, right_line]:
                pt = segmenter._line_intersection(h_line, v_line)
                if pt is not None:
                    corners_image.append(pt)

        if len(corners_image) < 4:
            print("[AutoCalibration] Impossible de trouver 4 coins.")
            return None

        # Associer aux coins du terrain
        corners_world = [
            (0.0, 0.0),
            (self.pitch.length, 0.0),
            (0.0, self.pitch.width),
            (self.pitch.length, self.pitch.width),
        ]

        estimator = HomographyEstimator(self.pitch)
        estimator.compute_from_correspondences(corners_image, corners_world)

        return estimator
