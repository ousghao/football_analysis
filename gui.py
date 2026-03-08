"""
╔══════════════════════════════════════════════════════════════════════╗
║        FOOTBALL TACTICAL INTELLIGENCE — Interface Graphique        ║
║                                                                    ║
║  GUI Tkinter complète pour piloter tout le pipeline d'analyse :    ║
║   • Sélection vidéo + configuration                                ║
║   • Calibration interactive (homographie)                          ║
║   • Lancement de l'analyse avec barre de progression               ║
║   • Lecteur vidéo intégré (résultat annoté)                        ║
║   • Visualisation des rapports et dashboards                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import threading
import webbrowser
import time
import math
from pathlib import Path
from io import StringIO

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ─── Imports du projet ──────────────────────────────────────
from config import OUTPUT_DIR, DATA_DIR
from main import FootballAnalysisPipeline
from src.pitch.pitch_template import PitchTemplate
from src.pitch.homography import HomographyEstimator


# ═══════════════════════════════════════════════════════════════
#  CONSTANTES STYLE
# ═══════════════════════════════════════════════════════════════
BG_DARK = "#1a1a2e"
BG_PANEL = "#16213e"
BG_CARD = "#0f3460"
ACCENT = "#e94560"
ACCENT_HOVER = "#ff6b6b"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#a8a8b3"
TEXT_MUTED = "#6c6c80"
SUCCESS = "#00d26a"
WARNING = "#f5a623"
ENTRY_BG = "#1e2a45"
BTN_BG = "#e94560"
BTN_SECONDARY = "#0f3460"


# ═══════════════════════════════════════════════════════════════
#  FENÊTRE DE CALIBRATION INTERACTIVE
# ═══════════════════════════════════════════════════════════════

class CalibrationWindow:
    """
    Fenêtre Tkinter pour la calibration interactive de l'homographie.

    Workflow:
      1. La vidéo joue dans un canvas — l'utilisateur navigue
      2. Appuyer sur C (ou bouton) pour capturer la frame
      3. Cliquer sur la frame capturée pour marquer des points du terrain
      4. Valider les points → revenir en mode navigation
      5. Répéter pour d'autres frames (multi-keyframe)
      6. Terminer → calcul homographie → sauvegarde

    Les points déjà marqués sont dessinés dynamiquement sur la vidéo.
    """

    PITCH_PANEL_W = 320
    PITCH_PANEL_H = 220

    def __init__(self, master: tk.Tk, video_path: str, on_complete=None):
        self.master = master
        self.video_path = video_path
        self.on_complete = on_complete  # callback(estimator, h_path_or_None)

        self.pitch = PitchTemplate()
        self.keypoints = self.pitch.get_all_keypoints()
        self.keypoint_names = list(self.keypoints.keys())

        # Points collectés
        self.image_points: list = []
        self.world_points: list = []
        self.point_sources: list = []  # frame_idx per point
        self.keyframes_used: list = []

        # Video state
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Erreur", f"Impossible d'ouvrir la vidéo:\n{video_path}")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_idx = 0
        self.playing = False
        self.current_frame = None     # raw BGR frame
        self.after_id = None

        # Marking state
        self.marking_mode = False
        self.captured_frame = None
        self.remaining_names: list = []
        self.marking_idx = 0
        self.local_image_pts: list = []
        self.local_world_pts: list = []

        self._build_window()
        self._read_frame(0)

    # ─────────────────────────────────────────────────────
    #  BUILD UI
    # ─────────────────────────────────────────────────────

    def _build_window(self):
        self.win = tk.Toplevel(self.master)
        self.win.title("🎯 Calibration Interactive — Homographie")
        self.win.geometry("1300x780")
        self.win.minsize(1100, 650)
        self.win.configure(bg=BG_DARK)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self.win.grab_set()  # modal

        # ── Layout: left (video) + right (pitch diagram + point list) ──
        self.win.columnconfigure(0, weight=1)
        self.win.columnconfigure(1, weight=0)
        self.win.rowconfigure(1, weight=1)

        # ── Top bar ────────────────────────────────────
        top = tk.Frame(self.win, bg=BG_PANEL)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)

        tk.Label(top, text="🎯 CALIBRATION INTERACTIVE",
                 font=("Segoe UI", 13, "bold"), fg=ACCENT, bg=BG_PANEL
                 ).pack(side="left", padx=15, pady=8)

        self.lbl_status = tk.Label(top, text="Mode Navigation — Naviguez dans la vidéo puis appuyez C pour capturer",
                                   font=("Segoe UI", 10), fg=TEXT_SECONDARY, bg=BG_PANEL)
        self.lbl_status.pack(side="left", padx=20)

        # Conseil tip (right side)
        tk.Label(top, text="💡 Conseil: Calibrez depuis UNE SEULE position caméra pour de meilleurs résultats",
                 font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_PANEL
                 ).pack(side="right", padx=5)

        self.lbl_points_count = tk.Label(top, text="Points: 0",
                                         font=("Segoe UI", 10, "bold"), fg=SUCCESS, bg=BG_PANEL)
        self.lbl_points_count.pack(side="right", padx=15)

        # ── Left: Video canvas ─────────────────────────
        left = tk.Frame(self.win, bg=BG_DARK)
        left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(left, bg="#0d0d1a", highlightthickness=0, cursor="crosshair")
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self.video_canvas.bind("<Button-1>", self._on_canvas_click)
        self.video_canvas.bind("<Configure>", lambda e: self._refresh_display())

        # Controls below canvas
        ctrl = tk.Frame(left, bg=BG_PANEL)
        ctrl.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        ctrl.columnconfigure(3, weight=1)

        self.btn_play = tk.Button(ctrl, text="▶", font=("Segoe UI", 12, "bold"),
                                  bg=ACCENT, fg=TEXT_PRIMARY, relief="flat", width=4,
                                  command=self._toggle_play)
        self.btn_play.grid(row=0, column=0, padx=5, pady=5)

        self.btn_capture = tk.Button(ctrl, text="📷 Capturer (C)", font=("Segoe UI", 10, "bold"),
                                     bg="#ff8c00", fg=TEXT_PRIMARY, relief="flat",
                                     padx=12, command=self._capture_frame)
        self.btn_capture.grid(row=0, column=1, padx=5, pady=5)

        self.btn_validate = tk.Button(ctrl, text="✅ Valider points", font=("Segoe UI", 10, "bold"),
                                      bg=SUCCESS, fg=TEXT_PRIMARY, relief="flat",
                                      padx=12, command=self._validate_marking, state="disabled")
        self.btn_validate.grid(row=0, column=2, padx=5, pady=5)

        self.scrubber = tk.Scale(ctrl, from_=0, to=max(self.total_frames - 1, 1),
                                 orient="horizontal", bg=BG_PANEL, fg=TEXT_PRIMARY,
                                 troughcolor=ENTRY_BG, highlightthickness=0,
                                 sliderrelief="flat", showvalue=False,
                                 command=self._on_scrub)
        self.scrubber.grid(row=0, column=3, sticky="ew", padx=10, pady=5)

        self.lbl_time = tk.Label(ctrl, text="0:00 / 0:00", font=("Consolas", 10),
                                 fg=TEXT_SECONDARY, bg=BG_PANEL)
        self.lbl_time.grid(row=0, column=4, padx=5)

        # Mark indicator beneath scrubber
        self.mark_canvas = tk.Canvas(ctrl, bg=BG_PANEL, height=10, highlightthickness=0)
        self.mark_canvas.grid(row=1, column=3, sticky="ew", padx=10, pady=(0, 3))

        # Keyboard hint bar
        hint = tk.Frame(left, bg=BG_DARK)
        hint.grid(row=2, column=0, sticky="ew", pady=(3, 0))
        tk.Label(hint, text="⌨  ESPACE=Play/Pause  |  A/D=±1 frame  |  ,/.=±1s  |  [/]=±5s  |  C=Capturer  |  Q=Fin",
                 font=("Consolas", 9), fg=TEXT_MUTED, bg=BG_DARK).pack(anchor="w", padx=5)

        # Marking-mode info bar (hidden initially)
        self.marking_bar = tk.Frame(left, bg="#331a00")
        self.marking_bar.grid(row=3, column=0, sticky="ew", pady=(3, 0))
        self.marking_bar.grid_remove()
        self.lbl_marking_info = tk.Label(self.marking_bar,
                                         text="", font=("Segoe UI", 11, "bold"),
                                         fg="#ffdd57", bg="#331a00")
        self.lbl_marking_info.pack(side="left", padx=10, pady=4)
        tk.Label(self.marking_bar, text="Clic = marquer  |  S = skip  |  U = undo",
                 font=("Segoe UI", 9), fg="#ccaa44", bg="#331a00").pack(side="right", padx=10)

        # ── Right: Pitch diagram + Point list ──────────
        right = tk.Frame(self.win, bg=BG_DARK, width=350)
        right.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        # Pitch diagram
        tk.Label(right, text="TERRAIN — Points de Référence",
                 font=("Segoe UI", 10, "bold"), fg=TEXT_MUTED, bg=BG_DARK
                 ).grid(row=0, column=0, sticky="w", pady=(0, 3))

        self.pitch_canvas = tk.Canvas(right, bg="#0a3d0a", highlightthickness=1,
                                      highlightbackground="#333",
                                      width=self.PITCH_PANEL_W, height=self.PITCH_PANEL_H)
        self.pitch_canvas.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self._draw_pitch_diagram()

        # Point list
        tk.Label(right, text="POINTS MARQUÉS",
                 font=("Segoe UI", 10, "bold"), fg=TEXT_MUTED, bg=BG_DARK
                 ).grid(row=2, column=0, sticky="nw", pady=(0, 3))

        list_frame = tk.Frame(right, bg=ENTRY_BG)
        list_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.points_listbox = tk.Listbox(list_frame, bg=ENTRY_BG, fg=TEXT_PRIMARY,
                                         font=("Consolas", 9), relief="flat",
                                         selectbackground=ACCENT, activestyle="none")
        self.points_listbox.grid(row=0, column=0, sticky="nsew")
        sb = tk.Scrollbar(list_frame, orient="vertical", command=self.points_listbox.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.points_listbox.config(yscrollcommand=sb.set)

        # Bottom buttons
        btn_bar = tk.Frame(right, bg=BG_DARK)
        btn_bar.grid(row=4, column=0, sticky="ew")

        self.btn_finish = tk.Button(btn_bar, text="🏁 Calculer Homographie",
                                    font=("Segoe UI", 11, "bold"),
                                    bg=ACCENT, fg=TEXT_PRIMARY, relief="flat",
                                    padx=15, pady=6, command=self._finish)
        self.btn_finish.pack(fill="x", pady=(0, 4))

        self.btn_cancel = tk.Button(btn_bar, text="Annuler",
                                    font=("Segoe UI", 9),
                                    bg=BTN_SECONDARY, fg=TEXT_PRIMARY, relief="flat",
                                    command=self._on_close)
        self.btn_cancel.pack(fill="x")

        # Bind keys
        self.win.bind("<Key>", self._on_key)
        self.win.focus_set()

    # ─────────────────────────────────────────────────────
    #  PITCH DIAGRAM
    # ─────────────────────────────────────────────────────

    def _draw_pitch_diagram(self):
        """Draw a mini pitch diagram showing keypoint positions."""
        c = self.pitch_canvas
        c.delete("all")
        pw, ph = self.PITCH_PANEL_W, self.PITCH_PANEL_H
        margin = 15
        field_w = pw - 2 * margin
        field_h = ph - 2 * margin

        def m2px(pt):
            x = margin + pt[0] / 105.0 * field_w
            y = margin + pt[1] / 68.0 * field_h
            return (x, y)

        # Field outline
        c.create_rectangle(margin, margin, margin + field_w, margin + field_h,
                           outline="white", width=1)
        # Halfway line
        mx = margin + field_w / 2
        c.create_line(mx, margin, mx, margin + field_h, fill="white", width=1)
        # Center circle
        cr = 9.15 / 105.0 * field_w
        cx, cy = m2px((52.5, 34.0))
        c.create_oval(cx - cr, cy - cr, cx + cr, cy + cr, outline="white", width=1)

        # Penalty areas
        for pts in [self.pitch.penalty_area_left, self.pitch.penalty_area_right]:
            coords = [m2px(p) for p in pts]
            c.create_polygon(*[v for p in coords for v in p], outline="white", fill="", width=1)

        # Goal areas
        for pts in [self.pitch.goal_area_left, self.pitch.goal_area_right]:
            coords = [m2px(p) for p in pts]
            c.create_polygon(*[v for p in coords for v in p], outline="white", fill="", width=1)

        # Keypoints — gray dots for unmarked, green for marked
        marked_names = self._get_marked_names()
        for name, wpt in self.keypoints.items():
            px, py = m2px(wpt)
            if name in marked_names:
                c.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#00ff66", outline="")
            else:
                c.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#888888", outline="")

    def _get_marked_names(self) -> set:
        """Return set of keypoint names that have already been marked."""
        marked = set()
        for wp in self.world_points:
            for name, ref_pt in self.keypoints.items():
                if abs(wp[0] - ref_pt[0]) < 0.1 and abs(wp[1] - ref_pt[1]) < 0.1:
                    marked.add(name)
                    break
        return marked

    # ─────────────────────────────────────────────────────
    #  VIDEO DISPLAY
    # ─────────────────────────────────────────────────────

    def _read_frame(self, idx):
        """Read a specific frame from the video."""
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.frame_idx = idx
            self._refresh_display()
            self._update_time_label()

    def _refresh_display(self):
        """Render the current frame (or captured frame) on the canvas with overlays."""
        frame = self.captured_frame if self.marking_mode else self.current_frame
        if frame is None:
            return

        canvas = self.video_canvas
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        display = frame.copy()

        # Draw previously marked points as orange circles
        for i, ip in enumerate(self.image_points):
            ix, iy = int(ip[0]), int(ip[1])
            cv2.circle(display, (ix, iy), 6, (0, 140, 255), -1)
            cv2.circle(display, (ix, iy), 8, (0, 100, 200), 2)

        # Draw local (current marking session) points as green
        if self.marking_mode:
            for ip in self.local_image_pts:
                ix, iy = int(ip[0]), int(ip[1])
                cv2.circle(display, (ix, iy), 7, (0, 255, 0), -1)
                cv2.circle(display, (ix, iy), 9, (0, 200, 0), 2)

        # Scale to fit canvas
        h, w = display.shape[:2]
        self._scale = min(cw / w, ch / h)
        new_w, new_h = int(w * self._scale), int(h * self._scale)
        self._offset_x = (cw - new_w) // 2
        self._offset_y = (ch - new_h) // 2
        self._frame_w = w
        self._frame_h = h

        resized = cv2.resize(display, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        canvas._photo = photo

    def _update_time_label(self):
        t = self.frame_idx / max(self.fps, 1)
        total = self.total_frames / max(self.fps, 1)
        cm, cs = divmod(int(t), 60)
        tm, ts = divmod(int(total), 60)
        self.lbl_time.config(text=f"{cm}:{cs:02d} / {tm}:{ts:02d}")
        self.scrubber.set(self.frame_idx)
        self._draw_keyframe_marks()

    def _draw_keyframe_marks(self):
        """Draw keyframe markers on the mark canvas."""
        c = self.mark_canvas
        c.delete("all")
        c.update_idletasks()
        w = c.winfo_width()
        if w < 10 or self.total_frames < 2:
            return
        for kf in self.keyframes_used:
            x = int(w * kf / max(self.total_frames - 1, 1))
            c.create_rectangle(x - 2, 0, x + 2, 10, fill="#ff8c00", outline="")

    # ─────────────────────────────────────────────────────
    #  PLAYBACK
    # ─────────────────────────────────────────────────────

    def _toggle_play(self):
        if self.marking_mode:
            return
        self.playing = not self.playing
        self.btn_play.config(text="⏸" if self.playing else "▶")
        if self.playing:
            self._play_loop()

    def _play_loop(self):
        if not self.playing or self.marking_mode:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.playing = False
            self.btn_play.config(text="▶")
            return
        self.current_frame = frame
        self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self._refresh_display()
        self._update_time_label()
        delay = max(1, int(1000 / self.fps))
        self.after_id = self.win.after(delay, self._play_loop)

    def _on_scrub(self, val):
        if self.playing or self.marking_mode:
            return
        self._read_frame(int(float(val)))

    # ─────────────────────────────────────────────────────
    #  CAPTURE & MARKING
    # ─────────────────────────────────────────────────────

    def _capture_frame(self):
        """Enter marking mode on the current frame."""
        if self.marking_mode:
            return
        if self.current_frame is None:
            return

        # Stop playback
        self.playing = False
        self.btn_play.config(text="▶")
        if self.after_id:
            self.win.after_cancel(self.after_id)
            self.after_id = None

        self.captured_frame = self.current_frame.copy()
        self.marking_mode = True

        # Figure out remaining keypoints
        marked = self._get_marked_names()
        self.remaining_names = [n for n in self.keypoint_names if n not in marked]
        self.marking_idx = 0
        self.local_image_pts = []
        self.local_world_pts = []

        # UI updates
        self.btn_capture.config(state="disabled")
        self.btn_play.config(state="disabled")
        self.btn_validate.config(state="normal")
        self.marking_bar.grid()
        self._update_marking_info()
        self.lbl_status.config(text="Mode Marquage — Cliquez sur les points du terrain dans la vidéo",
                               fg="#ffdd57")
        self._refresh_display()

    def _validate_marking(self):
        """Validate current marking session and return to navigation."""
        if not self.marking_mode:
            return

        # Add local points to global
        kf_idx = self.frame_idx
        self.image_points.extend(self.local_image_pts)
        self.world_points.extend(self.local_world_pts)
        self.point_sources.extend([kf_idx] * len(self.local_image_pts))
        if self.local_image_pts:
            self.keyframes_used.append(kf_idx)

        # Exit marking mode
        self.marking_mode = False
        self.captured_frame = None
        self.btn_capture.config(state="normal")
        self.btn_play.config(state="normal")
        self.btn_validate.config(state="disabled")
        self.marking_bar.grid_remove()
        self.lbl_status.config(text="Mode Navigation — Naviguez dans la vidéo puis appuyez C pour capturer",
                               fg=TEXT_SECONDARY)

        self._update_points_ui()
        self._refresh_display()

        # Compute partial H quality feedback if we have enough points
        self._check_calibration_quality()

    def _check_calibration_quality(self):
        """Compute a partial H from current points and show quality feedback."""
        n = len(self.image_points)
        if n < 4:
            self.lbl_points_count.config(
                text=f"Points: {n}  (min 4 pour calibrer)",
                fg=WARNING
            )
            return

        try:
            estimator = HomographyEstimator(self.pitch)
            estimator.compute_from_correspondences(self.image_points, self.world_points)
            err = estimator.reprojection_error

            if err < 1.5:
                quality = "✅ Excellente"
                color = SUCCESS
            elif err < 3.0:
                quality = "✔ Bonne"
                color = "#66dd66"
            elif err < 5.0:
                quality = "⚠ Moyenne"
                color = WARNING
            else:
                quality = "❌ Mauvaise"
                color = ACCENT

            self.lbl_points_count.config(
                text=f"Points: {n}   Qualité H: {quality} ({err:.1f}m erreur)",
                fg=color
            )

            if err > 4.0:
                # Find which points are outliers (error > 3m)
                bad_names = []
                if estimator.H is not None:
                    for ip, wp in zip(self.image_points, self.world_points):
                        proj = estimator.project_point(ip)
                        if proj:
                            e = math.sqrt((proj[0]-wp[0])**2 + (proj[1]-wp[1])**2)
                            if e > 3.0:
                                for name, ref in self.keypoints.items():
                                    if abs(wp[0]-ref[0]) < 0.1 and abs(wp[1]-ref[1]) < 0.1:
                                        bad_names.append(f"{name}({e:.1f}m)")
                                        break

                warning = (
                    f"⚠ Erreur de calibration élevée ({err:.1f}m)!\n\n"
                    "Cause probable : vous avez marqué des points depuis\n"
                    "des positions caméra DIFFÉRENTES (camera panning).\n\n"
                    "CONSEIL: Ne capturez qu'UNE SEULE position caméra\n"
                    "(celle avec le plus de repères visibles).\n\n"
                )
                if bad_names:
                    warning += f"Points suspects : {', '.join(bad_names[:5])}"
                messagebox.showwarning("Qualité de calibration faible", warning,
                                       parent=self.win)

            # ── Vérification de la couverture spatiale ────────────────
            # Si tous les points sont d'un seul côté du terrain, la minimap
            # sera incorrecte quand la caméra montre l'autre moitié.
            xs = [wp[0] for wp in self.world_points]
            FIELD_MID = 52.5  # ligne médiane
            has_left  = any(x < FIELD_MID for x in xs)
            has_right = any(x > FIELD_MID for x in xs)
            if not has_left or not has_right:
                missing = "gauche (x < 52.5m)" if not has_left else "droite (x > 52.5m)"
                present = "droite" if not has_left else "gauche"
                messagebox.showwarning(
                    "Couverture incomplète",
                    f"⚠ Tous vos points de calibration sont dans la moitié {present} du terrain !\n\n"
                    f"Côté manquant : {missing}\n\n"
                    "CONSÉQUENCE : Quand la caméra montre l'autre moitié,\n"
                    "la minimap affichera des positions incorrectes.\n\n"
                    "SOLUTION :\n"
                    "  1. Revenez en mode Navigation (la fenêtre reste ouverte)\n"
                    "  2. Avancez jusqu'à une frame montrant l'autre moitié du terrain\n"
                    "  3. Appuyez C → marquez les repères visibles de ce côté\n"
                    "  4. Validez → recalibrez\n\n"
                    "(Utilisez '.' ou ']' pour avancer rapidement dans la vidéo)",
                    parent=self.win
                )

        except Exception:
            pass

    def _on_canvas_click(self, event):
        """Handle mouse click on the video canvas for point marking."""
        if not self.marking_mode:
            return
        if self.marking_idx >= len(self.remaining_names):
            return

        # Convert canvas coords to frame coords
        if not hasattr(self, '_scale') or self._scale <= 0:
            return
        fx = (event.x - self._offset_x) / self._scale
        fy = (event.y - self._offset_y) / self._scale
        if fx < 0 or fy < 0 or fx >= self._frame_w or fy >= self._frame_h:
            return

        name = self.remaining_names[self.marking_idx]
        world_pt = self.keypoints[name]

        self.local_image_pts.append((float(fx), float(fy)))
        self.local_world_pts.append(world_pt)
        self.marking_idx += 1

        self._update_marking_info()
        self._refresh_display()

        # Update temp point list
        self._update_points_ui()

    def _update_marking_info(self):
        """Update the marking info bar with current point name."""
        if self.marking_idx < len(self.remaining_names):
            name = self.remaining_names[self.marking_idx]
            wp = self.keypoints[name]
            self.lbl_marking_info.config(
                text=f"▸ Cliquer sur : {name}  ({wp[0]:.0f}m, {wp[1]:.0f}m)   "
                     f"[{self.marking_idx + 1}/{len(self.remaining_names)} restants]"
            )
        else:
            self.lbl_marking_info.config(
                text=f"✔ Tous les points restants sont faits ! Cliquez 'Valider points'"
            )

    def _update_points_ui(self):
        """Refresh the points listbox and pitch diagram."""
        self.points_listbox.delete(0, "end")
        all_img = self.image_points + self.local_image_pts
        all_world = self.world_points + self.local_world_pts

        for i, (ip, wp) in enumerate(zip(all_img, all_world)):
            # Find the name
            pname = "?"
            for name, ref in self.keypoints.items():
                if abs(wp[0] - ref[0]) < 0.1 and abs(wp[1] - ref[1]) < 0.1:
                    pname = name
                    break
            self.points_listbox.insert("end",
                f"{i+1:2d}. {pname:<22s} px({ip[0]:.0f},{ip[1]:.0f})")

        total = len(all_img)
        self.lbl_points_count.config(text=f"Points: {total}")
        self._draw_pitch_diagram()

        # Temporarily include local points for diagram
        if self.marking_mode and self.local_world_pts:
            c = self.pitch_canvas
            pw, ph = self.PITCH_PANEL_W, self.PITCH_PANEL_H
            margin = 15
            field_w = pw - 2 * margin
            field_h = ph - 2 * margin
            for wp in self.local_world_pts:
                px = margin + wp[0] / 105.0 * field_w
                py = margin + wp[1] / 68.0 * field_h
                c.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#00ff00", outline="white", width=1)

    # ─────────────────────────────────────────────────────
    #  KEYBOARD
    # ─────────────────────────────────────────────────────

    def _on_key(self, event):
        k = event.keysym.lower()
        ch = event.char

        if self.marking_mode:
            if ch == "s" or k == "s":
                self._skip_point()
            elif ch == "u" or k == "u":
                self._undo_point()
            elif ch == "q" or k == "q":
                self._validate_marking()
            return

        # Navigation mode
        if ch == " " or k == "space":
            self._toggle_play()
        elif ch == "c" or k == "c":
            self._capture_frame()
        elif ch == "q" or k == "q":
            self._finish()
        elif ch == "a" or k == "a" or k == "left":
            self._step(-1)
        elif ch == "d" or k == "d" or k == "right":
            self._step(1)
        elif ch == "," or k == "comma":
            self._step(-int(self.fps))
        elif ch == "." or k == "period":
            self._step(int(self.fps))
        elif ch == "[" or k == "bracketleft":
            self._step(-int(self.fps * 5))
        elif ch == "]" or k == "bracketright":
            self._step(int(self.fps * 5))

    def _step(self, delta):
        if self.playing or self.marking_mode:
            return
        new_idx = max(0, min(self.total_frames - 1, self.frame_idx + delta))
        self._read_frame(new_idx)

    def _skip_point(self):
        if self.marking_idx < len(self.remaining_names):
            self.marking_idx += 1
            self._update_marking_info()

    def _undo_point(self):
        if self.local_image_pts:
            self.local_image_pts.pop()
            self.local_world_pts.pop()
            self.marking_idx = max(0, self.marking_idx - 1)
            self._update_marking_info()
            self._update_points_ui()
            self._refresh_display()

    # ─────────────────────────────────────────────────────
    #  FINISH / CLOSE
    # ─────────────────────────────────────────────────────

    def _finish(self):
        """Calculate homography and close."""
        # If still marking, validate first
        if self.marking_mode:
            self._validate_marking()

        total = len(self.image_points)
        if total < 4:
            messagebox.showwarning("Points insuffisants",
                                   f"Il faut au minimum 4 points pour calculer l'homographie.\n"
                                   f"Vous avez {total} point(s).\n\n"
                                   f"Continuez à marquer des points ou annulez.",
                                   parent=self.win)
            return

        # Compute homography (RANSAC + least-squares refinement)
        estimator = HomographyEstimator(self.pitch)
        estimator.compute_from_correspondences(self.image_points, self.world_points)

        err = estimator.reprojection_error
        if err > 5.0:
            proceed = messagebox.askyesno(
                "Erreur de calibration élevée",
                f"L'erreur de reprojection est élevée: {err:.1f}m\n\n"
                f"Cela peut causer:\n"
                f"  • Des positions incorrect sur la minimap\n"
                f"  • Des vitesses artificiellement élevées\n\n"
                f"CAUSE PROBABLE: Points depuis plusieurs positions caméra.\n"
                f"RECOMMANDATION: Recommencer depuis UNE SEULE frame\n"
                f"où un maximum de repères sont visibles.\n\n"
                f"Continuer quand même avec cette calibration ?",
                parent=self.win
            )
            if not proceed:
                return

        # Afficher les infos multi-scène si plusieurs positions caméra détectées
        n_scenes = len(estimator.scene_homographies)
        if n_scenes > 1:
            scene_info = "\n".join(
                f"  Scène {i+1}: {sc['n_points']} pts, erreur {sc['reprojection_error']:.2f}m"
                for i, sc in enumerate(estimator.scene_homographies)
            )

            # Détecter si la scène principale (vue large) couvre les deux moitiés
            main_wb = estimator.scene_homographies[0].get('world_bounds', {})
            main_x_min = main_wb.get('x_min', 0)
            coverage_warning = ""
            if main_x_min > 25:
                # La scène principale ne couvre pas la moitié gauche du terrain
                coverage_warning = (
                    "\n\n⚠ ATTENTION — Couverture partielle (vue large) :\n"
                    "  La scène principale couvre seulement x ≥ "
                    f"{main_x_min + 5:.0f}m (moitié droite).\n"
                    "  Les joueurs côté gauche auront une position approchée.\n\n"
                    "  RECOMMANDATION : Pour une précision maximale :\n"
                    "  → Recalibrez en ajoutant des points depuis la VUE LARGE\n"
                    "    mais côté GAUCHE (coin gauche, surface gauche, etc.)."
                )

            messagebox.showinfo(
                "Multi-scène activé ✅",
                f"✅ {n_scenes} positions caméra détectées automatiquement!\n\n"
                f"{scene_info}\n\n"
                "Pendant l'analyse, le système sélectionnera\n"
                "automatiquement la bonne scène selon le zoom/angle de la caméra."
                f"{coverage_warning}",
                parent=self.win
            )

        self._cleanup()
        self.win.destroy()

        if self.on_complete:
            self.on_complete(estimator)

    def _on_close(self):
        if self.image_points:
            if not messagebox.askyesno("Annuler ?",
                                       f"Vous avez {len(self.image_points)} point(s) marqués.\n"
                                       "Voulez-vous vraiment annuler ?",
                                       parent=self.win):
                return
        self._cleanup()
        self.win.destroy()
        if self.on_complete:
            self.on_complete(None)

    def _cleanup(self):
        self.playing = False
        if self.after_id:
            try:
                self.win.after_cancel(self.after_id)
            except Exception:
                pass
        if self.cap:
            self.cap.release()


class FootballTacticalGUI:
    """Application Tkinter principale."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("⚽ Football Tactical Intelligence")
        self.root.geometry("1280x820")
        self.root.minsize(1100, 750)
        self.root.configure(bg=BG_DARK)

        # ─── État ───────────────────────────────────────
        self.video_path = tk.StringVar()
        self.homography_path = tk.StringVar()
        self.team_a_name = tk.StringVar(value="Équipe A")
        self.team_b_name = tk.StringVar(value="Équipe B")
        self.output_name = tk.StringVar()
        self.max_frames = tk.IntVar(value=0)
        self.skip_frames = tk.IntVar(value=0)
        self.show_voronoi = tk.BooleanVar(value=False)
        self.show_display = tk.BooleanVar(value=True)

        self.pipeline = None
        self.analysis_thread = None
        self.is_running = False
        self.output_dir = None

        # Lecteur vidéo
        self.player_cap = None
        self.player_playing = False
        self.player_after_id = None
        self.player_total_frames = 0
        self.player_fps = 25.0

        # ─── Construction de l'UI ───────────────────────
        self._build_ui()
        self._apply_styles()

    # ═══════════════════════════════════════════════════════════
    #  CONSTRUCTION DE L'UI
    # ═══════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Menu Bar ───────────────────────────────────
        self._build_menu_bar()

        # Notebook (onglets)
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Custom.TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("Custom.TNotebook.Tab",
                        background=BG_PANEL, foreground=TEXT_SECONDARY,
                        padding=[18, 8], font=("Segoe UI", 10, "bold"))
        style.map("Custom.TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", TEXT_PRIMARY)])

        self.notebook = ttk.Notebook(self.root, style="Custom.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        # ─── Onglet 1 : Configuration & Lancement ──────
        self.tab_config = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(self.tab_config, text="  ⚙  Configuration  ")
        self._build_config_tab()

        # ─── Onglet 2 : Lecteur Vidéo ──────────────────
        self.tab_player = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(self.tab_player, text="  ▶  Lecteur Vidéo  ")
        self._build_player_tab()

        # ─── Onglet 3 : Résultats & Dashboard ──────────
        self.tab_results = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(self.tab_results, text="  📊  Résultats  ")
        self._build_results_tab()

        # ─── Onglet 4 : Analyse Avancée ────────────────
        self.tab_advanced = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(self.tab_advanced, text="  🔬  Analyse Avancée  ")
        self._build_advanced_tab()

        # ─── Onglet 5 : Console / Logs ─────────────────
        self.tab_console = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(self.tab_console, text="  📋  Console  ")
        self._build_console_tab()

    # ─────────────────────────────────────────────────────────
    #  MENU BAR
    # ─────────────────────────────────────────────────────────

    def _build_menu_bar(self):
        menubar = tk.Menu(self.root, bg=BG_PANEL, fg=TEXT_PRIMARY,
                          activebackground=ACCENT, activeforeground=TEXT_PRIMARY,
                          relief="flat")

        # ── Fichier ─────────────────────────────────────
        file_menu = tk.Menu(menubar, tearoff=0, bg=BG_PANEL, fg=TEXT_PRIMARY,
                            activebackground=ACCENT, activeforeground=TEXT_PRIMARY)
        file_menu.add_command(label="📂  Ouvrir vidéo...", command=self._browse_video,
                              accelerator="Ctrl+O")
        file_menu.add_command(label="📄  Charger homographie...", command=self._browse_homography)
        file_menu.add_command(label="💾  Sauvegarder homographie...", command=self._save_homography_as)
        file_menu.add_separator()
        file_menu.add_command(label="📁  Ouvrir dossier résultats", command=self._open_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="❌  Quitter", command=self._on_quit, accelerator="Alt+F4")
        menubar.add_cascade(label="Fichier", menu=file_menu)

        # ── Outils ─────────────────────────────────────
        tools_menu = tk.Menu(menubar, tearoff=0, bg=BG_PANEL, fg=TEXT_PRIMARY,
                             activebackground=ACCENT, activeforeground=TEXT_PRIMARY)
        tools_menu.add_command(label="🎯  Calibration interactive", command=self._launch_calibration,
                               accelerator="Ctrl+K")
        tools_menu.add_separator()
        tools_menu.add_command(label="▶  Lancer l'analyse", command=self._launch_analysis,
                               accelerator="Ctrl+R")
        tools_menu.add_command(label="■  Arrêter l'analyse", command=self._stop_analysis)
        menubar.add_cascade(label="Outils", menu=tools_menu)

        # ── Affichage ──────────────────────────────────
        view_menu = tk.Menu(menubar, tearoff=0, bg=BG_PANEL, fg=TEXT_PRIMARY,
                            activebackground=ACCENT, activeforeground=TEXT_PRIMARY)
        view_menu.add_command(label="⚙  Configuration",
                              command=lambda: self.notebook.select(self.tab_config))
        view_menu.add_command(label="🎬  Lecteur Vidéo",
                              command=lambda: self.notebook.select(self.tab_player))
        view_menu.add_command(label="📊  Résultats",
                              command=lambda: self.notebook.select(self.tab_results))
        view_menu.add_command(label="🔬  Analyse Avancée",
                              command=lambda: self.notebook.select(self.tab_advanced))
        view_menu.add_command(label="📋  Console",
                              command=lambda: self.notebook.select(self.tab_console))
        view_menu.add_separator()
        view_menu.add_command(label="🔄  Actualiser résultats", command=self._refresh_all)
        menubar.add_cascade(label="Affichage", menu=view_menu)

        # ── Rapports ───────────────────────────────────
        report_menu = tk.Menu(menubar, tearoff=0, bg=BG_PANEL, fg=TEXT_PRIMARY,
                              activebackground=ACCENT, activeforeground=TEXT_PRIMARY)
        report_menu.add_command(label="🌐  Rapport HTML", command=self._open_html_report)
        report_menu.add_command(label="📄  Rapport JSON", command=self._open_json_report)
        report_menu.add_command(label="🔬  Analyse Avancée JSON", command=self._open_advanced_json)
        menubar.add_cascade(label="Rapports", menu=report_menu)

        # ── Aide ────────────────────────────────────────
        help_menu = tk.Menu(menubar, tearoff=0, bg=BG_PANEL, fg=TEXT_PRIMARY,
                            activebackground=ACCENT, activeforeground=TEXT_PRIMARY)
        help_menu.add_command(label="📖  Raccourcis clavier", command=self._show_shortcuts)
        help_menu.add_command(label="ℹ  À propos", command=self._show_about)
        menubar.add_cascade(label="Aide", menu=help_menu)

        self.root.config(menu=menubar)

        # Keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self._browse_video())
        self.root.bind("<Control-k>", lambda e: self._launch_calibration())
        self.root.bind("<Control-r>", lambda e: self._launch_analysis())

    def _refresh_all(self):
        self._refresh_results()
        self._refresh_advanced()

    def _save_homography_as(self):
        """Save current homography to a user-chosen location."""
        src = self.homography_path.get()
        if not src or not Path(src).exists():
            messagebox.showinfo("Non disponible",
                                "Pas d'homographie chargée.\n"
                                "Lancez d'abord une calibration.")
            return
        dest = filedialog.asksaveasfilename(
            title="Sauvegarder l'homographie",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="homography.json"
        )
        if dest:
            import shutil
            shutil.copy2(src, dest)
            self._log(f"[Homographie] Sauvegardé : {dest}")

    def _show_shortcuts(self):
        text = (
            "RACCOURCIS CLAVIER\n"
            "══════════════════════════════\n\n"
            "GÉNÉRAL:\n"
            "  Ctrl+O      Ouvrir une vidéo\n"
            "  Ctrl+K      Calibration interactive\n"
            "  Ctrl+R      Lancer l'analyse\n"
            "  Alt+F4      Quitter\n\n"
            "CALIBRATION (fenêtre dédiée):\n"
            "  Espace      Play / Pause\n"
            "  A / D       ±1 frame\n"
            "  , / .       ±1 seconde\n"
            "  [ / ]       ±5 secondes\n"
            "  C           Capturer la frame\n"
            "  Q           Terminer / Valider\n"
            "  S           Skip un point\n"
            "  U           Undo dernier point\n\n"
            "LECTEUR VIDÉO:\n"
            "  ▶/⏸         Play / Pause\n"
            "  Slider      Navigation directe\n"
        )
        messagebox.showinfo("Raccourcis Clavier", text)

    def _show_about(self):
        text = (
            "⚽ Football Tactical Intelligence\n\n"
            "Système d'analyse tactique automatisée\n"
            "basé sur la vision par ordinateur.\n\n"
            "• Détection YOLO + ByteTrack\n"
            "• Classification K-Means\n"
            "• Homographie manuelle\n"
            "• Analyse Voronoi, Convex Hull, Phases\n"
            "• Space Control, Offside, Pass Lanes\n\n"
            "Python 3.12 | OpenCV | Tkinter"
        )
        messagebox.showinfo("À propos", text)

    def _on_quit(self):
        if self.is_running:
            if not messagebox.askyesno("Confirmer", "Une analyse est en cours.\nQuitter quand même ?"):
                return
        self._player_stop()
        self.root.destroy()

    # ─────────────────────────────────────────────────────────
    #  ONGLET CONFIGURATION
    # ─────────────────────────────────────────────────────────

    def _build_config_tab(self):
        # Deux colonnes : gauche (config) + droite (aperçu vidéo)
        self.tab_config.columnconfigure(0, weight=1)
        self.tab_config.columnconfigure(1, weight=1)
        self.tab_config.rowconfigure(0, weight=1)

        left = tk.Frame(self.tab_config, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        right = tk.Frame(self.tab_config, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # ── LEFT: Configuration ─────────────────────────
        self._build_config_left(left)

        # ── RIGHT: Aperçu vidéo ─────────────────────────
        self._build_config_right(right)

    def _build_config_left(self, parent):
        parent.columnconfigure(0, weight=1)

        row = 0

        # ── Titre ───────────────────────────────────────
        title = tk.Label(parent, text="⚽ Football Tactical Intelligence",
                         font=("Segoe UI", 16, "bold"), fg=ACCENT, bg=BG_DARK)
        title.grid(row=row, column=0, sticky="w", pady=(5, 15), padx=10)
        row += 1

        # ── CARD: Vidéo ────────────────────────────────
        card_video = self._card(parent, "VIDÉO SOURCE")
        card_video.grid(row=row, column=0, sticky="ew", padx=10, pady=4)
        row += 1

        fr = tk.Frame(card_video, bg=BG_CARD)
        fr.pack(fill="x", padx=10, pady=5)
        fr.columnconfigure(1, weight=1)

        tk.Label(fr, text="Fichier :", fg=TEXT_SECONDARY, bg=BG_CARD,
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        e = tk.Entry(fr, textvariable=self.video_path, bg=ENTRY_BG,
                     fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                     font=("Consolas", 9), relief="flat", bd=0)
        e.grid(row=0, column=1, sticky="ew", padx=5)
        self._btn_small(fr, "Parcourir", self._browse_video).grid(row=0, column=2)

        # ── CARD: Équipes ──────────────────────────────
        card_teams = self._card(parent, "ÉQUIPES")
        card_teams.grid(row=row, column=0, sticky="ew", padx=10, pady=4)
        row += 1

        fr2 = tk.Frame(card_teams, bg=BG_CARD)
        fr2.pack(fill="x", padx=10, pady=5)
        fr2.columnconfigure(1, weight=1)
        fr2.columnconfigure(3, weight=1)

        tk.Label(fr2, text="Équipe A :", fg=TEXT_SECONDARY, bg=BG_CARD,
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        tk.Entry(fr2, textvariable=self.team_a_name, bg=ENTRY_BG,
                 fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                 font=("Segoe UI", 9), relief="flat", width=15).grid(
                     row=0, column=1, sticky="ew", padx=(5, 15))

        tk.Label(fr2, text="Équipe B :", fg=TEXT_SECONDARY, bg=BG_CARD,
                 font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w")
        tk.Entry(fr2, textvariable=self.team_b_name, bg=ENTRY_BG,
                 fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                 font=("Segoe UI", 9), relief="flat", width=15).grid(
                     row=0, column=3, sticky="ew", padx=5)

        # ── CARD: Homographie ──────────────────────────
        card_homo = self._card(parent, "HOMOGRAPHIE (CALIBRATION)")
        card_homo.grid(row=row, column=0, sticky="ew", padx=10, pady=4)
        row += 1

        fr3 = tk.Frame(card_homo, bg=BG_CARD)
        fr3.pack(fill="x", padx=10, pady=5)
        fr3.columnconfigure(1, weight=1)

        tk.Label(fr3, text="Fichier H :", fg=TEXT_SECONDARY, bg=BG_CARD,
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        tk.Entry(fr3, textvariable=self.homography_path, bg=ENTRY_BG,
                 fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                 font=("Consolas", 9), relief="flat").grid(
                     row=0, column=1, sticky="ew", padx=5)
        self._btn_small(fr3, "Parcourir", self._browse_homography).grid(row=0, column=2)

        fr3b = tk.Frame(card_homo, bg=BG_CARD)
        fr3b.pack(fill="x", padx=10, pady=(0, 8))
        self._btn_accent(fr3b, "🎯  Nouvelle Calibration Interactive",
                         self._launch_calibration).pack(side="left")
        self.lbl_homo_status = tk.Label(fr3b, text="", fg=TEXT_MUTED, bg=BG_CARD,
                                        font=("Segoe UI", 8))
        self.lbl_homo_status.pack(side="left", padx=15)

        # ── CARD: Options ──────────────────────────────
        card_opts = self._card(parent, "OPTIONS")
        card_opts.grid(row=row, column=0, sticky="ew", padx=10, pady=4)
        row += 1

        fr4 = tk.Frame(card_opts, bg=BG_CARD)
        fr4.pack(fill="x", padx=10, pady=5)

        tk.Label(fr4, text="Max frames (0=tout) :", fg=TEXT_SECONDARY,
                 bg=BG_CARD, font=("Segoe UI", 9)).pack(side="left")
        tk.Spinbox(fr4, from_=0, to=100000, textvariable=self.max_frames,
                   width=7, bg=ENTRY_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9),
                   relief="flat", buttonbackground=BG_PANEL).pack(side="left", padx=5)

        tk.Label(fr4, text="Skip N :", fg=TEXT_SECONDARY,
                 bg=BG_CARD, font=("Segoe UI", 9)).pack(side="left", padx=(15, 0))
        tk.Spinbox(fr4, from_=0, to=10, textvariable=self.skip_frames,
                   width=4, bg=ENTRY_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9),
                   relief="flat", buttonbackground=BG_PANEL).pack(side="left", padx=5)

        fr4b = tk.Frame(card_opts, bg=BG_CARD)
        fr4b.pack(fill="x", padx=10, pady=(0, 8))

        tk.Checkbutton(fr4b, text="Voronoi sur minimap", variable=self.show_voronoi,
                       fg=TEXT_SECONDARY, bg=BG_CARD, selectcolor=ENTRY_BG,
                       activebackground=BG_CARD, activeforeground=TEXT_PRIMARY,
                       font=("Segoe UI", 9)).pack(side="left")
        tk.Checkbutton(fr4b, text="Affichage live (cv2)", variable=self.show_display,
                       fg=TEXT_SECONDARY, bg=BG_CARD, selectcolor=ENTRY_BG,
                       activebackground=BG_CARD, activeforeground=TEXT_PRIMARY,
                       font=("Segoe UI", 9)).pack(side="left", padx=20)

        tk.Label(fr4b, text="Nom sortie :", fg=TEXT_SECONDARY, bg=BG_CARD,
                 font=("Segoe UI", 9)).pack(side="left", padx=(15, 0))
        tk.Entry(fr4b, textvariable=self.output_name, bg=ENTRY_BG,
                 fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                 font=("Segoe UI", 9), relief="flat", width=20).pack(side="left", padx=5)

        # ── BOUTONS PRINCIPAUX ─────────────────────────
        btn_frame = tk.Frame(parent, bg=BG_DARK)
        btn_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=15)
        row += 1

        self.btn_run = tk.Button(
            btn_frame, text="▶  LANCER L'ANALYSE", font=("Segoe UI", 13, "bold"),
            bg=ACCENT, fg=TEXT_PRIMARY, activebackground=ACCENT_HOVER,
            activeforeground=TEXT_PRIMARY, relief="flat", cursor="hand2",
            padx=30, pady=10, command=self._launch_analysis
        )
        self.btn_run.pack(side="left")

        self.btn_stop = tk.Button(
            btn_frame, text="■  STOP", font=("Segoe UI", 11, "bold"),
            bg="#444", fg=TEXT_PRIMARY, activebackground="#666",
            relief="flat", cursor="hand2", padx=20, pady=10,
            command=self._stop_analysis, state="disabled"
        )
        self.btn_stop.pack(side="left", padx=15)

        # ── Progress ───────────────────────────────────
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            parent, variable=self.progress_var, maximum=100,
            mode="determinate", length=400
        )
        self.progress_bar.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 5))
        row += 1

        self.lbl_progress = tk.Label(parent, text="Prêt", fg=TEXT_MUTED, bg=BG_DARK,
                                     font=("Segoe UI", 9))
        self.lbl_progress.grid(row=row, column=0, sticky="w", padx=12)

    def _build_config_right(self, parent):
        """Panneau droit : aperçu de la vidéo source."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        tk.Label(parent, text="APERÇU VIDÉO", font=("Segoe UI", 10, "bold"),
                 fg=TEXT_MUTED, bg=BG_DARK).grid(row=0, column=0, sticky="w",
                                                   padx=10, pady=(10, 5))

        self.preview_canvas = tk.Canvas(parent, bg="#0d0d1a", highlightthickness=0)
        self.preview_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Info vidéo
        self.lbl_video_info = tk.Label(parent, text="Aucune vidéo sélectionnée",
                                       fg=TEXT_MUTED, bg=BG_DARK,
                                       font=("Consolas", 9), justify="left")
        self.lbl_video_info.grid(row=2, column=0, sticky="w", padx=10, pady=5)

    # ─────────────────────────────────────────────────────────
    #  ONGLET LECTEUR VIDÉO
    # ─────────────────────────────────────────────────────────

    def _build_player_tab(self):
        self.tab_player.columnconfigure(0, weight=1)
        self.tab_player.rowconfigure(0, weight=1)

        # Canvas vidéo
        self.video_canvas = tk.Canvas(self.tab_player, bg="#0d0d1a",
                                      highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))

        # Contrôles
        ctrl = tk.Frame(self.tab_player, bg=BG_PANEL)
        ctrl.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        ctrl.columnconfigure(2, weight=1)

        self.btn_play = tk.Button(ctrl, text="▶", font=("Segoe UI", 12, "bold"),
                                  bg=ACCENT, fg=TEXT_PRIMARY, relief="flat",
                                  width=4, command=self._player_toggle_play)
        self.btn_play.grid(row=0, column=0, padx=5, pady=5)

        self.btn_player_stop = tk.Button(ctrl, text="■", font=("Segoe UI", 12),
                                          bg=BTN_SECONDARY, fg=TEXT_PRIMARY,
                                          relief="flat", width=4,
                                          command=self._player_stop)
        self.btn_player_stop.grid(row=0, column=1, padx=2, pady=5)

        self.player_scale = tk.Scale(
            ctrl, from_=0, to=100, orient="horizontal",
            bg=BG_PANEL, fg=TEXT_PRIMARY, troughcolor=ENTRY_BG,
            highlightthickness=0, sliderrelief="flat",
            command=self._player_seek
        )
        self.player_scale.grid(row=0, column=2, sticky="ew", padx=10, pady=5)

        self.lbl_player_time = tk.Label(ctrl, text="00:00 / 00:00",
                                        fg=TEXT_SECONDARY, bg=BG_PANEL,
                                        font=("Consolas", 10))
        self.lbl_player_time.grid(row=0, column=3, padx=10)

        # Bouton charger
        btns = tk.Frame(self.tab_player, bg=BG_DARK)
        btns.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        self._btn_accent(btns, "📂 Charger vidéo analysée",
                         self._player_load_result).pack(side="left")
        self._btn_small(btns, "Charger autre vidéo",
                        self._player_load_any).pack(side="left", padx=10)

        self.lbl_player_file = tk.Label(btns, text="", fg=TEXT_MUTED, bg=BG_DARK,
                                        font=("Segoe UI", 8))
        self.lbl_player_file.pack(side="left", padx=10)

    # ─────────────────────────────────────────────────────────
    #  ONGLET RÉSULTATS
    # ─────────────────────────────────────────────────────────

    def _build_results_tab(self):
        self.tab_results.columnconfigure(0, weight=1)
        self.tab_results.columnconfigure(1, weight=1)
        self.tab_results.rowconfigure(1, weight=1)

        # ── Boutons d'action ───────────────────────────
        top = tk.Frame(self.tab_results, bg=BG_DARK)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=8)

        self._btn_accent(top, "📂 Ouvrir dossier résultats",
                         self._open_output_folder).pack(side="left")
        self._btn_small(top, "🌐 Rapport HTML",
                        self._open_html_report).pack(side="left", padx=10)
        self._btn_small(top, "📄 Rapport JSON",
                        self._open_json_report).pack(side="left", padx=5)
        self._btn_small(top, "� Avancé JSON",
                        self._open_advanced_json).pack(side="left", padx=5)
        self._btn_small(top, "�🔄 Actualiser",
                        self._refresh_results).pack(side="right")

        # ── Gauche : Rapport texte ─────────────────────
        left = tk.Frame(self.tab_results, bg=BG_DARK)
        left.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        tk.Label(left, text="RAPPORT TACTIQUE", font=("Segoe UI", 10, "bold"),
                 fg=TEXT_MUTED, bg=BG_DARK).grid(row=0, column=0, sticky="w")

        self.txt_report = scrolledtext.ScrolledText(
            left, bg=ENTRY_BG, fg=TEXT_PRIMARY, font=("Consolas", 9),
            relief="flat", insertbackground=TEXT_PRIMARY, wrap="word"
        )
        self.txt_report.grid(row=1, column=0, sticky="nsew", pady=5)

        # ── Droite : Dashboard images ──────────────────
        right = tk.Frame(self.tab_results, bg=BG_DARK)
        right.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        tk.Label(right, text="DASHBOARD", font=("Segoe UI", 10, "bold"),
                 fg=TEXT_MUTED, bg=BG_DARK).grid(row=0, column=0, sticky="w")

        # Liste des images dashboard
        dash_frame = tk.Frame(right, bg=BG_PANEL)
        dash_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        dash_frame.columnconfigure(0, weight=1)
        dash_frame.rowconfigure(1, weight=1)

        self.dashboard_listbox = tk.Listbox(
            dash_frame, bg=ENTRY_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9),
            relief="flat", selectbackground=ACCENT, selectforeground=TEXT_PRIMARY,
            activestyle="none"
        )
        self.dashboard_listbox.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.dashboard_listbox.bind("<<ListboxSelect>>", self._on_dashboard_select)

        self.dashboard_canvas = tk.Canvas(dash_frame, bg="#0d0d1a",
                                          highlightthickness=0)
        self.dashboard_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    # ─────────────────────────────────────────────────────────
    #  ONGLET ANALYSE AVANCÉE
    # ─────────────────────────────────────────────────────────

    def _build_advanced_tab(self):
        self.tab_advanced.columnconfigure(0, weight=1)
        self.tab_advanced.columnconfigure(1, weight=1)
        self.tab_advanced.columnconfigure(2, weight=1)
        self.tab_advanced.rowconfigure(1, weight=1)

        # ── Barre du haut ──────────────────────────────
        top = tk.Frame(self.tab_advanced, bg=BG_DARK)
        top.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=8)

        tk.Label(top, text="🔬 ANALYSE TACTIQUE AVANCÉE",
                 font=("Segoe UI", 12, "bold"), fg=ACCENT, bg=BG_DARK).pack(side="left")
        self._btn_small(top, "🔄 Actualiser", self._refresh_advanced).pack(side="right")

        # ── Colonne 1 : Space Control ──────────────────
        col1 = tk.Frame(self.tab_advanced, bg=BG_DARK)
        col1.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        col1.columnconfigure(0, weight=1)
        col1.rowconfigure(1, weight=1)

        card_sc = self._card(col1, "SPACE CONTROL")
        card_sc.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.adv_sc_text = tk.Text(card_sc, bg=BG_CARD, fg=TEXT_PRIMARY,
                                    font=("Consolas", 10), relief="flat",
                                    height=6, wrap="word", bd=0)
        self.adv_sc_text.pack(fill="x", padx=10, pady=(5, 10))
        self.adv_sc_text.insert("1.0", "En attente d'analyse...")
        self.adv_sc_text.config(state="disabled")

        # Image Space Control timeline
        self.adv_sc_canvas = tk.Canvas(col1, bg="#0d0d1a", highlightthickness=0)
        self.adv_sc_canvas.grid(row=1, column=0, sticky="nsew", pady=5)

        # ── Colonne 2 : Offside / Defensive Line ──────
        col2 = tk.Frame(self.tab_advanced, bg=BG_DARK)
        col2.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        col2.columnconfigure(0, weight=1)
        col2.rowconfigure(1, weight=1)

        card_off = self._card(col2, "LIGNE DE HORS-JEU")
        card_off.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.adv_off_text = tk.Text(card_off, bg=BG_CARD, fg=TEXT_PRIMARY,
                                     font=("Consolas", 10), relief="flat",
                                     height=6, wrap="word", bd=0)
        self.adv_off_text.pack(fill="x", padx=10, pady=(5, 10))
        self.adv_off_text.insert("1.0", "En attente d'analyse...")
        self.adv_off_text.config(state="disabled")

        # Placeholder canvas for future offside chart
        self.adv_off_canvas = tk.Canvas(col2, bg="#0d0d1a", highlightthickness=0)
        self.adv_off_canvas.grid(row=1, column=0, sticky="nsew", pady=5)

        # ── Colonne 3 : Pass Lanes ────────────────────
        col3 = tk.Frame(self.tab_advanced, bg=BG_DARK)
        col3.grid(row=1, column=2, sticky="nsew", padx=(5, 10), pady=5)
        col3.columnconfigure(0, weight=1)
        col3.rowconfigure(1, weight=1)

        card_pl = self._card(col3, "LIGNES DE PASSE")
        card_pl.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.adv_pl_text = tk.Text(card_pl, bg=BG_CARD, fg=TEXT_PRIMARY,
                                    font=("Consolas", 10), relief="flat",
                                    height=6, wrap="word", bd=0)
        self.adv_pl_text.pack(fill="x", padx=10, pady=(5, 10))
        self.adv_pl_text.insert("1.0", "En attente d'analyse...")
        self.adv_pl_text.config(state="disabled")

        # Image Pass Availability timeline
        self.adv_pl_canvas = tk.Canvas(col3, bg="#0d0d1a", highlightthickness=0)
        self.adv_pl_canvas.grid(row=1, column=0, sticky="nsew", pady=5)

    def _refresh_advanced(self):
        """Charge et affiche les données d'analyse avancée."""
        od = self.output_dir
        if od is None:
            subdirs = sorted(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.exists() else []
            subdirs = [d for d in subdirs if d.is_dir()]
            if subdirs:
                od = subdirs[-1]
            else:
                return
        od = Path(od)

        # Charger le JSON avancé
        adv_path = od / "advanced_tactical.json"
        if not adv_path.exists():
            return

        try:
            with open(adv_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        # ── Space Control ──────────────────────────────
        sc = data.get("space_control", {})
        sc_lines = [
            f"  Équipe A :  {sc.get('team_a_avg_pct', '?')}%",
            f"  Équipe B :  {sc.get('team_b_avg_pct', '?')}%",
            f"",
            f"  Penalty A :  {sc.get('team_a_penalty_area_control_pct', '?')}%",
            f"  Penalty B :  {sc.get('team_b_penalty_area_control_pct', '?')}%",
        ]
        self._set_text(self.adv_sc_text, "\n".join(sc_lines))

        # ── Offside / Defensive Line ───────────────────
        dl = data.get("defensive_line", {})
        off_lines = [
            f"  Stabilité A :  {dl.get('team_a_stability_m', '?')} m",
            f"  Stabilité B :  {dl.get('team_b_stability_m', '?')} m",
            f"",
            f"  Situations offside A :  {dl.get('team_a_offside_situations', 0)}",
            f"  Situations offside B :  {dl.get('team_b_offside_situations', 0)}",
        ]
        self._set_text(self.adv_off_text, "\n".join(off_lines))

        # ── Pass Lanes ─────────────────────────────────
        pl = data.get("pass_lanes", {})
        pl_lines = [
            f"  Pass Availability (global) :  {pl.get('avg_pass_availability_pct', '?')}%",
            f"  Pass Availability Éq. A    :  {pl.get('team_a_avg_pass_availability_pct', '?')}%",
            f"  Pass Availability Éq. B    :  {pl.get('team_b_avg_pass_availability_pct', '?')}%",
        ]
        self._set_text(self.adv_pl_text, "\n".join(pl_lines))

        # ── Charger les images dashboard associées ─────
        dash_dir = od / "dashboard"
        sc_img = dash_dir / "space_control_timeline.png"
        if sc_img.exists():
            img = cv2.imread(str(sc_img))
            if img is not None:
                self._display_on_canvas(img, self.adv_sc_canvas)

        pa_img = dash_dir / "pass_availability_timeline.png"
        if pa_img.exists():
            img = cv2.imread(str(pa_img))
            if img is not None:
                self._display_on_canvas(img, self.adv_pl_canvas)

    def _set_text(self, widget: tk.Text, text: str):
        """Met à jour le contenu d'un widget Text en lecture seule."""
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def _open_advanced_json(self):
        """Affiche le rapport advanced_tactical.json dans la console."""
        if self.output_dir:
            jp = Path(self.output_dir) / "advanced_tactical.json"
            if jp.exists():
                try:
                    with open(jp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.notebook.select(self.tab_console)
                    self._log("\n" + "=" * 35 + " ANALYSE AVANCÉE " + "=" * 35)
                    self._log(json.dumps(data, indent=2, ensure_ascii=False))
                except Exception as e:
                    self._log(f"Erreur lecture JSON avancé: {e}")
                return
        messagebox.showinfo("Non disponible",
                            "Rapport d'analyse avancée non trouvé.\n"
                            "Lancez d'abord une analyse.")

    # ─────────────────────────────────────────────────────────
    #  ONGLET CONSOLE
    # ─────────────────────────────────────────────────────────

    def _build_console_tab(self):
        self.tab_console.columnconfigure(0, weight=1)
        self.tab_console.rowconfigure(0, weight=1)

        self.console_text = scrolledtext.ScrolledText(
            self.tab_console, bg="#0a0a14", fg="#00ff88",
            font=("Consolas", 9), relief="flat",
            insertbackground="#00ff88", wrap="word"
        )
        self.console_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        btn_fr = tk.Frame(self.tab_console, bg=BG_DARK)
        btn_fr.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self._btn_small(btn_fr, "Effacer", self._clear_console).pack(side="right")

    # ═══════════════════════════════════════════════════════════
    #  HELPERS UI
    # ═══════════════════════════════════════════════════════════

    def _card(self, parent, title: str) -> tk.Frame:
        """Crée une carte UI avec titre."""
        outer = tk.Frame(parent, bg=BG_CARD, bd=0, relief="flat")
        tk.Label(outer, text=title, font=("Segoe UI", 9, "bold"),
                 fg=ACCENT, bg=BG_CARD).pack(anchor="w", padx=10, pady=(8, 0))
        return outer

    def _btn_small(self, parent, text, command) -> tk.Button:
        btn = tk.Button(parent, text=text, command=command,
                        bg=BTN_SECONDARY, fg=TEXT_PRIMARY,
                        activebackground="#1a4a7a", activeforeground=TEXT_PRIMARY,
                        font=("Segoe UI", 9), relief="flat", cursor="hand2",
                        padx=10, pady=3)
        return btn

    def _btn_accent(self, parent, text, command) -> tk.Button:
        btn = tk.Button(parent, text=text, command=command,
                        bg=ACCENT, fg=TEXT_PRIMARY,
                        activebackground=ACCENT_HOVER,
                        activeforeground=TEXT_PRIMARY,
                        font=("Segoe UI", 10, "bold"), relief="flat",
                        cursor="hand2", padx=14, pady=5)
        return btn

    def _apply_styles(self):
        """Configure le style ttk pour la barre de progression."""
        style = ttk.Style()
        style.configure("TProgressbar",
                        troughcolor=ENTRY_BG,
                        background=ACCENT,
                        thickness=12)

    def _log(self, msg: str):
        """Écrit un message dans la console."""
        self.console_text.insert("end", msg + "\n")
        self.console_text.see("end")

    # ═══════════════════════════════════════════════════════════
    #  ACTIONS — CONFIGURATION
    # ═══════════════════════════════════════════════════════════

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Sélectionner la vidéo du match",
            initialdir=str(DATA_DIR / "sample_videos"),
            filetypes=[
                ("Vidéos", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("Tous", "*.*"),
            ]
        )
        if path:
            self.video_path.set(path)
            self._load_video_preview(path)

    def _browse_homography(self):
        path = filedialog.askopenfilename(
            title="Sélectionner le fichier homography.json",
            initialdir=str(OUTPUT_DIR),
            filetypes=[("JSON", "*.json"), ("Tous", "*.*")]
        )
        if path:
            self.homography_path.set(path)
            self._update_homo_status(path)

    def _load_video_preview(self, path: str):
        """Charge un aperçu de la vidéo et affiche les métadonnées."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.lbl_video_info.config(text="❌ Impossible d'ouvrir la vidéo")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps

        info = (f"📁 {Path(path).name}\n"
                f"📐 {w}×{h}  |  🎞 {fps:.0f} FPS  |  "
                f"⏱ {duration:.1f}s ({total} frames)")
        self.lbl_video_info.config(text=info)

        # Lire la première frame comme aperçu
        ret, frame = cap.read()
        cap.release()
        if ret:
            self._display_on_canvas(frame, self.preview_canvas)

    def _display_on_canvas(self, frame, canvas: tk.Canvas):
        """Affiche une frame OpenCV sur un canvas Tkinter."""
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 600, 400

        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        canvas._photo = photo  # garder la référence

    def _update_homo_status(self, path: str):
        """Affiche l'erreur de reprojection du fichier homography."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            err = data.get("reprojection_error", "?")
            n_pts = len(data.get("src_points", []))
            self.lbl_homo_status.config(
                text=f"✅ {n_pts} points | Erreur: {err:.2f}m"
                     if isinstance(err, (int, float)) else f"✅ {n_pts} points",
                fg=SUCCESS
            )
        except Exception:
            self.lbl_homo_status.config(text="⚠ Fichier invalide", fg=WARNING)

    # ═══════════════════════════════════════════════════════════
    #  ACTIONS — CALIBRATION
    # ═══════════════════════════════════════════════════════════

    def _launch_calibration(self):
        """Lance la calibration interactive dans une fenêtre Tkinter dédiée."""
        vpath = self.video_path.get()
        if not vpath or not Path(vpath).exists():
            messagebox.showwarning("Vidéo manquante",
                                   "Sélectionnez d'abord une vidéo.")
            return

        self._log("[Calibration] Ouverture de la fenêtre de calibration...")

        def on_calibration_done(estimator):
            if estimator is None:
                self._log("[Calibration] ❌ Annulée par l'utilisateur")
                return

            # Save homography
            oname = self.output_name.get() or (Path(vpath).stem + "_analyzed")
            out_dir = OUTPUT_DIR / oname
            out_dir.mkdir(parents=True, exist_ok=True)
            h_path = str(out_dir / "homography.json")
            estimator.save(h_path)

            self.homography_path.set(h_path)
            self._update_homo_status(h_path)
            self._log(f"[Calibration] ✅ Terminée! Sauvegardé: {h_path}")

        CalibrationWindow(self.root, vpath, on_complete=on_calibration_done)

    # ═══════════════════════════════════════════════════════════
    #  ACTIONS — ANALYSE
    # ═══════════════════════════════════════════════════════════

    def _launch_analysis(self):
        """Lance le pipeline d'analyse dans un thread séparé."""
        vpath = self.video_path.get()
        if not vpath or not Path(vpath).exists():
            messagebox.showwarning("Vidéo manquante",
                                   "Sélectionnez d'abord une vidéo source.")
            return

        if self.is_running:
            messagebox.showinfo("En cours", "Une analyse est déjà en cours.")
            return

        hpath = self.homography_path.get() or None
        oname = self.output_name.get() or None

        self.is_running = True
        self.btn_run.config(state="disabled", bg="#555")
        self.btn_stop.config(state="normal", bg=ACCENT)
        self.progress_var.set(0)
        self.lbl_progress.config(text="Initialisation...", fg=WARNING)
        self.notebook.select(self.tab_console)

        self._log("\n" + "=" * 60)
        self._log("  LANCEMENT DE L'ANALYSE")
        self._log("=" * 60)

        def run_pipeline():
            original_stdout = sys.stdout
            try:
                # Rediriger stdout vers la console
                sys.stdout = ConsoleRedirector(self.root, self.console_text)

                pipeline = FootballAnalysisPipeline(
                    video_path=vpath,
                    output_name=oname,
                    team_a_name=self.team_a_name.get(),
                    team_b_name=self.team_b_name.get(),
                    homography_path=hpath,
                    max_frames=self.max_frames.get(),
                    skip_frames=self.skip_frames.get(),
                    display=self.show_display.get(),
                    show_voronoi=self.show_voronoi.get(),
                )
                self.pipeline = pipeline
                self.output_dir = pipeline.output_dir

                total = pipeline.max_frames if pipeline.max_frames > 0 else pipeline.total_frames

                def on_progress(frame_idx, eff_total):
                    pct = min(100, ((frame_idx + 1) / max(eff_total, 1)) * 100)
                    self.root.after(0, lambda p=pct: self.progress_var.set(p))
                    self.root.after(0, lambda c=frame_idx + 1, t=eff_total: self.lbl_progress.config(
                        text=f"Frame {c}/{t} ({c/max(t,1)*100:.0f}%)",
                        fg=WARNING
                    ))

                # Lancer avec le callback de progression
                result_dir = pipeline.run(progress_callback=on_progress)

                sys.stdout = original_stdout

                self.root.after(0, lambda rd=result_dir: self._on_analysis_complete(rd))

            except Exception as e:
                err_msg = str(e)
                sys.stdout = original_stdout
                self.root.after(0, lambda em=err_msg: self._on_analysis_error(Exception(em)))
            finally:
                sys.stdout = original_stdout
                # S'assurer que l'UI est toujours restaur\u00e9e
                self.root.after(100, self._ensure_analysis_ui_reset)

        self.analysis_thread = threading.Thread(target=run_pipeline, daemon=True)
        self.analysis_thread.start()

    def _stop_analysis(self):
        """Tente d'arrêter l'analyse en fermant la capture vidéo."""
        if self.pipeline and self.is_running:
            try:
                self.pipeline.cap.release()
                self._log("[Pipeline] ⛔ Arrêt demandé...")
            except Exception:
                pass

    def _ensure_analysis_ui_reset(self):
        """Filet de sécurité : réinitialise les boutons si l'analyse n'est plus active."""
        if not self.is_running:
            return
        # Si on arrive ici, _on_analysis_complete/_error n'a pas été appelé
        self.is_running = False
        try:
            self.btn_run.config(state="normal", bg=ACCENT)
            self.btn_stop.config(state="disabled", bg="#444")
            self.lbl_progress.config(text="⚠ Analyse interrompue", fg=WARNING)
        except tk.TclError:
            pass

    def _on_analysis_complete(self, result_dir):
        """Callback quand l'analyse est terminée."""
        self.is_running = False
        self.btn_run.config(state="normal", bg=ACCENT)
        self.btn_stop.config(state="disabled", bg="#444")
        self.progress_var.set(100)
        self.lbl_progress.config(text="✅ Analyse terminée!", fg=SUCCESS)

        self._log(f"\n✅ ANALYSE TERMINÉE — Résultats: {result_dir}")

        self.output_dir = result_dir
        self._refresh_results()
        self._refresh_advanced()

        # Charger automatiquement la vidéo résultat dans le lecteur
        result_video = list(Path(result_dir).glob("*.mp4"))
        if result_video:
            self._player_open(str(result_video[0]))

        self.notebook.select(self.tab_results)

    def _on_analysis_error(self, error):
        """Callback en cas d'erreur."""
        self.is_running = False
        self.btn_run.config(state="normal", bg=ACCENT)
        self.btn_stop.config(state="disabled", bg="#444")
        self.progress_var.set(0)
        self.lbl_progress.config(text=f"❌ Erreur: {error}", fg=ACCENT)
        self._log(f"\n❌ ERREUR: {error}")
        messagebox.showerror("Erreur d'analyse", str(error))

    # ═══════════════════════════════════════════════════════════
    #  LECTEUR VIDÉO
    # ═══════════════════════════════════════════════════════════

    def _player_load_result(self):
        """Charge la vidéo résultat du dernier run."""
        if self.output_dir and Path(self.output_dir).exists():
            videos = list(Path(self.output_dir).glob("*.mp4"))
            if videos:
                self._player_open(str(videos[0]))
                return
        messagebox.showinfo("Pas de résultat",
                            "Lancez d'abord une analyse pour avoir une vidéo résultat.")

    def _player_load_any(self):
        """Charge n'importe quelle vidéo."""
        path = filedialog.askopenfilename(
            title="Charger une vidéo",
            initialdir=str(OUTPUT_DIR),
            filetypes=[("Vidéos", "*.mp4 *.avi *.mkv"), ("Tous", "*.*")]
        )
        if path:
            self._player_open(path)

    def _player_open(self, path: str):
        """Ouvre une vidéo dans le lecteur intégré."""
        self._player_stop()

        self.player_cap = cv2.VideoCapture(path)
        if not self.player_cap.isOpened():
            messagebox.showerror("Erreur", f"Impossible d'ouvrir: {path}")
            return

        self.player_total_frames = int(self.player_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.player_fps = self.player_cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.player_scale.config(to=max(self.player_total_frames - 1, 1))
        self.player_scale.set(0)
        self.lbl_player_file.config(text=Path(path).name)

        # Afficher la première frame
        ret, frame = self.player_cap.read()
        if ret:
            self._display_on_canvas(frame, self.video_canvas)
            self.player_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._player_update_time(0)
        self.notebook.select(self.tab_player)

    def _player_toggle_play(self):
        if self.player_cap is None:
            return
        self.player_playing = not self.player_playing
        try:
            self.btn_play.config(text="⏸" if self.player_playing else "▶")
        except tk.TclError:
            return
        if self.player_playing:
            self._player_loop()

    def _player_loop(self):
        if not self.player_playing or self.player_cap is None:
            return

        ret, frame = self.player_cap.read()
        if not ret:
            self.player_playing = False
            try:
                self.btn_play.config(text="▶")
            except tk.TclError:
                pass
            return

        self._display_on_canvas(frame, self.video_canvas)
        pos = int(self.player_cap.get(cv2.CAP_PROP_POS_FRAMES))
        try:
            self.player_scale.set(pos)
        except tk.TclError:
            return
        self._player_update_time(pos)

        delay = max(1, int(1000 / self.player_fps))
        self.player_after_id = self.root.after(delay, self._player_loop)

    def _player_seek(self, val):
        if self.player_cap is None:
            return
        if self.player_playing:
            return  # Ne pas seek pendant la lecture

        frame_no = int(float(val))
        self.player_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.player_cap.read()
        if ret:
            self._display_on_canvas(frame, self.video_canvas)
        self._player_update_time(frame_no)

    def _player_stop(self):
        self.player_playing = False
        try:
            self.btn_play.config(text="▶")
        except tk.TclError:
            pass
        if self.player_after_id:
            try:
                self.root.after_cancel(self.player_after_id)
            except (tk.TclError, ValueError):
                pass
            self.player_after_id = None
        if self.player_cap:
            self.player_cap.release()
            self.player_cap = None

    def _player_update_time(self, frame_no: int):
        current = frame_no / max(self.player_fps, 1)
        total = self.player_total_frames / max(self.player_fps, 1)
        cm, cs = divmod(int(current), 60)
        tm, ts = divmod(int(total), 60)
        self.lbl_player_time.config(text=f"{cm:02d}:{cs:02d} / {tm:02d}:{ts:02d}")

    # ═══════════════════════════════════════════════════════════
    #  RÉSULTATS & DASHBOARD
    # ═══════════════════════════════════════════════════════════

    def _refresh_results(self):
        """Actualise l'onglet résultats."""
        od = self.output_dir
        if od is None:
            # Essayer le dernier dossier output
            subdirs = sorted(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.exists() else []
            subdirs = [d for d in subdirs if d.is_dir()]
            if subdirs:
                od = subdirs[-1]
            else:
                return

        self.output_dir = Path(od)

        # Charger le rapport texte
        txt_path = self.output_dir / "rapport_tactique.txt"
        if txt_path.exists():
            self.txt_report.delete("1.0", "end")
            self.txt_report.insert("1.0", txt_path.read_text(encoding="utf-8"))

        # Charger les images dashboard
        dash_dir = self.output_dir / "dashboard"
        self.dashboard_listbox.delete(0, "end")
        self._dashboard_images = {}
        if dash_dir.exists():
            for img_path in sorted(dash_dir.glob("*.png")):
                name = img_path.stem.replace("_", " ").title()
                self.dashboard_listbox.insert("end", name)
                self._dashboard_images[name] = str(img_path)

    def _on_dashboard_select(self, event):
        sel = self.dashboard_listbox.curselection()
        if not sel:
            return
        name = self.dashboard_listbox.get(sel[0])
        img_path = self._dashboard_images.get(name)
        if img_path and Path(img_path).exists():
            img = cv2.imread(img_path)
            if img is not None:
                self._display_on_canvas(img, self.dashboard_canvas)

    def _open_output_folder(self):
        if self.output_dir and Path(self.output_dir).exists():
            os.startfile(str(self.output_dir))
        else:
            messagebox.showinfo("Pas de résultats",
                                "Aucun dossier de résultats trouvé.")

    def _open_html_report(self):
        if self.output_dir:
            html = Path(self.output_dir) / "rapport_tactique.html"
            if html.exists():
                webbrowser.open(str(html))
                return
        messagebox.showinfo("Non disponible", "Rapport HTML non trouvé.")

    def _open_json_report(self):
        if self.output_dir:
            jp = Path(self.output_dir) / "rapport_tactique.json"
            if jp.exists():
                try:
                    with open(jp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Afficher dans la console
                    self.notebook.select(self.tab_console)
                    self._log("\n" + "=" * 40 + " RAPPORT JSON " + "=" * 40)
                    self._log(json.dumps(data, indent=2, ensure_ascii=False))
                except Exception as e:
                    self._log(f"Erreur lecture JSON: {e}")
                return
        messagebox.showinfo("Non disponible", "Rapport JSON non trouvé.")

    def _clear_console(self):
        self.console_text.delete("1.0", "end")

    # ═══════════════════════════════════════════════════════════
    #  LANCEMENT
    # ═══════════════════════════════════════════════════════════

    def run(self):
        """Lance la boucle principale Tkinter."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        # Charger les résultats existants au démarrage
        self.root.after(500, self._refresh_results)
        self.root.after(600, self._refresh_advanced)
        self.root.mainloop()

        # Cleanup
        self._player_stop()


class ConsoleRedirector:
    """Redirige sys.stdout vers un widget ScrolledText Tkinter."""

    def __init__(self, root: tk.Tk, widget: scrolledtext.ScrolledText):
        self.root = root
        self.widget = widget

    def write(self, text: str):
        if text.strip():
            self.root.after(0, self._append, text)

    def _append(self, text: str):
        self.widget.insert("end", text + "\n")
        self.widget.see("end")

    def flush(self):
        pass


# ═══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = FootballTacticalGUI()
    app.run()
