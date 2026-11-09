"""Desktop interface for the project's two saved models.

Run with:
    .venv/bin/python app.py

MoViNet A0 classifies videos. YOLOv8n detects labeled
littering regions in sampled frames. Opening the app never starts training.
"""
from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from common import CHECKPOINT_PATHS, PROJECT_ROOT, analyze_video


CLASSIFIERS = {
    "movinet_a0": "MoViNet A0",
}
YOLO_SLUG = "yolov8n"
YOLO_MODEL = PROJECT_ROOT / "src" / "yolo" / "model.pt"
BOTH_MODELS = "Both models"
MODEL_CHOICES = {
    BOTH_MODELS: (*CLASSIFIERS, YOLO_SLUG),
    **{label: (slug,) for slug, label in CLASSIFIERS.items()},
    "YOLOv8n": (YOLO_SLUG,),
}
EVALUATION_PATH = PROJECT_ROOT / "results" / "comparison" / "model_comparison.json"
YOLO_EVALUATION_PATH = PROJECT_ROOT / "results" / "yolo" / "yolov8n_run.json"


def run_yolo(video: Path) -> dict:
    """Load trained YOLO only on request; never download weights or train."""
    from src.yolo.main import detect_video

    return detect_video(video, model_path=YOLO_MODEL, frames=12)


class App(tk.Tk):
    """Analyse videos and show each model's task-specific results."""

    def __init__(self):
        super().__init__()
        self.title("Illegal Waste Dumping Detection")
        self.geometry("1100x730")
        self.minsize(970, 620)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.video: Path | None = None
        self.busy = False
        self.status = tk.StringVar(value="Select an MP4 video to begin.")
        self.filename = tk.StringVar(value="No video selected")
        self.model_choice = tk.StringVar(value=BOTH_MODELS)
        self.yolo_details = tk.StringVar(value="YOLOv8n is ready. Select a video to analyze sampled frames.")

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self.build_video_tab()
        self.build_evaluation_tab()
        self.after(100, self.poll)

    def build_video_tab(self):
        panel = ttk.Frame(self.tabs, padding=24)
        self.tabs.add(panel, text="Video Analysis")
        ttk.Label(panel, text="Video Analysis", font=("Helvetica", 25, "bold")).pack(anchor="w")
        ttk.Label(
            panel,
            text=(
                "Analyze a surveillance video with MoViNet A0 and YOLOv8n. "
                "Choose both models or run one individually."
            ),
            wraplength=1000,
        ).pack(anchor="w", pady=(6, 16))

        controls = ttk.Frame(panel)
        controls.pack(fill="x")
        self.select = ttk.Button(controls, text="Select Video", command=self.choose_video)
        self.select.pack(side="left")
        self.model_select = ttk.Combobox(
            controls, textvariable=self.model_choice, values=tuple(MODEL_CHOICES),
            state="readonly", width=25,
        )
        self.model_select.pack(side="left", padx=(14, 18))
        self.run_button = ttk.Button(controls, text="Analyze Video", command=self.start, state="disabled")
        self.run_button.pack(side="left")

        ttk.Label(panel, textvariable=self.filename, wraplength=1000).pack(anchor="w", pady=(14, 6))
        self.progress = ttk.Progressbar(panel, mode="indeterminate")
        self.progress.pack(fill="x")
        ttk.Label(panel, textvariable=self.status, wraplength=1000).pack(anchor="w", pady=(8, 12))

        action_box = ttk.LabelFrame(panel, text="MoViNet A0 — Video Classification", padding=10)
        action_box.pack(fill="both", expand=True)
        columns = ("model", "result", "score", "meaning")
        self.table = ttk.Treeview(action_box, columns=columns, show="headings", height=1)
        for name, title, width in zip(
            columns,
            ("Model", "Prediction", "Score", "Score meaning"),
            (180, 165, 105, 480),
        ):
            self.table.heading(name, text=title)
            self.table.column(name, width=width, minwidth=70)
        self.table.pack(fill="both", expand=True)
        ttk.Label(
            action_box,
            text="MoViNet uses an uncalibrated sigmoid score. Scores of 0.5 or higher predict Dumping.",
            wraplength=980,
        ).pack(anchor="w", pady=(8, 0))

        detection_box = ttk.LabelFrame(panel, text="YOLOv8n — Littering Region Detection", padding=10)
        detection_box.pack(fill="x", pady=(12, 0))
        detection_columns = ("model", "result", "confidence", "regions", "frames")
        self.yolo_table = ttk.Treeview(detection_box, columns=detection_columns, show="headings", height=1)
        for name, title, width in zip(
            detection_columns,
            ("Model", "Detection result", "Strongest box score", "Regions", "Sampled frames"),
            (145, 300, 190, 100, 140),
        ):
            self.yolo_table.heading(name, text=title)
            self.yolo_table.column(name, width=width, minwidth=70)
        self.yolo_table.pack(fill="x")
        ttk.Label(detection_box, textvariable=self.yolo_details, wraplength=980).pack(anchor="w", pady=(8, 0))
        ttk.Label(
            panel,
            text=(
                "YOLO detects the labeled 'littering' region, not a specific trash type or a proven dumping event. "
                "Region counts across frames are not unique objects. "
                "AI predictions should be reviewed by an authorized human."
            ),
            wraplength=1000,
        ).pack(anchor="w", pady=(14, 0))

    def build_evaluation_tab(self):
        outer = ttk.Frame(self.tabs)
        self.tabs.add(outer, text="Evaluation Results")
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        panel = ttk.Frame(canvas, padding=22)
        content = canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(content, width=event.width))
        ttk.Label(panel, text="Evaluation Results", font=("Helvetica", 24, "bold")).pack(anchor="w")

        mivia_box = ttk.LabelFrame(panel, text="MoViNet A0 — MIVIA video classification", padding=10)
        mivia_box.pack(fill="x", pady=(12, 10))
        ttk.Label(
            mivia_box,
            text=(
                "100 official MIVIA test videos: 50 dumping / 50 no dumping. "
                "Training: 320 videos • Validation: 80 videos • 24 frames per video."
            ),
            wraplength=980,
        ).pack(anchor="w")
        self.eval_status = tk.StringVar()
        ttk.Label(mivia_box, textvariable=self.eval_status, wraplength=980).pack(anchor="w", pady=(5, 8))
        columns = ("model", "accuracy", "precision", "recall", "f1", "train", "infer", "size")
        self.eval_table = ttk.Treeview(mivia_box, columns=columns, show="headings", height=1, selectmode="browse")
        for col, title, width in zip(
            columns,
            ("Model", "Accuracy", "Precision", "Recall", "F1", "Train (s)", "Infer (ms)", "Size (MiB)"),
            (175, 90, 90, 90, 90, 95, 95, 95),
        ):
            self.eval_table.heading(col, text=title)
            self.eval_table.column(col, width=width, minwidth=width, anchor="w" if col == "model" else "center")
        self.eval_table.pack(fill="x")
        self.eval_table.bind("<<TreeviewSelect>>", self.show_confusion)

        yolo_box = ttk.LabelFrame(panel, text="YOLOv8n — Roboflow Littering bounding-box evaluation", padding=10)
        yolo_box.pack(fill="x", pady=(0, 10))
        self.yolo_eval_status = tk.StringVar()
        ttk.Label(yolo_box, textvariable=self.yolo_eval_status, wraplength=980).pack(anchor="w", pady=(0, 6))
        detection_columns = ("model", "precision", "recall", "map50", "map50_95", "train", "validation", "size")
        self.yolo_eval_table = ttk.Treeview(yolo_box, columns=detection_columns, show="headings", height=1)
        for col, title, width in zip(
            detection_columns,
            ("Model", "Precision", "Recall", "mAP@50", "mAP@50:95", "Train (s)", "Val. (s)", "Size (MiB)"),
            (175, 90, 90, 90, 100, 95, 95, 95),
        ):
            self.yolo_eval_table.heading(col, text=title)
            self.yolo_eval_table.column(col, width=width, minwidth=width, anchor="w" if col == "model" else "center")
        self.yolo_eval_table.pack(fill="x")

        self.matrix_title = tk.StringVar(value="Confusion matrix")
        matrix = ttk.LabelFrame(panel, text="Confusion matrix — MIVIA video counts", padding=10)
        matrix.pack(fill="x", pady=(0, 10))
        ttk.Label(matrix, textvariable=self.matrix_title).grid(row=0, column=0, sticky="w")
        for col, text in enumerate(("", "Predicted: No Dumping", "Predicted: Dumping")):
            ttk.Label(matrix, text=text).grid(row=1, column=col, padx=12, pady=4)
        self.matrix_cells: list[tk.StringVar] = []
        for row, label in enumerate(("Actual: No Dumping", "Actual: Dumping"), start=2):
            ttk.Label(matrix, text=label).grid(row=row, column=0, sticky="w", padx=12, pady=5)
            for col in (1, 2):
                variable = tk.StringVar(value="—")
                self.matrix_cells.append(variable)
                ttk.Label(matrix, textvariable=variable).grid(row=row, column=col, padx=12)

        ttk.Label(
            panel,
            text=(
                "Video accuracy/F1 and box mAP measure different tasks on different datasets. "
                "MIVIA inference includes saved-frame "
                "loading and prediction, excluding video extraction/model loading; YOLO Val. is full validation time."
            ),
            wraplength=980,
        ).pack(anchor="w")
        ttk.Label(
            panel,
            text=(
                "The MoViNet results are exploratory because the official test set was reused after changing the selection criterion. "
                "Saved evaluation numbers are not confidence scores for the selected video."
            ),
            wraplength=980,
        ).pack(anchor="w", pady=(8, 10))
        ttk.Button(panel, text="Refresh Saved Results", command=self.load_evaluation).pack(anchor="w")
        self.load_evaluation()

    def load_evaluation(self):
        """Load saved results for MoViNet and YOLO."""
        self.eval_table.delete(*self.eval_table.get_children())
        self.evaluations: dict[str, dict] = {}
        self.matrix_title.set("Confusion matrix")
        for cell in self.matrix_cells:
            cell.set("—")
        try:
            models = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))["models"]
            unavailable = []
            for slug, name in CLASSIFIERS.items():
                try:
                    data = models[slug]
                    metrics = data["metrics"]
                    matrix = metrics["confusion_matrix"]
                    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
                        raise ValueError("Expected a 2×2 confusion matrix")
                    values = [name] + [
                        f"{float(metrics[key]):.2%}" for key in ("accuracy", "precision", "recall", "f1")
                    ]
                    for key in ("training_seconds", "inference_mean_ms_per_video", "model_size_mib"):
                        values.append("—" if data.get(key) is None else f"{float(data[key]):.2f}")
                    self.evaluations[slug] = data
                    self.eval_table.insert("", "end", iid=slug, values=values)
                except (KeyError, TypeError, ValueError) as exc:
                    unavailable.append(f"{name}: {exc}")
            if self.evaluations:
                self.eval_table.selection_set(next(iter(self.evaluations)))
                self.show_confusion()
            self.eval_status.set(
                "Source: results/comparison/model_comparison.json • Saved results; no retraining."
                + (" Missing/invalid: " + "; ".join(unavailable) if unavailable else "")
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.eval_status.set(f"Video classification results unavailable: {exc}")
        self.load_yolo_evaluation()

    def load_yolo_evaluation(self):
        """Display actual box metrics; never fabricate video classification metrics."""
        self.yolo_eval_table.delete(*self.yolo_eval_table.get_children())
        try:
            try:
                report = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))["models"].get(YOLO_SLUG)
            except (OSError, ValueError, KeyError, TypeError):
                report = None
            if report is None:
                report = json.loads(YOLO_EVALUATION_PATH.read_text(encoding="utf-8"))
            metrics = report["metrics"]
            values = ["YOLOv8n"] + [
                "—" if metrics.get(key) is None else f"{float(metrics[key]):.2%}"
                for key in ("precision", "recall", "map50", "map50_95")
            ]
            values.extend(
                "—" if report.get(key) is None else f"{float(report[key]):.2f}"
                for key in ("training_seconds", "validation_seconds")
            )
            size = report.get("model_size_bytes")
            values.append("—" if size is None else f"{float(size) / 1024 ** 2:.2f}")
            self.yolo_eval_table.insert("", "end", iid=YOLO_SLUG, values=values)
            self.yolo_eval_status.set(
                "Included in evaluation • " + str(report["evaluation_split"])
                + ". Test labels are unavailable, so this is validation performance, not an unseen labeled test."
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.yolo_eval_status.set(f"YOLO evaluation results unavailable: {exc}")

    def show_confusion(self, event=None):
        if not self.eval_table.selection():
            return
        slug = self.eval_table.selection()[0]
        if slug not in self.evaluations:
            return
        self.matrix_title.set(CLASSIFIERS[slug])
        matrix = self.evaluations[slug]["metrics"]["confusion_matrix"]
        descriptions = ("Correct rejection (TN)", "False alarm (FP)", "Missed dumping (FN)", "Detected dumping (TP)")
        for cell, value, description in zip(
            self.matrix_cells,
            (matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]),
            descriptions,
        ):
            cell.set(f"{value} — {description}")

    def choose_video(self):
        selected = filedialog.askopenfilename(
            title="Select surveillance video",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if not selected:
            return
        self.video = Path(selected)
        self.filename.set(str(self.video))
        self.table.delete(*self.table.get_children())
        self.yolo_table.delete(*self.yolo_table.get_children())
        self.yolo_details.set("YOLOv8n is ready when selected. No results for this video yet.")
        self.status.set("Ready to analyze the selected video.")
        self.run_button.configure(state="normal")

    def start(self):
        if self.busy or self.video is None:
            return
        model_names = MODEL_CHOICES[self.model_choice.get()]
        self.busy = True
        self.table.delete(*self.table.get_children())
        self.yolo_table.delete(*self.yolo_table.get_children())
        self.yolo_details.set("YOLOv8n analysis pending." if YOLO_SLUG in model_names else "YOLOv8n was not selected for this run.")
        self.select.configure(state="disabled")
        self.model_select.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.status.set("Selected algorithms are starting...")
        threading.Thread(target=self.worker, args=(self.video, model_names), daemon=True).start()

    def worker(self, video: Path, model_names=None):
        """Run selected modules off the GUI thread; retain partial successes."""
        model_names = tuple(MODEL_CHOICES[BOTH_MODELS] if model_names is None else model_names)
        actions, yolo, errors = [], None, []
        try:
            if not model_names or any(name not in (*CLASSIFIERS, YOLO_SLUG) for name in model_names):
                raise ValueError("Select a supported algorithm.")
            available = []
            for name in model_names:
                path = YOLO_MODEL if name == YOLO_SLUG else CHECKPOINT_PATHS[name]
                if not path.is_file():
                    errors.append(f"{CLASSIFIERS.get(name, 'YOLOv8n')}: trained model missing at {path}")
                elif name != YOLO_SLUG:
                    available.append(name)
            if available:
                try:
                    actions = analyze_video(
                        video, available, progress=lambda text: self.events.put(("status", str(text)))
                    )
                    if len(actions) != len(available) or {row["model"] for row in actions} != set(available):
                        actions = []
                        raise ValueError("Unexpected or missing video-classifier predictions.")
                except Exception as exc:
                    actions = []
                    errors.append(f"Video classification: {exc}")
            if YOLO_SLUG in model_names and YOLO_MODEL.is_file():
                try:
                    self.events.put(("status", "YOLOv8n: analyzing 12 sampled video frames..."))
                    yolo = run_yolo(video)
                except Exception as exc:
                    errors.append(f"YOLOv8n: {exc}")
            if not actions and yolo is None:
                raise RuntimeError("\n".join(errors) or "No algorithm produced results.")
            self.events.put(("complete", {"actions": actions, "yolo": yolo, "errors": errors}))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def finish_analysis(self):
        self.busy = False
        self.progress.stop()
        self.select.configure(state="normal")
        self.model_select.configure(state="readonly")
        self.run_button.configure(state="normal" if self.video else "disabled")

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.set(str(value))
                elif kind == "error":
                    self.finish_analysis()
                    self.status.set("Analysis failed. Select a valid video and try again.")
                    messagebox.showerror("Analysis failed", str(value))
                elif kind == "complete":
                    self.table.delete(*self.table.get_children())
                    for row in value["actions"]:
                        self.table.insert(
                            "",
                            "end",
                            values=(
                                CLASSIFIERS[row["model"]],
                                "DUMPING" if row["prediction"] else "NO DUMPING",
                                f"{row['score']:.4f}",
                                row["score_kind"],
                            ),
                        )
                    report = value["yolo"]
                    if report is not None:
                        detected = int(report["total_detections"])
                        strongest = report.get("strongest_detection")
                        self.yolo_table.insert(
                            "", "end", values=(
                                "YOLOv8n", "LITTERING REGIONS DETECTED" if detected else "NO REGIONS DETECTED",
                                "—" if strongest is None else f"{float(strongest['confidence']):.2%}",
                                detected, report["frames_sampled"],
                            ),
                        )
                        self.yolo_details.set(
                            f"Detection threshold: {report['confidence_threshold']:.2f}. "
                            f"Annotated frames: {report['output_dir']}"
                        )
                    if value["errors"]:
                        self.status.set("Analysis completed with module errors. Available results are shown.")
                        if report is None and any(error.startswith("YOLOv8n:") for error in value["errors"]):
                            self.yolo_details.set("YOLOv8n failed on this run. See the error message; no detection result is claimed.")
                        messagebox.showwarning("Some algorithms could not complete", "\n".join(value["errors"]))
                    else:
                        self.status.set("Analysis complete. Results are shown below.")
                    self.finish_analysis()
        except queue.Empty:
            pass
        self.after(100, self.poll)


if __name__ == "__main__":
    App().mainloop()
