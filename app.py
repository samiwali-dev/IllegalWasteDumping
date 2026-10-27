"""Desktop interface for MIVIA action recognition plus optional YOLO evidence.

Run with:
    .venv/bin/python app.py
"""
from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from common import PROJECT_ROOT, analyze_video


MODELS = {
    "Compare all four models": None,
    "Optical Flow + SVM": "optical_flow_svm",
    "CNN + LSTM": "cnn_lstm",
    "3D CNN": "cnn_3d",
    "MoViNet A0": "movinet_a0",
}
NAMES = {value: key for key, value in MODELS.items() if value}
YOLO_MODEL = PROJECT_ROOT / "src" / "yolo" / "model.pt"


class App(tk.Tk):
    """Keep MIVIA video classification and YOLO image-region evidence distinct."""

    def __init__(self):
        super().__init__()
        self.title("Illegal Waste Dumping Detection")
        self.geometry("1100x890")
        self.minsize(970, 750)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.video: Path | None = None
        self.yolo_output: Path | None = None
        self.busy = False
        self.model = tk.StringVar(value="Compare all four models")
        self.use_yolo = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Select an MP4 video to begin.")
        self.filename = tk.StringVar(value="No video selected")
        self.yolo_status = tk.StringVar(value="YOLO scan will use 12 evenly spaced video frames.")
        self.yolo_summary = tk.StringVar(value="No YOLO scan has been run.")

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
                "One selected video is analysed by the MIVIA action models and, if enabled, "
                "by YOLOv8n on 12 sampled frames."
            ),
            wraplength=1000,
        ).pack(anchor="w", pady=(6, 16))

        controls = ttk.Frame(panel)
        controls.pack(fill="x")
        self.select = ttk.Button(controls, text="Select Video", command=self.choose_video)
        self.select.pack(side="left")
        self.selector = ttk.Combobox(
            controls,
            textvariable=self.model,
            values=list(MODELS),
            state="readonly",
            width=29,
        )
        self.selector.pack(side="left", padx=(14, 10))
        self.yolo_check = ttk.Checkbutton(
            controls,
            text="Run YOLO scan (12 frames)",
            variable=self.use_yolo,
            command=self.refresh_yolo_option_text,
        )
        self.yolo_check.pack(side="left", padx=(0, 12))
        self.run_button = ttk.Button(controls, text="Analyze Video", command=self.start, state="disabled")
        self.run_button.pack(side="left")
        self.open_yolo_button = ttk.Button(
            controls,
            text="Open YOLO Frames",
            command=self.open_yolo_output,
            state="disabled",
        )
        self.open_yolo_button.pack(side="left", padx=(12, 0))

        ttk.Label(panel, textvariable=self.filename, wraplength=1000).pack(anchor="w", pady=(14, 6))
        self.progress = ttk.Progressbar(panel, mode="indeterminate")
        self.progress.pack(fill="x")
        ttk.Label(panel, textvariable=self.status, wraplength=1000).pack(anchor="w", pady=(8, 3))
        ttk.Label(panel, textvariable=self.yolo_status, wraplength=1000).pack(anchor="w", pady=(0, 12))

        action_box = ttk.LabelFrame(panel, text="Action Analysis — MIVIA video classifiers", padding=10)
        action_box.pack(fill="both", expand=True)
        columns = ("model", "result", "score", "meaning")
        self.table = ttk.Treeview(action_box, columns=columns, show="headings", height=5)
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
            text=(
                "Each row is a separate model prediction. The models' scores use different scales, "
                "so they are not directly comparable confidence percentages."
            ),
            wraplength=980,
        ).pack(anchor="w", pady=(8, 0))

        yolo_box = ttk.LabelFrame(panel, text="Littering Evidence — YOLOv8n (separate supporting detector)", padding=10)
        yolo_box.pack(fill="both", expand=True, pady=(14, 0))
        ttk.Label(yolo_box, textvariable=self.yolo_summary, wraplength=980).pack(anchor="w", pady=(0, 7))
        yolo_columns = ("frame", "time", "class", "confidence", "bbox")
        self.yolo_table = ttk.Treeview(yolo_box, columns=yolo_columns, show="headings", height=4)
        for name, title, width in zip(
            yolo_columns,
            ("Frame", "Time (s)", "Class", "Confidence", "Bounding box (x1, y1, x2, y2)"),
            (70, 95, 160, 120, 410),
        ):
            self.yolo_table.heading(name, text=title)
            self.yolo_table.column(name, width=width, minwidth=75)
        self.yolo_table.pack(fill="both", expand=True)
        ttk.Label(
            yolo_box,
            text=(
                "YOLO identifies labelled littering regions in individual frames. It is supporting visual evidence only; "
                "it does not decide whether the person performed a dumping action."
            ),
            wraplength=980,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Label(
            panel,
            text=(
                "This demo classifies the action in the whole video; it does not identify the exact dumped object "
                "or the exact event time. AI predictions should be reviewed by an authorized human."
            ),
            wraplength=1000,
        ).pack(anchor="w", pady=(14, 0))

    def build_evaluation_tab(self):
        panel = ttk.Frame(self.tabs, padding=22)
        self.tabs.add(panel, text="Evaluation Results")
        ttk.Label(panel, text="Evaluation Results", font=("Helvetica", 24, "bold")).pack(anchor="w")

        mivia_box = ttk.LabelFrame(panel, text="MIVIA action-classification evaluation", padding=10)
        mivia_box.pack(fill="x", pady=(12, 10))
        ttk.Label(
            mivia_box,
            text=(
                "Same 100 official MIVIA test videos: 50 dumping / 50 no dumping. "
                "Training: 320 videos • Validation: 80 videos • 24 frames per video."
            ),
            wraplength=980,
        ).pack(anchor="w")
        self.eval_status = tk.StringVar()
        ttk.Label(mivia_box, textvariable=self.eval_status, wraplength=980).pack(anchor="w", pady=(5, 8))
        columns = ("model", "accuracy", "precision", "recall", "f1", "train", "infer", "size")
        self.eval_table = ttk.Treeview(mivia_box, columns=columns, show="headings", height=4, selectmode="browse")
        for col, title, width in zip(
            columns,
            ("Model", "Accuracy", "Precision", "Recall", "F1", "Train (s)", "Infer (ms)", "Size (MiB)"),
            (175, 90, 90, 90, 90, 95, 95, 95),
        ):
            self.eval_table.heading(col, text=title)
            self.eval_table.column(col, width=width, minwidth=width, anchor="w" if col == "model" else "center")
        self.eval_table.pack(fill="x")
        self.eval_table.bind("<<TreeviewSelect>>", self.show_confusion)
        ttk.Label(mivia_box, text="Select a model above to view its confusion matrix.").pack(anchor="w", pady=(7, 0))

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

        yolo_box = ttk.LabelFrame(panel, text="YOLOv8n object-detection evaluation — separate task", padding=10)
        yolo_box.pack(fill="x", pady=(0, 10))
        ttk.Label(
            yolo_box,
            text=(
                "These are bounding-box detection metrics, not a fifth result in the MIVIA video-classification comparison."
            ),
            wraplength=980,
        ).pack(anchor="w")
        self.yolo_eval_status = tk.StringVar()
        ttk.Label(yolo_box, textvariable=self.yolo_eval_status, wraplength=980).pack(anchor="w", pady=(5, 8))
        yolo_columns = ("dataset", "precision", "recall", "map50", "map50_95", "train", "size")
        self.yolo_eval_table = ttk.Treeview(yolo_box, columns=yolo_columns, show="headings", height=1)
        for col, title, width in zip(
            yolo_columns,
            ("Evaluation split", "Precision", "Recall", "mAP@50", "mAP@50:95", "Train (s)", "Size (MiB)"),
            (350, 100, 100, 100, 110, 110, 110),
        ):
            self.yolo_eval_table.heading(col, text=title)
            self.yolo_eval_table.column(col, width=width, minwidth=80, anchor="w" if col == "dataset" else "center")
        self.yolo_eval_table.pack(fill="x")

        ttk.Label(
            panel,
            text=(
                "Accuracy: all correct predictions. Precision: correctness of dumping alerts. Recall: proportion of actual "
                "dumping detected. F1: balance of precision and recall. Inference time includes saved-frame loading and prediction; "
                "it excludes video extraction and model loading."
            ),
            wraplength=980,
        ).pack(anchor="w")
        ttk.Label(
            panel,
            text=(
                "MIVIA results are exploratory because the official test set was reused after changing the selection criterion. "
                "These saved evaluation numbers are not confidence scores for the selected video."
            ),
            wraplength=980,
        ).pack(anchor="w", pady=(8, 10))
        ttk.Button(panel, text="Refresh Saved Results", command=self.load_evaluation).pack(anchor="w")
        self.load_evaluation()

    def load_evaluation(self):
        """Refresh the saved MIVIA and YOLO metrics without retraining either model."""
        self.load_mivia_evaluation()
        self.load_yolo_evaluation()

    def load_mivia_evaluation(self):
        self.eval_table.delete(*self.eval_table.get_children())
        self.evaluations: dict[str, dict] = {}
        self.matrix_title.set("Confusion matrix")
        for cell in self.matrix_cells:
            cell.set("—")
        try:
            path = PROJECT_ROOT / "results" / "comparison" / "model_comparison.json"
            models = json.loads(path.read_text(encoding="utf-8"))["models"]
            rows = []
            for name, data in models.items():
                metrics = data["metrics"]
                matrix = metrics["confusion_matrix"]
                if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
                    raise ValueError("Expected a 2×2 MIVIA confusion matrix")
                values = [NAMES.get(name, name)] + [
                    f"{float(metrics[key]):.2%}" for key in ("accuracy", "precision", "recall", "f1")
                ]
                for key in ("training_seconds", "inference_mean_ms_per_video", "model_size_mib"):
                    values.append("—" if data.get(key) is None else f"{float(data[key]):.2f}")
                rows.append((name, values))
            self.evaluations = models
            for name, values in rows:
                self.eval_table.insert("", "end", iid=name, values=values)
            self.eval_status.set("Source: results/comparison/model_comparison.json • Saved evaluation; no retraining required.")
            if rows:
                self.eval_table.selection_set(rows[0][0])
                self.show_confusion()
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.eval_status.set(f"MIVIA evaluation results unavailable: {exc}")

    def load_yolo_evaluation(self):
        self.yolo_eval_table.delete(*self.yolo_eval_table.get_children())
        try:
            path = PROJECT_ROOT / "results" / "yolo" / "yolov8n_run.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            metrics = report["metrics"]
            size_mib = float(report["model_size_bytes"]) / (1024 * 1024)
            values = (
                report["evaluation_split"],
                f"{float(metrics['precision']):.2%}",
                f"{float(metrics['recall']):.2%}",
                f"{float(metrics['map50']):.2%}",
                f"{float(metrics['map50_95']):.2%}",
                f"{float(report['training_seconds']):.2f}",
                f"{size_mib:.2f}",
            )
            self.yolo_eval_table.insert("", "end", values=values)
            self.yolo_eval_status.set(
                "Source: results/yolo/yolov8n_run.json • YOLOv8n trained from scratch; supplied test images have no labels."
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.yolo_eval_status.set(f"YOLO evaluation results unavailable: {exc}")

    def show_confusion(self, event=None):
        selection = self.eval_table.selection()
        if not selection:
            return
        name = selection[0]
        self.matrix_title.set(NAMES.get(name, name))
        matrix = self.evaluations[name]["metrics"]["confusion_matrix"]
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
        self.yolo_output = None
        self.filename.set(str(self.video))
        self.table.delete(*self.table.get_children())
        self.yolo_table.delete(*self.yolo_table.get_children())
        self.status.set("Ready to analyze the selected video.")
        self.yolo_summary.set("No YOLO scan has been run for this video.")
        self.open_yolo_button.configure(state="disabled")
        self.refresh_yolo_option_text()
        self.run_button.configure(state="normal")

    def refresh_yolo_option_text(self):
        if self.use_yolo.get():
            self.yolo_status.set("YOLO scan enabled: the same video will be checked at 12 evenly spaced frames.")
        else:
            self.yolo_status.set("YOLO scan disabled: only the MIVIA action classifier will run.")

    def start(self):
        if self.busy or self.video is None:
            return
        selected = MODELS[self.model.get()]
        names = [selected] if selected else list(NAMES)
        run_yolo = self.use_yolo.get()
        if run_yolo and not YOLO_MODEL.is_file():
            messagebox.showerror(
                "YOLO model missing",
                f"The trained YOLO model was not found:\n{YOLO_MODEL}\n\nRun src/yolo/main.py train first or disable the YOLO scan.",
            )
            return

        self.busy = True
        self.yolo_output = None
        self.table.delete(*self.table.get_children())
        self.yolo_table.delete(*self.yolo_table.get_children())
        self.yolo_summary.set("Waiting for the YOLO scan." if run_yolo else "YOLO scan was disabled for this analysis.")
        self.select.configure(state="disabled")
        self.selector.configure(state="disabled")
        self.yolo_check.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.open_yolo_button.configure(state="disabled")
        self.progress.start(12)
        self.status.set("Action analysis is starting...")
        self.yolo_status.set("YOLO scan is waiting for action analysis." if run_yolo else "YOLO scan disabled.")
        threading.Thread(target=self.worker, args=(self.video, names, run_yolo), daemon=True).start()

    def worker(self, video: Path, names: list[str], run_yolo: bool):
        try:
            action_rows = analyze_video(
                video,
                names,
                progress=lambda text: self.events.put(("status", f"Action: {text}")),
            )
            yolo_result = None
            if run_yolo:
                self.events.put(("yolo_status", "YOLO: scanning 12 evenly spaced frames..."))
                from src.yolo.main import detect_video

                yolo_result = detect_video(video, frames=12)
            self.events.put(("complete", {"action_rows": action_rows, "yolo_result": yolo_result}))
        except Exception as exc:  # User-facing UI should show a readable error dialog.
            self.events.put(("error", str(exc)))

    def populate_yolo_evidence(self, result: dict | None):
        self.yolo_table.delete(*self.yolo_table.get_children())
        self.yolo_output = None
        if result is None:
            self.yolo_summary.set("YOLO scan was disabled for this analysis.")
            self.yolo_status.set("YOLO scan disabled.")
            return

        for frame in result["frame_results"]:
            for detection in frame["detections"]:
                self.yolo_table.insert(
                    "",
                    "end",
                    values=(
                        frame["frame_number"],
                        f"{frame['timestamp_seconds']:.2f}",
                        detection["class_name"],
                        f"{detection['confidence']:.2%}",
                        ", ".join(str(value) for value in detection["bbox_xyxy"]),
                    ),
                )
        if not result["total_detections"]:
            self.yolo_table.insert("", "end", values=("—", "—", "No regions at selected threshold", "—", "—"))

        strongest = result.get("strongest_detection")
        if strongest is None:
            strongest_text = "No region reached the 0.25 detection threshold."
        else:
            strongest_text = (
                f"Strongest: {strongest['class_name']} at {strongest['confidence']:.2%} "
                f"(frame {strongest['frame_number']}, {strongest['timestamp_seconds']:.2f}s)."
            )
        self.yolo_output = Path(result["output_dir"])
        self.yolo_summary.set(
            f"YOLO scanned {result['frames_sampled']} frames and found {result['total_detections']} region(s) "
            f"in {result['frames_with_detections']} frame(s). {strongest_text}"
        )
        self.yolo_status.set(f"YOLO scan complete. Annotated frames saved in: {self.yolo_output}")
        self.open_yolo_button.configure(state="normal")

    def open_yolo_output(self):
        if self.yolo_output and self.yolo_output.exists():
            webbrowser.open(self.yolo_output.as_uri())

    def finish_analysis(self):
        self.busy = False
        self.progress.stop()
        self.select.configure(state="normal")
        self.selector.configure(state="readonly")
        self.yolo_check.configure(state="normal")
        self.run_button.configure(state="normal" if self.video else "disabled")

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.set(str(value))
                    continue
                if kind == "yolo_status":
                    self.yolo_status.set(str(value))
                    continue
                if kind == "error":
                    self.finish_analysis()
                    self.status.set("Analysis failed. Select a valid video and try again.")
                    self.yolo_status.set("YOLO scan did not complete.")
                    messagebox.showerror("Analysis failed", str(value))
                    continue
                if kind == "complete":
                    result = value
                    for row in result["action_rows"]:
                        self.table.insert(
                            "",
                            "end",
                            values=(
                                NAMES[row["model"]],
                                "DUMPING" if row["prediction"] else "NO DUMPING",
                                f"{row['score']:.4f}",
                                row["score_kind"],
                            ),
                        )
                    self.status.set(
                        f"Action analysis complete — {len(result['action_rows'])} model(s), 24 shared frames."
                    )
                    self.populate_yolo_evidence(result["yolo_result"])
                    self.finish_analysis()
        except queue.Empty:
            pass
        self.after(100, self.poll)


if __name__ == "__main__":
    App().mainloop()
