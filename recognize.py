"""
recognize.py — Real-Time Face Recognition (Threaded ArcFace)
=============================================================
Camera display runs on the main thread at full FPS.
InsightFace inference + ChromaDB lookup run on a background thread.
Bounding boxes and labels from the last completed inference are overlaid
on every display frame — zero freezing even on slow CPUs.

Usage:
    python recognize.py
Controls:
    Press 'q' to quit.
"""

import cv2
import numpy as np
import sys
import time
import threading
import queue
from collections import deque

try:
    from insightface.app import FaceAnalysis
except ImportError:
    sys.exit("[ERROR] insightface not installed.")

try:
    import chromadb
except ImportError:
    sys.exit("[ERROR] chromadb not installed.")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH               = "./face_db"
COLLECTION_NAME       = "faces"
MODEL_NAME            = "buffalo_l"
RECOGNITION_THRESHOLD = 0.45
N_RESULTS             = 5
MIN_FACE_SIZE         = 60
DET_SIZE              = (320, 320)
FRAME_WIDTH           = 640
FRAME_HEIGHT          = 480
SMOOTH_WINDOW         = 5           # Frames for label smoothing


# ─────────────────────────────────────────────────────────────────────────────
# Inference Thread
# ─────────────────────────────────────────────────────────────────────────────

class InferenceThread(threading.Thread):
    """
    Background thread that handles:
      1. InsightFace face detection + ArcFace embedding extraction
      2. ChromaDB cosine similarity query

    Communication with main thread:
      Main  →  input_queue   (latest camera frame)
      This  →  result_queue  (list of (bbox, name, distance) tuples)
    """

    def __init__(self, analyzer, collection, threshold, n_results, min_face_size):
        super().__init__(daemon=True)
        self.analyzer      = analyzer
        self.collection    = collection
        self.threshold     = threshold
        self.n_results     = n_results
        self.min_face_size = min_face_size
        self.input_queue   = queue.Queue(maxsize=1)
        self.result_queue  = queue.Queue(maxsize=1)
        self._stop_event   = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # ── Step 1: Detect faces ──────────────────────────────────────
            faces = self.analyzer.get(frame)
            valid = [f for f in faces if self._is_large_enough(f)]

            # ── Step 2: Embed + query DB for each face ────────────────────
            detections = []
            for face in valid:
                emb = face.normed_embedding
                if emb is None:
                    detections.append((face.bbox.copy(), "Unknown", 1.0))
                    continue
                name, dist = self._query(emb)
                detections.append((face.bbox.copy(), name, dist))

            # Push result — drop stale result first if needed
            self._put_result(detections)

    def _is_large_enough(self, face):
        x1, y1, x2, y2 = face.bbox.astype(int)
        return (x2 - x1) >= self.min_face_size and (y2 - y1) >= self.min_face_size

    def _query(self, embedding):
        """Query ChromaDB. Returns (name, normalised_distance)."""
        count = self.collection.count()
        if count == 0:
            return "Unknown", 1.0
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(self.n_results, count),
            include=["metadatas", "distances"],
        )
        dist = results["distances"][0][0] / 2.0   # [0,2] → [0,1]
        name = results["metadatas"][0][0]["name"]
        return name, dist

    def _put_result(self, result):
        """Non-blocking put — always replace stale result."""
        if not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.result_queue.put_nowait(result)
        except queue.Full:
            pass

    def submit_frame(self, frame):
        """Non-blocking frame submit — always send the latest frame."""
        if not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.input_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_result(self):
        """Returns latest detections list or None."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# Label Smoother
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoother:
    """
    Stabilises flickering labels using a centroid-proximity tracker.
    Each track keeps a short history → majority-vote name + mean distance.
    """

    def __init__(self, window=SMOOTH_WINDOW, max_age=20, proximity=100):
        self.tracks    = {}
        self.next_id   = 0
        self.window    = window
        self.max_age   = max_age
        self.proximity = proximity

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox.astype(int)
        return np.array([(x1+x2)/2, (y1+y2)/2], dtype=float)

    def _match(self, c):
        best_id, best_d = None, self.proximity
        for tid, t in self.tracks.items():
            d = float(np.linalg.norm(c - t["c"]))
            if d < best_d:
                best_id, best_d = tid, d
        return best_id

    def update(self, detections):
        """
        detections: list of (bbox, name, distance)
        Returns:    list of (bbox, smoothed_name, smoothed_distance)
        """
        matched = set()
        output  = []

        for bbox, name, dist in detections:
            c   = self._centroid(bbox)
            tid = self._match(c)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"c": c, "h": deque(maxlen=self.window), "age": 0}

            t = self.tracks[tid]
            t["c"]   = c
            t["age"] = 0
            t["h"].append((name, dist))
            matched.add(tid)

            names  = [h[0] for h in t["h"]]
            dists  = [h[1] for h in t["h"]]
            output.append((bbox, max(set(names), key=names.count), float(np.mean(dists))))

        # Age out unmatched tracks
        for tid in list(self.tracks):
            if tid not in matched:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        return output


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_result(frame, bbox, name, distance, threshold):
    x1, y1, x2, y2 = bbox.astype(int)
    known  = distance < threshold
    color  = (0, 200, 0) if known else (0, 0, 220)
    label  = f"{name}  {distance:.2f}" if known else f"Unknown  {distance:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_face_analyzer():
    print("[INFO] Loading InsightFace model …")
    app = FaceAnalysis(
        name=MODEL_NAME,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    print("[INFO] Model ready.")
    return app


def get_chroma_collection(db_path, collection_name):
    client = chromadb.PersistentClient(path=db_path)
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        sys.exit(
            f"[ERROR] Collection '{collection_name}' not found.\n"
            "        Run register.py first."
        )


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam.")
    return cap


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    analyzer   = init_face_analyzer()
    collection = get_chroma_collection(DB_PATH, COLLECTION_NAME)
    smoother   = LabelSmoother()
    cap        = open_camera()

    # Start background inference thread
    worker = InferenceThread(
        analyzer, collection,
        RECOGNITION_THRESHOLD, N_RESULTS, MIN_FACE_SIZE
    )
    worker.start()

    total_reg = collection.count()
    print(f"[INFO] DB: {total_reg} embeddings  |  Threshold: {RECOGNITION_THRESHOLD}")
    print("[INFO] Inference thread started. Press 'q' to quit.\n")

    fps_hist      = deque(maxlen=30)
    prev_time     = time.time()
    last_results  = []    # Last smoothed detections — drawn every frame

    # ── Main display loop ─────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame   = cv2.flip(frame, 1)
        display = frame.copy()

        # Send frame to inference thread (non-blocking, always latest frame)
        worker.submit_frame(frame)

        # Check if inference thread has a new result ready
        raw = worker.get_result()
        if raw is not None:
            # Smooth the new result and cache it
            last_results = smoother.update(raw)

        # Draw cached results — runs every frame even when inference is busy
        for bbox, name, dist in last_results:
            draw_result(display, bbox, name, dist, RECOGNITION_THRESHOLD)

        # ── Overlay stats ─────────────────────────────────────────────────────
        now = time.time()
        fps_hist.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        avg_fps   = np.mean(fps_hist)

        cv2.putText(display, f"FPS: {avg_fps:.1f}",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display,
                    f"Faces: {len(last_results)}  DB: {total_reg}  Thread: inference",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Face Recognition — press 'q' to quit", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
