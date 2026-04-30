"""
register.py — Face Registration (Threaded ArcFace)
===================================================
Camera display runs on the main thread.
InsightFace inference runs on a dedicated background thread.
This keeps the webcam feed perfectly smooth with zero freezing.

Usage:
    python register.py
Controls:
    Press 'q' to quit.
"""

import cv2
import numpy as np
import uuid
import time
import sys
import threading
import queue

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
DB_PATH          = "./face_db"
COLLECTION_NAME  = "faces"
SAMPLES_REQUIRED = 7
CAPTURE_INTERVAL = 0.8          # Min seconds between saved samples
MIN_FACE_SIZE    = 60
MODEL_NAME       = "buffalo_l"
DET_SIZE         = (320, 320)
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480


# ─────────────────────────────────────────────────────────────────────────────
# Inference Thread
# ─────────────────────────────────────────────────────────────────────────────

class InferenceThread(threading.Thread):
    """
    Runs InsightFace detection + embedding in the background.

    Main thread  →  puts frames into  self.input_queue
    This thread  →  puts results into self.result_queue
    """

    def __init__(self, analyzer):
        super().__init__(daemon=True)   # Dies automatically when main thread exits
        self.analyzer     = analyzer
        self.input_queue  = queue.Queue(maxsize=1)   # Only keep the latest frame
        self.result_queue = queue.Queue(maxsize=1)   # Only keep the latest result
        self._stop_event  = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                # Block up to 0.1s waiting for a new frame
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # ── Run InsightFace (this is the slow part) ───────────────────
            faces = self.analyzer.get(frame)

            # Discard old result if main thread hasn't consumed it yet
            if not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass

            self.result_queue.put(faces)

    def stop(self):
        self._stop_event.set()

    def submit_frame(self, frame):
        """Send a frame for inference. Drops old frame if queue is full."""
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
        """Returns latest faces list, or None if not ready yet."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None


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
    print("[INFO] Model loaded.")
    return app


def get_chroma_collection(db_path, collection_name):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
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


def is_face_large_enough(face, min_size):
    x1, y1, x2, y2 = face.bbox.astype(int)
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def draw_face_box(frame, face, label="", color=(0, 220, 0)):
    x1, y1, x2, y2 = face.bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    user_name = input("Enter the name to register: ").strip()
    if not user_name:
        sys.exit("[ERROR] Name cannot be empty.")
    print(f"\n[INFO] Registering: '{user_name}'  |  Samples needed: {SAMPLES_REQUIRED}")
    print("[INFO] Press 'q' to quit.\n")

    # ── Setup ─────────────────────────────────────────────────────────────────
    analyzer   = init_face_analyzer()
    collection = get_chroma_collection(DB_PATH, COLLECTION_NAME)
    cap        = open_camera()

    # Start inference thread
    worker = InferenceThread(analyzer)
    worker.start()
    print("[INFO] Inference thread started. Webcam is live …\n")

    captured          = 0
    last_capture_time = 0.0
    last_faces        = []          # Last known detection result
    embeddings_batch  = []
    ids_batch         = []
    meta_batch        = []

    # ── Main display loop (never blocks on inference) ─────────────────────────
    while captured < SAMPLES_REQUIRED:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame   = cv2.flip(frame, 1)
        display = frame.copy()

        # Send frame to inference thread (non-blocking)
        worker.submit_frame(frame)

        # Collect latest result if inference thread is done (non-blocking)
        result = worker.get_result()
        if result is not None:
            last_faces = result

        # ── Draw using the most recent detection result ───────────────────────
        status = f"Captured: {captured}/{SAMPLES_REQUIRED}"

        if not last_faces:
            cv2.putText(display, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)
        else:
            face = max(last_faces,
                       key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

            if not is_face_large_enough(face, MIN_FACE_SIZE):
                cv2.putText(display, "Move closer", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                draw_face_box(display, face, color=(0, 165, 255))
            else:
                draw_face_box(display, face, user_name)

                now = time.time()
                if now - last_capture_time >= CAPTURE_INTERVAL:
                    emb = face.normed_embedding
                    if emb is not None:
                        embeddings_batch.append(emb.tolist())
                        ids_batch.append(str(uuid.uuid4()))
                        meta_batch.append({"name": user_name})
                        captured         += 1
                        last_capture_time = now
                        print(f"  ✔  Captured {captured}/{SAMPLES_REQUIRED}")

        cv2.putText(display, status,
                    (20, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Inference: background thread",
                    (20, display.shape[0] - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Registration — press 'q' to quit", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Cancelled by user.")
            worker.stop()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    worker.stop()
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] Saving {len(embeddings_batch)} embeddings to ChromaDB …")
    collection.add(
        embeddings=embeddings_batch,
        ids=ids_batch,
        metadatas=meta_batch,
    )
    print(f"[SUCCESS] '{user_name}' registered with {captured} samples.")
    print(f"[INFO] Total embeddings in DB: {collection.count()}")


if __name__ == "__main__":
    main()
