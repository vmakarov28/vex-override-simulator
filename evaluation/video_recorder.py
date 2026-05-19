"""
evaluation/video_recorder.py
────────────────────────────────────────────────────────────────────────────
Saves pygame frames to an MP4 file using OpenCV (cv2).

Usage
-----
    recorder = VideoRecorder("output.mp4", screen_w, screen_h, fps=30)
    recorder.write_frame(numpy_rgb_frame)   # called once per render step
    recorder.close()

Falls back gracefully if cv2 is not installed.
"""

import os
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[VideoRecorder] opencv-python not installed. "
          "Run: pip install opencv-python")


class VideoRecorder:
    """
    Records frames to an MP4 video file.

    Parameters
    ----------
    path   : str   — output file path (e.g. 'artifacts/videos/match_001.mp4')
    width  : int   — frame width in pixels
    height : int   — frame height in pixels
    fps    : int   — frames per second (default 30)
    """

    def __init__(self, path: str, width: int, height: int, fps: int = 30):
        self.path   = path
        self.width  = width
        self.height = height
        self.fps    = fps
        self._writer = None
        self._frame_count = 0

        if not CV2_AVAILABLE:
            print("[VideoRecorder] cv2 not available — recording disabled.")
            return

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                    exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            print(f"[VideoRecorder] Could not open writer for {path}")
            self._writer = None

    def write_frame(self, frame_rgb: np.ndarray):
        """
        Write one frame. frame_rgb should be shape (H, W, 3), dtype uint8,
        in RGB order (as returned by pygame.surfarray.array3d().T).
        """
        if self._writer is None:
            return
        # OpenCV expects BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._writer.write(frame_bgr)
        self._frame_count += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"[VideoRecorder] Wrote {self._frame_count} frames → {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
