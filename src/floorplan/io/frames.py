"""Video to still frames for the video tier."""
from pathlib import Path

import cv2


def sharpness(bgr) -> float:
    """Variance of the Laplacian of a 480 px wide grayscale copy: low for blurred or featureless frames."""
    g = cv2.cvtColor(cv2.resize(bgr, (480, int(bgr.shape[0] * 480 / bgr.shape[1])), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def extract_video_frames(video: str | Path, out_dir: str | Path, fps: float, width: int, sharpest: bool = False) -> list[tuple[int, Path]]:
    """Decode sequentially and keep about `fps` frames per second, resized to `width` pixels wide, written as JPEGs named f<index>.jpg.

    With `sharpest`, each time window of 1/fps seconds contributes its sharpest frame instead of the one at a fixed time: hand-held turns blur
    some frames and a blurred frame has no features to match. Returns (frame index in the video, path).
    Sequential decoding because seeking in mp4 can land on the wrong frame.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / fps))
    res: list[tuple[int, Path]] = []
    best = None  # (sharpness, index, frame) within the current window

    def write(i, bgr):
        h = int(round(bgr.shape[0] * width / bgr.shape[1]))
        p = out / f"f{i:06d}.jpg"
        cv2.imwrite(str(p), cv2.resize(bgr, (width, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
        res.append((i, p))

    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if sharpest:
            s = sharpness(bgr)
            if best is None or s > best[0]:
                best = (s, i, bgr)
            if (i + 1) % step == 0:
                write(best[1], best[2])
                best = None
        elif i % step == 0:
            write(i, bgr)
        i += 1
    if sharpest and best is not None:
        write(best[1], best[2])
    cap.release()
    return res
