"""Fast single-frame grabs from an HLS media playlist.

ffmpeg's HLS seek (`-ss T -i playlist.m3u8`) reads several segments to land on
T — 20-70 s per frame on a slow line. Instead: parse the playlist once (cached),
download only the segment that holds T in one request, and decode the frame
from it locally. Falls back to the ffmpeg HLS seek when the playlist isn't a
plain media playlist (e.g. a master playlist) or anything fails.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import urllib.parse
import urllib.request

import cv2
import numpy as np

_playlists: dict[str, list[tuple[float, float, str]]] = {}
_lock = threading.Lock()


def _segments(url: str) -> list[tuple[float, float, str]]:
    with _lock:
        if url in _playlists:
            return _playlists[url]
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8", "replace")
    segs: list[tuple[float, float, str]] = []
    acc, dur = 0.0, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line and not line.startswith("#") and dur is not None:
            segs.append((acc, dur, urllib.parse.urljoin(url, line)))
            acc += dur
            dur = None
    with _lock:
        _playlists[url] = segs
    return segs


def playlist_duration(url: str) -> float:
    """Total length (s) of an HLS media playlist, from its segment list."""
    segs = _segments(url)
    return segs[-1][0] + segs[-1][1] if segs else 0.0


def _decode(path_or_url: str, ss: float, input_seek: bool) -> np.ndarray | None:
    pre = ["-ss", f"{ss:.3f}"] if input_seek else []
    post = [] if input_seek else ["-ss", f"{ss:.3f}"]
    net = (["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
            "-rw_timeout", "30000000"]   # give up on a stalled connection after 30 s
           if path_or_url.startswith("http") else [])
    cmd = ["ffmpeg", "-nostdin", "-v", "error", *net, *pre, "-i", path_or_url, *post,
           "-frames:v", "1", "-an", "-sn", "-f", "image2pipe", "-vcodec", "png", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    except subprocess.TimeoutExpired:
        return None
    return cv2.imdecode(np.frombuffer(out, dtype=np.uint8), cv2.IMREAD_COLOR) if out else None


def grab_hls_frame(url: str, t: float) -> np.ndarray | None:
    """One BGR frame at `t` seconds of an HLS stream (segment fetch, else ffmpeg seek)."""
    try:
        segs = _segments(url) if ".m3u8" in url else []
        seg = next((s for s in segs if t < s[0] + s[1]), None)
        if seg:
            fd, tmp = tempfile.mkstemp(suffix=".ts")
            try:
                with urllib.request.urlopen(seg[2], timeout=60) as r, os.fdopen(fd, "wb") as f:
                    f.write(r.read())
                frame = _decode(tmp, max(0.0, min(seg[1] - 0.05, t - seg[0])), input_seek=False)
                if frame is not None:
                    return frame
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    except Exception:
        pass
    return _decode(url, t, input_seek=True)


def preview_url(url: str) -> str:
    """The 720p rendition of a Pixellot venue stream (same camera + timeline,
    ~5x less data), else the URL itself."""
    return url.replace("/venue_hls/hd_hls/hd1/", "/venue_hls/hd_hls/hd0/")


def grab_preview_frame(url: str, t: float, size: tuple[int, int] = (1920, 1080)) -> np.ndarray | None:
    """A frame for analysis that doesn't need full resolution (scoreboard
    discovery, change heatmaps): from the 720p rendition, upscaled to `size`
    so every pixel coordinate stays in the full-HD frame space."""
    p = preview_url(url)
    frame = grab_hls_frame(p, t) if p != url else None
    if frame is None:
        return grab_hls_frame(url, t)
    if (frame.shape[1], frame.shape[0]) != size:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
    return frame
