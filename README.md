# plyOCR

Read a basketball broadcast's scoreboard (game clock, quarter, home and away score)
from video, frame by frame, into a clean timeline CSV.

It handles a local video file or a remote HLS stream, runs Tesseract on small
regions of interest (ROIs) around each scoreboard field, and cleans the readings
(smoothing, quarter inference from the clock, break skipping). Optional readers:
a seven-segment decoder for LED boards, per-court quarter-digit templates, and
EasyOCR.

## Install

Needs the `tesseract` and `ffmpeg` binaries on the PATH
(`brew install tesseract ffmpeg` / `apt-get install tesseract-ocr ffmpeg`).

```bash
pip install "plyocr @ git+https://github.com/asharon10/plyOCR@v0.1.2"
# extras: plyocr[claude] (Claude vision ROI review), plyocr[easyocr], plyocr[dev]
```

Python 3.9+.

## Input: a game folder

```
<game>/
  game.json          required
  roi.json           optional — scoreboard boxes for this game (else a learned template / the default)
  markers.json       optional — known period start/end times, bounds the scan
  app/data.json      optional — synced events with quarter_markers (quarter templates, spot checks)
  raw/game.mp4       optional — local video (else video.downloadUrl is streamed)
```

`game.json` fields read:

| Field | Meaning |
|---|---|
| `video.downloadUrl` (or `videoUrl`) | Video file or HLS `.m3u8` URL |
| `video.localFile` | Local video path, relative to the game folder |
| `cloudMode` | `true` → always stream the URL, even with a local file |
| `ocrMethod` | `tesseract` (default), `sevenseg` or `easyocr` |
| `ocrQuarterMethod` | Read just the quarter with a different method |
| `ocrGrayscale` | Grayscale preprocessing instead of colour-aware |
| `ocrSkipBreaks` | Skip long breaks (default on) |
| `gameFormat` | `{quarters, quarterMinutes, otMinutes}` — default FIBA 4 × 10, OT 5 |

`roi.json`:

```json
{
  "frame_width": 1920, "frame_height": 1080,
  "rois": {
    "clock":      {"x": 404,  "y": 969, "w": 133, "h": 67, "color": "red",    "font": "clock"},
    "quarter":    {"x": 565,  "y": 967, "w": 68,  "h": 55, "color": "yellow", "font": "quarter"},
    "home_score": {"x": 1189, "y": 968, "w": 83,  "h": 55, "color": "red",    "font": "score"},
    "away_score": {"x": 660,  "y": 968, "w": 76,  "h": 57, "color": "red",    "font": "score"}
  }
}
```

Boxes are in `frame_width × frame_height` pixels and scale to the real video size.

## Output

A full run writes to `<game>/derived/` (or `--derived <dir>`):

| File | Contents |
|---|---|
| `ocr_timeline.csv` | `video_time_seconds,quarter,game_clock,home_score,away_score,confidence` |
| `timeline.json` | The same cleaned rows as JSON |
| `timeline_raw.json` | Every sampled frame, before cleaning |
| `ocr_coverage.json` | How much of the game was read |
| `break_plan.json` | Long breaks that were skipped |

Progress is printed as `PROGRESS <done>/<total>` lines for a job runner to follow.

## Commands

```bash
python -m plyocr.run_tesseract_ocr --game <game>/game.json                  # full game
python -m plyocr.run_tesseract_ocr --game <game>/game.json \
    --start 600 --end 900 --fps 1 --out range.json                           # one window → JSON
python -m plyocr.run_tesseract_ocr --game <game>/game.json --detect-method   # pick the best method
python -m plyocr.detect_rois --video <video or URL>                          # find the scoreboard boxes
python -m plyocr.read_frame --game <game>/game.json --at 2674                # read one frame
python -m plyocr.verify_ocr --game <game>/game.json --n 20                   # annotated spot checks
```

Also: `spot_check`, `spots_roi`, `change_map`, `quarter_templates`, `compare_decoders`,
`extract_scoreboard`, `claude_review` — each has `--help`.

## Run on GitHub Actions

`.github/workflows/ocr-job.yml` ("OCR job", manual trigger) runs a full OCR on a job
folder in an S3-compatible bucket such as Cloudflare R2, so a host app can offload
the work:

```
s3://$OCR_BUCKET/ocr-jobs/<job_id>/
  game.json  roi.json  markers.json  app/data.json      inputs (game.json required)
  defaults/roi.json  defaults/roi_templates.json         optional fallback boxes
  params.json   {"fps": 0.5, "workers": 0, "decoder": "keyframe"}
→ derived/*  status.json  log.txt                        written back by the run
```

Trigger it with `gh workflow run ocr-job.yml -R asharon10/plyOCR -f job_id=<32 hex>`.
The only input is that opaque id; the OCR output goes to `log.txt` in the bucket,
not the public run log, and no artifacts are kept. Repo secrets:
`OCR_S3_ENDPOINT`, `OCR_BUCKET`, `OCR_S3_ACCESS_KEY_ID`, `OCR_S3_SECRET_ACCESS_KEY`.

## Default boxes

When a game has no `roi.json`, plyOCR uses a learned per-resolution template, then a
baseline ROI. Both ship in `plyocr/data/`; point at your own with
`PLYOCR_ROI_TEMPLATES` and `PLYOCR_DEFAULT_ROI`.

## Development

```bash
pip install -e ".[dev]"
ruff check . && pytest -q
```

The golden tests in `tests/fixtures/scoreboards/` are scoreboard-only crops from a
real broadcast; `expected.json` holds the values read by eye.

## License

MIT
