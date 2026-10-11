# Opt-in timed watermark regions

The default detector workflow remains available. When a watermark appears intermittently, or the detector selects captions, supply reviewed boxes with individual start/stop frames instead. Only those pixels during those intervals are passed to LaMa. No detector weights are loaded when `automatic_detection=False`.

```python
from pathlib import Path
from src.demark_world.core import DeMarkWorld

engine = DeMarkWorld(automatic_detection=False)
engine.run(
    Path("input.mp4"),
    Path("cleaned.mp4"),
    watermark_regions=[
        {"bbox": [0.78, 0.04, 0.97, 0.13], "start_frame": 300, "end_frame": 360},
        {"bbox": [0.78, 0.04, 0.97, 0.13], "start_frame": 720, "end_frame": 780},
    ],
)
```

Coordinates are normalized `[left, top, right, bottom]`. Frame numbers are zero-based: start inclusive, end exclusive. A bare four-coordinate box applies throughout the video. Invalid boxes, invalid intervals, and automatic suggestions lacking an identity-check marker are rejected before encoding. This API does not itself detect or verify branding; callers remain responsible for supplying reviewed regions.

This first contribution supports LaMa and MP4 output. It streams frames, preserves pixels outside active masks before encoding, muxes available audio, fully decodes the result, and publishes only on success. Existing outputs and input files are preserved. Video encoding is lossy, so pixel preservation before encoding does not mean a bit-identical output. Constant-frame-rate inputs are the tested case; variable-frame-rate timing needs further work. CUDA/E2FGVI timed processing and UI controls are outside this PR.

## Where this came from

[MAC WATERMARK CLEANER](https://github.com/Hax0rgurl/WatermarkCleanerMac) builds on DeMark-World. Its Mac app adds named-brand Apple Vision OCR, optional local MLX vision confirmation, visual tracking, audited boundaries, an editable timeline, playback/scrubbing, and unavailable-server fallback. Unrestricted white-text detection had erased unrelated captions; named checks and timed masks give users a narrower, inspectable scope.

The Mac scan of a 607.9-second film found 59 checked regions, including 43 repeating corner InVideo appearances at a median interval of 14.033 seconds. A full Mac render completed and decoded with audio. A short proof was visually checked for watermark removal and preservation of two captions. These results describe the companion app, not automatic detection added by this PR, and do not establish perfect removal from every film.

This focused API port avoids replacing upstream's interfaces, dependency lock, detector, or native packaging. See the [Mac changelog](https://github.com/Hax0rgurl/WatermarkCleanerMac/blob/main/CHANGELOG.md) and [verification](https://github.com/Hax0rgurl/WatermarkCleanerMac/blob/main/docs/VERIFICATION.md).

## Tests

```bash
uv run pytest tests/test_timed_regions.py tests/test_placeholder.py
```

The media integration tests require FFmpeg on PATH and use synthetic fixtures. They test timed masks, RGB/BGR compositing, audio, output frame count, source preservation, failed inference cleanup, and existing-output protection. They inject a cleaner, so no model downloads are required. Actual LaMa inference is verified separately in the Mac app; this upstream port does not claim a full CUDA test run.
