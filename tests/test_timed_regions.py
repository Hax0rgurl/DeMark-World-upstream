import hashlib
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from src.demark_world.timed_cleaning import clean_timed_frame, clean_timed_video
from src.demark_world.timed_regions import compile_regions


class Cleaner:
    def __init__(self):
        self.calls = 0

    def clean(self, image, mask):
        self.calls += 1
        # Deliberately alter everything; compositing must protect the rest.
        return np.full_like(image, [200, 80, 20])


def test_masks_apply_only_inside_their_independent_frame_ranges():
    frame = np.full((80, 120, 3), [10, 40, 70], dtype=np.uint8)
    regions = compile_regions(
        [
            {"bbox": [0.1, 0.1, 0.3, 0.3], "start_frame": 2, "end_frame": 4},
            {"bbox": [0.7, 0.6, 0.9, 0.9], "start_frame": 5, "end_frame": 7},
        ],
        120,
        80,
        8,
    )
    cleaner = Cleaner()
    for index in range(8):
        output = clean_timed_frame(frame, index, regions, cleaner)
        if index in [0, 1, 4, 7]:
            assert np.array_equal(output, frame)
        else:
            active = regions[0] if index in [2, 3] else regions[1]
            x1, y1, x2, y2 = active.bbox
            expected = frame.copy()
            expected[y1:y2, x1:x2] = [20, 80, 200]
            assert np.array_equal(output, expected)
    assert cleaner.calls == 4


@pytest.mark.parametrize("start,end", [(-1, 10), (5, 5), (10, 9), (0, 31), (1.5, 10), (True, 10)])
def test_invalid_times_are_rejected(start, end):
    with pytest.raises(ValueError):
        compile_regions(
            [{"bbox": [0.1, 0.1, 0.2, 0.2], "start_frame": start, "end_frame": end}], 100, 100, 30
        )


@pytest.mark.parametrize("bbox", [[0, 0, 2, 1], [0, 0, 0, 1], [0, 0, float("nan"), 1]])
def test_invalid_boxes_are_rejected(bbox):
    with pytest.raises(ValueError):
        compile_regions([bbox], 100, 100, 30)


def test_legacy_boxes_cover_the_full_video_and_unchecked_suggestions_are_rejected():
    region = compile_regions([[0.1, 0.1, 0.2, 0.2]], 100, 100, 30)[0]
    assert region.active(0) and region.active(29) and not region.active(30)
    with pytest.raises(ValueError, match="identity"):
        compile_regions([{"bbox": [0.1, 0.1, 0.2, 0.2], "source": "suggested"}], 100, 100, 30)


@pytest.fixture
def video(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for the media integration checks")
    path = tmp_path / "input.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=120x80:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_streamed_output_preserves_audio_frame_count_and_source(video, tmp_path):
    original = hashlib.sha256(video.read_bytes()).hexdigest()
    output = tmp_path / "cleaned.mp4"
    progress = []
    cleaner = Cleaner()
    from src.demark_world.core import DeMarkWorld

    engine = DeMarkWorld(automatic_detection=False, cleaner=cleaner)
    engine.run(
        video,
        output,
        watermark_regions=[{"bbox": [0.1, 0.1, 0.3, 0.3], "start_frame": 2, "end_frame": 4}],
        progress_callback=progress.append,
    )
    capture = cv2.VideoCapture(str(output))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 20
    capture.release()
    assert cleaner.calls == 2 and progress[-1] == 100
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(output),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    assert hashlib.sha256(video.read_bytes()).hexdigest() == original
    assert not list(tmp_path.glob(".demark-timed-*"))


def test_failure_never_publishes_output_or_leaves_encoder_files(video, tmp_path):
    class FailingCleaner:
        def clean(self, image, mask):
            raise RuntimeError("inference failed")

    output = tmp_path / "failed.mp4"
    with pytest.raises(RuntimeError, match="inference failed"):
        clean_timed_video(video, output, [[0.1, 0.1, 0.3, 0.3]], FailingCleaner())
    assert video.exists() and not output.exists()
    assert not list(tmp_path.glob(".demark-timed-*"))


def test_existing_output_is_preserved(video, tmp_path):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep this output")
    with pytest.raises(FileExistsError):
        clean_timed_video(video, output, [[0.1, 0.1, 0.3, 0.3]], Cleaner())
    assert output.read_bytes() == b"keep this output"


def test_engine_does_not_load_detector_or_optional_backends_for_injected_cleaner(monkeypatch):
    from src.demark_world import core

    def forbidden(*args, **kwargs):
        raise AssertionError("Detector weights must not load")

    monkeypatch.setattr(core, "DeMarkWorldDetector", forbidden)
    engine = core.DeMarkWorld(automatic_detection=False, cleaner=Cleaner())
    assert engine.detector is None
    with pytest.raises(ValueError, match="watermark_regions"):
        engine.run("unused.mp4", "unused-output.mp4")
