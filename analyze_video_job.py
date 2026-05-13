import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from urllib import request
from urllib.parse import urlparse

import cv2
from ultralytics import YOLO

import api_client


def emit(event):
    print(json.dumps(event, ensure_ascii=False), flush=True)


def prepare_source(source, output_dir, job_id):
    parsed = urlparse(source)
    youtube_hosts = ("youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com")

    if parsed.netloc.lower() in youtube_hosts:
        try:
            import yt_dlp
        except ImportError as error:
            raise RuntimeError("YouTube linki icin yt-dlp gerekli. Komut: python -m pip install yt-dlp") from error

        target_template = str(Path(output_dir) / f"job_{job_id}_source.%(ext)s")
        options = {
            "format": "best[ext=mp4]/best",
            "outtmpl": target_template,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(source, download=True)
            downloaded = downloader.prepare_filename(info)
            if not downloaded.lower().endswith(".mp4"):
                mp4_candidate = str(Path(downloaded).with_suffix(".mp4"))
                if Path(mp4_candidate).exists():
                    downloaded = mp4_candidate
            return downloaded

    if parsed.scheme in ("http", "https"):
        suffix = Path(parsed.path).suffix or ".mp4"
        target = Path(output_dir) / f"job_{job_id}_source{suffix}"
        request.urlretrieve(source, target)
        return str(target)

    return source


def class_exists(classes, class_id):
    return any(int(item) == class_id for item in classes)


def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        color = detection["color"]
        label = detection["label"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def finalize_video(raw_video_path, source_path, final_video_path):
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_video_path),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(final_video_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0 and final_video_path.exists():
        raw_video_path.unlink(missing_ok=True)
        return final_video_path

    silent_command = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_video_path),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_video_path),
    ]
    silent_result = subprocess.run(silent_command, capture_output=True, text=True)
    if silent_result.returncode == 0 and final_video_path.exists():
        raw_video_path.unlink(missing_ok=True)
        return final_video_path

    raise RuntimeError((result.stderr or silent_result.stderr or "Video donusturme basarisiz oldu").strip()[-1200:])


def analyze(args):
    api_client.SAFETY_API_URL = args.api_url

    output_dir = Path(args.output_dir)
    snapshot_dir = Path(args.snapshot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    source = prepare_source(args.source, output_dir, args.job_id)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError("Video kaynagi acilamadi. Dogrudan mp4 linki veya yerel video dosyasi kullanin.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    raw_video_path = output_dir / f"job_{args.job_id}_annotated_raw.mp4"
    output_video_path = output_dir / f"job_{args.job_id}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(raw_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raw_video_path = output_dir / f"job_{args.job_id}_annotated_raw.avi"
        writer = cv2.VideoWriter(
            str(raw_video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )

    if not writer.isOpened():
        raise RuntimeError("Cikti videosu icin OpenCV VideoWriter acilamadi.")

    insan_modeli = YOLO(os.getenv("INSAN_MODEL_PATH", "yolov8n.pt"))
    ekipman_modeli = YOLO(os.getenv("EKIPMAN_MODEL_PATH", "santiye_modeli.pt"))

    processed_frames = 0
    violation_count = 0
    preview_frame_path = None
    last_violation_frame = -args.violation_cooldown_frames
    pause_file = Path(args.pause_file) if args.pause_file else None
    last_detections = []
    last_detection_frame = 0

    while True:
        paused_event_sent = False
        while pause_file and pause_file.exists():
            if not paused_event_sent:
                emit(
                    {
                        "event": "paused",
                        "processedFrames": processed_frames,
                        "totalFrames": total_frames,
                        "previewFramePath": str(preview_frame_path) if preview_frame_path else None,
                    }
                )
                paused_event_sent = True
            time.sleep(0.5)

        ok, frame = capture.read()
        if not ok:
            break

        processed_frames += 1
        annotated = frame.copy()
        should_analyze = processed_frames % args.frame_step == 0
        pending_violations = []

        if should_analyze:
            insan_sonuclar = insan_modeli.predict(source=frame, classes=[0], verbose=False)
            current_detections = []

            for sonuc in insan_sonuclar:
                for kutu in sonuc.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, kutu)
                    insan_bolgesi = frame[y1:y2, x1:x2]

                    if insan_bolgesi.shape[0] < 20 or insan_bolgesi.shape[1] < 20:
                        continue

                    ekipman_sonuclar = ekipman_modeli.predict(source=insan_bolgesi, verbose=False)
                    baret_var_mi = False
                    yelek_var_mi = False
                    confidence = None

                    for ekipman_sonuc in ekipman_sonuclar:
                        bulunan_siniflar = ekipman_sonuc.boxes.cls
                        baret_var_mi = baret_var_mi or class_exists(bulunan_siniflar, 0)
                        yelek_var_mi = yelek_var_mi or class_exists(bulunan_siniflar, 1)
                        if len(ekipman_sonuc.boxes.conf) > 0:
                            confidence = float(max(ekipman_sonuc.boxes.conf).item())

                    should_save_violation = False
                    if baret_var_mi and yelek_var_mi:
                        renk = (0, 180, 80)
                        etiket = "Guvenli"
                        eksikler = []
                    else:
                        renk = (0, 0, 255)
                        eksikler = []
                        if not baret_var_mi:
                            eksikler.append("Baret Yok")
                        if not yelek_var_mi:
                            eksikler.append("Yelek Yok")
                        etiket = "IHLAL: " + ", ".join(eksikler)

                        should_save_violation = processed_frames - last_violation_frame >= args.violation_cooldown_frames

                    current_detections.append(
                        {
                            "box": (x1, y1, x2, y2),
                            "color": renk,
                            "label": etiket,
                        }
                    )

                    if should_save_violation:
                        pending_violations.append(
                            {
                                "missing": ", ".join(eksikler),
                                "helmet": baret_var_mi,
                                "vest": yelek_var_mi,
                                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "confidence": confidence,
                            }
                        )
                        violation_count += 1
                        last_violation_frame = processed_frames

            if current_detections:
                last_detections = current_detections
                last_detection_frame = processed_frames
            elif processed_frames - last_detection_frame > args.annotation_hold_frames:
                last_detections = []

        if last_detections and processed_frames - last_detection_frame <= args.annotation_hold_frames:
            draw_detections(annotated, last_detections)

        for violation in pending_violations:
            snapshot_path = snapshot_dir / f"job_{args.job_id}_ihlal_{processed_frames}.jpg"
            cv2.imwrite(str(snapshot_path), annotated)
            api_client.ihlal_gonder(
                ihlal_turu=violation["missing"],
                fotograf_yolu=str(snapshot_path),
                baret_var_mi=violation["helmet"],
                yelek_var_mi=violation["vest"],
                bbox=violation["bbox"],
                kaynak=f"video-job:{args.job_id}",
                video_job_id=args.job_id,
                confidence=violation["confidence"],
            )

        if processed_frames % args.preview_interval == 0 or preview_frame_path is None:
            preview_frame_path = output_dir / f"job_{args.job_id}_preview.jpg"
            cv2.imwrite(str(preview_frame_path), annotated)
            emit(
                {
                    "event": "progress",
                    "processedFrames": processed_frames,
                    "totalFrames": total_frames,
                    "previewFramePath": str(preview_frame_path),
                }
            )

        writer.write(annotated)

        if args.max_frames and processed_frames >= args.max_frames:
            break

    capture.release()
    writer.release()
    output_video_path = finalize_video(raw_video_path, source, output_video_path)

    emit(
        {
            "event": "completed",
            "processedFrames": processed_frames,
            "totalFrames": total_frames,
            "violationCount": violation_count,
            "outputVideoPath": str(output_video_path),
            "previewFramePath": str(preview_frame_path) if preview_frame_path else None,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--api-url", default=os.getenv("SAFETY_API_URL", "http://localhost:4000/api/violations"))
    parser.add_argument("--output-dir", default="backend/uploads/videos")
    parser.add_argument("--snapshot-dir", default="ihlaller")
    parser.add_argument("--frame-step", type=int, default=int(os.getenv("ANALYZE_FRAME_STEP", "5")))
    parser.add_argument("--preview-interval", type=int, default=20)
    parser.add_argument("--violation-cooldown-frames", type=int, default=35)
    parser.add_argument("--annotation-hold-frames", type=int, default=int(os.getenv("ANNOTATION_HOLD_FRAMES", "12")))
    parser.add_argument("--max-frames", type=int, default=int(os.getenv("ANALYZE_MAX_FRAMES", "0")))
    parser.add_argument("--pause-file", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
