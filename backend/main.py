
import subprocess
from pathlib import Path
from uuid import uuid4
import shutil
import re
import tempfile
import cv2
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from pytubefix import YouTube

app = FastAPI(
    title="AI YouTube Clipper API",
    description="Backend untuk aplikasi pemotong video berbasis AI",
    version="0.3.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",   # localhost dan 127.0.0.1 dianggap origin berbeda
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

def correct_transcript(text: str) -> str:
    corrections = [
        # Koreksi nama Maudy
        (r"\bkamu\s+mau\s+di\b", "Kamu, Maudy"),
        (r"\bkamu\s+maudi\b", "Kamu, Maudy"),

        # Koreksi frasa bahasa Indonesia
        (r"\bbelajar\s+varang\b", "belajar bareng"),
        (r"\bdi\s+alam\s+hidup\b", "di dalam hidup"),

        # Koreksi nama Kak Roma
        (r"\bka?k\s*roma\b", "Kak Roma"),

        # Koreksi Bebersih
        (r"\bbeber[\s-]*bersih\b", "Bebersih"),
        (r"\bbeber\s*sih\b", "Bebersih"),
        (r"\bbebersi\b", "Bebersih"),

        # Koreksi sederas-derasnya
        (r"\bsederhas\s*-\s*derhasnya\b", "sederas-derasnya"),
        (r"\bsederas\s*-\s*derasnya\b", "sederas-derasnya"),
    ]

    for pattern, replacement in corrections:
        def replace_word(match, replacement=replacement):
            # Nama khusus harus selalu memakai kapitalisasi yang konsisten.
            if replacement == "Kamu, Maudy":
                return "Kamu, Maudy"

            if replacement == "Kak Roma":
                return "Kak Roma"

            original = match.group(0)

            if original.isupper():
                return replacement.upper()

            if original[:1].isupper():
                return replacement[0].upper() + replacement[1:]

            return replacement.lower()

        text = re.sub(
            pattern,
            replace_word,
            text,
            flags=re.IGNORECASE,
        )

    return text

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models" / "faster-whisper"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
FACE_CASCADE = cv2.CascadeClassifier(
    str(
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )
)

PROFILE_FACE_CASCADE = cv2.CascadeClassifier(
    str(
        Path(cv2.data.haarcascades)
        / "haarcascade_profileface.xml"
    )
)

if FACE_CASCADE.empty():
    print("WARNING: Model deteksi wajah depan gagal dimuat.")
else:
    print("INFO: Model deteksi wajah depan berhasil dimuat.")

if PROFILE_FACE_CASCADE.empty():
    print("WARNING: Model deteksi wajah profil gagal dimuat.")
else:
    print("INFO: Model deteksi wajah profil berhasil dimuat.")


def deduplicate_faces(faces, iou_threshold=0.40):
    """
    Menghapus deteksi wajah yang saling tumpang tindih.
    Setiap wajah berbentuk (x, y, width, height).
    """
    if len(faces) == 0:
        return []

    faces = [tuple(map(int, face)) for face in faces]
    faces = sorted(
        faces,
        key=lambda face: face[2] * face[3],
        reverse=True,
    )

    kept = []

    for face in faces:
        x1, y1, w1, h1 = face
        area1 = w1 * h1
        duplicate = False

        for other in kept:
            x2, y2, w2, h2 = other

            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            intersection = inter_w * inter_h

            area2 = w2 * h2
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0

            if iou >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(face)

    return kept


def detect_face_crop_positions(
    video_path: Path,
    start: float,
    end: float,
    output_width: int,
    output_height: int,
    sample_interval: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Mengambil posisi tengah wajah secara berkala.
    Hasil berupa (waktu_rel_clip, posisi_tengah_wajah).
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return []

    positions = []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or frame_width <= 0 or frame_height <= 0:
            return []

        duration = frame_height / max(fps, 1)
        start = max(0.0, start)
        end = min(end, cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

        if end <= start:
            return []

        target_ratio = output_width / output_height
        crop_width = min(
            frame_width,
            int(frame_height * target_ratio) // 2 * 2,
        )

        if crop_width <= 0:
            return []


        timestamp = start
        previous_center = None
        previous_y = None
        previous_size = None
        lost_frames = 0

        while timestamp <= end:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, frame = cap.read()

            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Deteksi wajah dari arah depan.
                front_faces = FACE_CASCADE.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=4,
                    minSize=(40, 40),
                )

                # Deteksi wajah dari arah profil.
                profile_faces = PROFILE_FACE_CASCADE.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=4,
                    minSize=(40, 40),
                )

                # deteksi wajah depan dan profil yang sangat bertumpang tindih akan digabungkan menjadi satu kandidat.
                faces = deduplicate_faces(
                    list(front_faces) + list(profile_faces),
                    iou_threshold=0.40,
                )

                candidates = []

                for x, y, w, h in faces:
                    center = float(x + w / 2)
                    center_y = float(y + h / 2)
                    size = float(max(w, h))
                    area = float(w * h)

                    # Beri bobot berdasarkan luas wajah.
                    area_score = area / max(
                        frame_width * frame_height,
                        1,
                    )

                    if previous_center is None:
                        # Pada awal clip, pilih wajah terbesar.
                        score = area_score
                    else:
                        dx = abs(center - previous_center)
                        dy = abs(center_y - previous_y)
                        size_change = abs(size - previous_size)

                        # Tolak kandidat yang terlalu jauh.
                        if (
                            dx > frame_width * 0.20
                            or dy > frame_height * 0.15
                        ):
                            continue

                        # Normalisasi jarak dan perubahan ukuran.
                        distance_penalty = (
                            dx / max(frame_width * 0.20, 1)
                            + dy / max(frame_height * 0.15, 1)
                        )

                        size_penalty = (
                            size_change / max(previous_size, 1)
                        )

                        # Posisi konsisten lebih penting daripada luas.
                        score = (
                            1.0
                            / (1.0 + 3.0 * distance_penalty
                               + 1.5 * size_penalty)
                        ) + area_score * 0.25

                    candidates.append(
                        (score, center, center_y, size)
                    )

                if candidates:
                    print(
                        f"[FACE DEBUG] t={timestamp:.2f}s, "
                        f"kandidat valid={len(candidates)}"
                    )

                    _, center, center_y, size = max(
                        candidates,
                        key=lambda item: item[0],
                    )

                    positions.append((timestamp - start, center))
                    previous_center = center
                    previous_y = center_y
                    previous_size = size
                    lost_frames = 0
                else:
                    lost_frames += 1

                    print(
                        f"[FACE DEBUG] t={timestamp:.2f}s, "
                        f"wajah tidak terdeteksi, "
                        f"durasi kehilangan={lost_frames * sample_interval:.1f}s"
                    )

                    # Pertahankan posisi crop terakhir selama deteksi hilang.
                    if previous_center is not None:
                        positions.append(
                            (timestamp - start, previous_center)
                        )

            timestamp += sample_interval

    finally:
        cap.release()

    return positions


def build_face_crop_filter(
    video_path: Path,
    start: float,
    end: float,
    output_width: int,
    output_height: int,
) -> str:
    """Membuat crop vertikal yang mengikuti wajah secara halus."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return (
            f"scale={output_width}:{output_height}:"
            "force_original_aspect_ratio=increase,"
            f"crop={output_width}:{output_height}"
        )

    try:
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if source_width <= 0 or source_height <= 0:
        return (
            f"scale={output_width}:{output_height}:"
            "force_original_aspect_ratio=increase,"
            f"crop={output_width}:{output_height}"
        )

    target_ratio = output_width / output_height
    source_ratio = source_width / source_height

    if source_ratio <= target_ratio:
        crop_width = source_width
        crop_height = int(source_width / target_ratio) // 2 * 2
        crop_x = 0
        crop_y = max(0, (source_height - crop_height) // 2)

        return (
            f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
            f"scale={output_width}:{output_height}"
        )

    crop_width = int(source_height * target_ratio) // 2 * 2
    crop_height = source_height

    positions = detect_face_crop_positions(
        video_path,
        start,
        end,
        output_width,
        output_height,
        sample_interval=0.5,
    )

    if not positions:
        crop_x = (source_width - crop_width) // 2
        return (
            f"crop={crop_width}:{crop_height}:{crop_x}:0,"
            f"scale={output_width}:{output_height}"
        )

    max_x = source_width - crop_width
    positions = [
        (t, max(0.0, min(float(x - crop_width / 2), float(max_x))))
        for t, x in positions
    ]

    # Filter crop menggunakan waktu relatif terhadap video hasil potongan.
    # Smoothing eksponensial pada sampel mengurangi perpindahan mendadak.
    smoothed = []

    alpha = 0.12   # crop merespons perubahan posisi lebih cepat.
    dead_zone = 28.0   # perpindahan kecil lebih mudah diikuti.
    max_speed = 160.0    # crop dapat bergerak lebih cepat ketika wajah berpindah.

    print(
        f"[CROP DEBUG] alpha={alpha}, "
        f"dead_zone={dead_zone}, max_speed={max_speed}"
    )

    for i, (t, x) in enumerate(positions):
        if i == 0:
            smooth_x = x
        else:
            previous_t, previous_x = smoothed[-1]
            dt = max(t - previous_t, 0.001)
            delta = x - previous_x

            # Abaikan perubahan posisi kecil agar crop tidak bergetar.
            if abs(delta) <= dead_zone:
                target_x = previous_x
            else:
                target_x = x

            # Haluskan perubahan posisi.
            movement = alpha * (target_x - previous_x)

            # Batasi kecepatan pergerakan crop.
            max_movement = max_speed * dt
            movement = max(
                -max_movement,
                min(movement, max_movement),
            )

            smooth_x = previous_x + movement

        smoothed.append((t, smooth_x))

    # Buat ekspresi FFmpeg piecewise-linear berdasarkan waktu.
    expression = f"{smoothed[-1][1]:.2f}"

    for i in range(len(smoothed) - 2, -1, -1):
        t0, x0 = smoothed[i]
        t1, x1 = smoothed[i + 1]

        if t1 <= t0:
            continue

        segment = (
            f"({x0:.2f}+({x1 - x0:.2f})*"
            f"(t-{t0:.3f})/{t1 - t0:.3f})"
        )
        expression = f"if(lt(t,{t1:.3f}),{segment},{expression})"

    # FFmpeg crop membutuhkan koordinat genap untuk kompatibilitas H.264.
    expression = f"trunc(({expression})/2)*2"

    return (
        f"crop={crop_width}:{crop_height}:"
        f"x='{expression}':y=0,"
        f"scale={output_width}:{output_height}"
    )

for folder in (UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR, TRANSCRIPT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE = 1024 * 1024 * 1024

# Model dimuat saat transkripsi pertama kali diminta.
whisper_model = None


def score_clip_candidate(text: str, duration: float) -> dict:
    """Menilai potensi klip berdasarkan hook, insight, cerita, dan durasi."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    words = normalized.split()

    if not words or duration <= 0:
        return {
            "score": 0,
            "reasons": ["Teks atau durasi tidak valid."],
        }

    score = 0
    reasons = []

    # 1. Hook: periksa pembuka, bukan seluruh teks.
    opening = " ".join(words[:15])
    strong_hooks = [
        "pernah nggak", "pernah gak", "bayangkan", "imagine",
        "what if", "kenapa", "mengapa", "rahasia", "ternyata",
        "gimana kalau", "bagaimana jika", "the reason",
        "here's why", "the truth",
    ]

    if any(phrase in opening for phrase in strong_hooks):
        score += 2
        reasons.append("Pembuka berpotensi menarik perhatian.")

    # 2. Insight atau gagasan yang bisa dibawa penonton.
    insight_phrases = [
        "yang penting", "pelajarannya", "artinya", "karena itu",
        "it means", "the key", "the lesson", "it's about",
        "bukan tentang", "yang harus kita", "kita perlu",
        "you should", "you need to", "the point is",
        "the important thing",
    ]

    if any(phrase in normalized for phrase in insight_phrases):
        score += 2
        reasons.append("Mengandung indikasi insight atau pelajaran.")

    # 3. Kontras atau perubahan perspektif.
    contrast_phrases = [
        "sebaliknya", "di sisi lain", "bukan tentang",
        "instead of", "rather than", "not about",
        "versus", "tapi lebih", "melainkan",
    ]

    if any(phrase in normalized for phrase in contrast_phrases):
        score += 1
        reasons.append("Mengandung kontras atau perubahan perspektif.")

    # 4. Cerita atau pengalaman personal.
    story_phrases = [
        "pengalaman gue", "pengalaman saya", "waktu itu",
        "ketika saya", "ketika gue", "i learned",
        "my experience", "i felt", "gue merasa",
        "saya merasa", "gue pernah", "saya pernah",
        "i realized", "I realized".lower(),
    ]

    if any(phrase in normalized for phrase in story_phrases):
        score += 1
        reasons.append("Mengandung indikasi pengalaman personal.")

    # 5. Penutup atau kesimpulan.
    conclusion_phrases = [
        "kesimpulannya", "jadi intinya", "pada akhirnya",
        "in conclusion", "the point is", "at the end",
        "that's why", "makanya", "so that's",
    ]

    if any(phrase in normalized for phrase in conclusion_phrases):
        score += 1
        reasons.append("Mengandung indikasi kesimpulan.")

    # 6. Penalti kecil jika pembuka terlihat bergantung pada konteks.
    weak_openings = [
        "tapi", "tetapi", "namun", "karena", "jadi",
        "nah", "dan", "oh iya", "but", "because",
        "and then", "so", "that's why",
    ]

    if any(opening.startswith(phrase) for phrase in weak_openings):
        score -= 1
        reasons.append("Pembuka mungkin membutuhkan konteks sebelumnya.")

    # 7. Durasi: durasi yang sesuai mendapat nilai tambahan.
    if 20 <= duration <= 60:
        score += 2
        reasons.append("Durasi sesuai target klip pendek.")
    elif 15 <= duration < 20 or 60 < duration <= 90:
        score += 1
        reasons.append("Durasi masih berpotensi cocok untuk klip pendek.")

    # 8. Isi yang cukup panjang berpeluang menyampaikan satu gagasan.
    if len(words) >= 25:
        score += 1
        reasons.append("Isi memiliki cukup teks untuk menyampaikan gagasan.")

    # Skala skor konsisten 0–10.
    score = max(0, min(10, score))

    return {
        "score": score,
        "reasons": reasons or ["Belum ditemukan indikator kuat."],
    }

@app.get("/videos/find-clips/{filename}")
def find_clips(
    filename: str,
    min_duration: float = 20,
    max_duration: float = 60,
    max_candidates: int = 10,
):
    # Cegah akses ke file di luar folder transkrip.
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Nama file tidak valid.",
        )

    if not 5 <= min_duration <= max_duration <= 180:
        raise HTTPException(
            status_code=400,
            detail="Durasi harus memenuhi 5 <= min_duration <= max_duration <= 180.",
        )

    if not 1 <= max_candidates <= 30:
        raise HTTPException(
            status_code=400,
            detail="max_candidates harus antara 1 dan 30.",
        )

    print("DEBUG find-clips filename:", repr(filename))
    print("DEBUG TRANSCRIPT_DIR:", TRANSCRIPT_DIR.resolve())
    print(
        "DEBUG transcript_path:",
        (TRANSCRIPT_DIR / f"{filename}.json").resolve(),
    )

    transcript_path = TRANSCRIPT_DIR / f"{filename}.json"

    if not transcript_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Transkrip belum ditemukan. Jalankan transkripsi penuh terlebih dahulu.",
        )

    try:
        with transcript_path.open("r", encoding="utf-8") as file:
            transcript_data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal membaca transkrip: {error}",
        ) from error

    segments = transcript_data.get("segments", [])
    duration = float(transcript_data.get("duration_seconds", 0))

    if not segments or duration <= 0:
        raise HTTPException(
            status_code=422,
            detail="Transkrip tidak memiliki segmen atau durasi valid.",
        )

    # Gabungkan segmen berurutan dan prioritaskan batas kalimat yang lebih alami.
    candidates = []

    for i in range(len(segments)):
        collected = []

        for j in range(i, len(segments)):
            collected.append(segments[j])

            start = float(collected[0]["start"])
            end = float(collected[-1]["end"])
            clip_duration = end - start

            if clip_duration > max_duration:
                break

            if clip_duration < min_duration:
                continue

            text = " ".join(
                str(segment.get("text", "")).strip()
                for segment in collected
            ).strip()

            if not text:
                continue

            analysis = score_clip_candidate(text, clip_duration)
            score = analysis["score"]
            reasons = list(analysis["reasons"])

            # Nilai batas awal dan akhir berdasarkan tanda baca transkrip.
            first_text = str(collected[0].get("text", "")).strip()
            last_text = str(collected[-1].get("text", "")).strip()

            sentence_endings = (".", "!", "?", "。", "！", "？")
            clause_endings = (",", ";", ":")

            # Awal kalimat lebih mandiri daripada sambungan yang jelas.
            if first_text and first_text[0].isupper():
                score += 1
                reasons.append("Awal kandidat tampak seperti awal kalimat.")

            # Akhir kalimat yang jelas membantu menjaga gagasan tetap utuh.
            if last_text.endswith(sentence_endings):
                score += 2
                reasons.append("Berakhir pada tanda akhir kalimat.")
            elif last_text.endswith(clause_endings):
                score += 1
                reasons.append("Berakhir pada jeda antarklausa.")
            else:
                score -= 1
                reasons.append("Akhir kandidat mungkin masih melanjutkan kalimat.")

            # Normalisasi skor supaya tetap berada pada rentang 0–10.
            score = max(0, min(10, score))

            candidate_segments = [
                {
                    "start": round(float(segment["start"]) - start, 3),
                    "end": round(float(segment["end"]) - start, 3),
                    "text": str(segment.get("text", "")).strip(),
                }
                for segment in collected
                if str(segment.get("text", "")).strip()
            ]

            candidates.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(clip_duration, 2),
                "title": text[:90] + ("..." if len(text) > 90 else ""),
                "transcript": text,
                "segments": candidate_segments,
                "score": score,
                "reasons": reasons,
            })

    # Urutkan skor tertinggi dahulu, lalu hilangkan kandidat yang terlalu tumpang tindih.
    candidates.sort(
        key=lambda item: (item["score"], -item["duration"]),
        reverse=True,
    )

    selected = []

    for candidate in candidates:
        overlaps = any(
            candidate["start"] < item["end"]
            and candidate["end"] > item["start"]
            for item in selected
        )

        if overlaps:
            continue

        selected.append(candidate)

        if len(selected) >= max_candidates:
            break

    return {
        "status": "success",
        "filename": transcript_data.get("filename", filename),
        "duration_seconds": duration,
        "candidate_count": len(selected),
        "candidates": selected,
    }

@app.get("/")
def home():
    return {
        "app": "AI YouTube Clipper",
        "status": "running",
        "message": "Backend berhasil dijalankan!",
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API berjalan normal"}


@app.get("/system/ffmpeg")
def check_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        return {"available": False, "message": "FFmpeg tidak ditemukan"}

    return {"available": True, "path": ffmpeg_path}


@app.post("/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Format harus MP4, MOV, MKV, atau WEBM.",
        )

    saved_name = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / saved_name
    total_size = 0

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Ukuran video maksimal 1 GB.",
                    )

                output.write(chunk)

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    return {
        "status": "success",
        "message": "Video berhasil diunggah.",
        "filename": saved_name,
        "original_name": original_name,
        "size_mb": round(total_size / (1024 * 1024), 2),
    }

@app.post("/videos/transcribe/{filename}")
def transcribe_video(filename: str):
    global whisper_model

    # Pastikan nama file tidak digunakan untuk mengakses folder lain.
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Nama file tidak valid.")

    video_path = UPLOAD_DIR / filename

    if video_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Format video tidak didukung.")

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video tidak ditemukan.")

    try:
        if whisper_model is None:
            whisper_model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8",
                download_root=str(MODEL_DIR),
            )

        segments, info = whisper_model.transcribe(
            str(video_path),
	    language="id",
            beam_size=5,
            initial_prompt=(
                "Transkrip percakapan dalam bahasa Indonesia, termasuk "
                "percakapan santai dan kata tidak baku. "
                "Jika pembicara mengucapkan frasa bahasa Inggris, "
                "pertahankan ejaan bahasa Inggris sesuai ucapan aslinya; "
                "jangan menerjemahkan atau mengubahnya menjadi kata Indonesia. "
                "Pertahankan nama orang dan istilah khusus dengan konsisten. "
                "Istilah penting: BEBERSIH, bebersih diri, "
                "bebersih emosional, bebersih lingkungan. "
                "Gunakan ejaan BEBERSIH ketika istilah tersebut memang diucapkan."
            ),
            vad_filter=True,
        )

        transcript_segments = []

        for segment in segments:
            corrected_text = correct_transcript(segment.text.strip())

            transcript_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": corrected_text,
            })


        full_text = " ".join(
            segment["text"] for segment in transcript_segments
        )

        transcript_data = {
            "status": "success",
            "filename": filename,
            "language": info.language,
            "duration_seconds": round(info.duration, 2),
            "transcript": full_text,
            "segments": transcript_segments,
        }

        transcript_path = TRANSCRIPT_DIR / f"{filename}.json"

        with transcript_path.open("w", encoding="utf-8") as file:
            json.dump(
                transcript_data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return transcript_data

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Transkripsi gagal: {str(error)}",
        ) from error


@app.post("/videos/create-clip/{filename}")
def create_clip(
    filename: str,
    start: float = Form(...),
    end: float = Form(...),
):
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Nama file tidak valid.",
        )

    input_path = UPLOAD_DIR / filename

    if not input_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video tidak ditemukan.",
        )

    if input_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Format video tidak didukung.",
        )

    if start < 0 or end <= start:
        raise HTTPException(
            status_code=400,
            detail="Waktu mulai dan selesai tidak valid.",
        )

    # Baca durasi video menggunakan FFprobe
    probe_command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]

    try:
        probe = subprocess.run(
            probe_command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if probe.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail="Gagal membaca durasi video.",
            )

        video_duration = float(probe.stdout.strip())

    except (subprocess.TimeoutExpired, ValueError):
        raise HTTPException(
            status_code=500,
            detail="Durasi video tidak dapat dibaca.",
        )

    if end > video_duration:
        raise HTTPException(
            status_code=400,
            detail=f"Waktu akhir melebihi durasi video ({video_duration:.2f} detik).",
        )

    output_filename = f"clip_{uuid4().hex}.mp4"
    output_path = OUTPUT_DIR / output_filename

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(end - start),
        "-vf",
        build_face_crop_filter(
            input_path,
            start,
            end,
            1080,
            1920,
        ),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

        if result.returncode != 0:
            output_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail="FFmpeg gagal membuat klip.",
            )

        return {
            "status": "success",
            "message": "Klip berhasil dibuat.",
            "filename": output_filename,
            "duration_seconds": round(end - start, 2),
            "resolution": "1080x1920",
            "download_url": f"/videos/download/{output_filename}",
        }

    except subprocess.TimeoutExpired:
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=504,
            detail="Pemrosesan video terlalu lama.",
        )


@app.get("/videos/download/{filename}")
def download_clip(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Nama file tidak valid.",
        )

    file_path = OUTPUT_DIR / filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="File klip tidak ditemukan.",
        )

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename,
    )


def format_srt_time(seconds: float) -> str:
    """Mengubah detik menjadi format timestamp subtitle SRT."""
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)

    return (
        f"{hours:02}:{minutes:02}:{secs:02},"
        f"{milliseconds:03}"
    )


@app.post("/videos/create-subtitled-clip/{filename}")
def create_subtitled_clip(
    filename: str,
    start: float = Form(...),
    end: float = Form(...),
):
    global whisper_model

    source_path = UPLOAD_DIR / Path(filename).name

    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="Video sumber tidak ditemukan.")

    if source_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Format video tidak didukung.")

    if start < 0 or end <= start:
        raise HTTPException(
            status_code=400,
            detail="Waktu akhir harus lebih besar dari waktu awal."
        )

    clip_id = uuid4().hex
    temp_clip = OUTPUT_DIR / f"temp_{clip_id}.mp4"
    srt_path = OUTPUT_DIR / f"subtitle_{clip_id}.srt"
    output_path = OUTPUT_DIR / f"subtitled_{clip_id}.mp4"

    try:
        # 1. Potong bagian video yang diminta.
        cut_command = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(source_path),
            "-t", str(end - start),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-b:a", "128k",
            str(temp_clip),
        ]

        cut_result = subprocess.run(
            cut_command,
            capture_output=True,
            text=True,
        )

        if cut_result.returncode != 0 or not temp_clip.is_file():
            raise RuntimeError(
                f"Gagal memotong video: {cut_result.stderr[-1500:]}"
            )

        # 2. Muat model jika belum dimuat.
        if whisper_model is None:
            whisper_model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8",
                download_root=str(MODEL_DIR),
            )

        # 3. Transkripsikan bagian video yang sudah dipotong.
        segments, _ = whisper_model.transcribe(
        str(temp_clip),
        language="id",
        beam_size=5,
        initial_prompt=(
            "Transkrip percakapan bahasa Indonesia yang dapat "
            "bercampur bahasa Inggris. Pertahankan frasa bahasa "
            "Inggris sesuai ucapan aslinya, termasuk: "
            "mindset, outcomes versus process, outcome focused, "
            "process focused, oh yes yes, sense of achievement."
        ),
        vad_filter=True,
        )

        # 4. Buat file subtitle SRT dengan timestamp.
        srt_entries = []

        for index, segment in enumerate(segments, start=1):
            text = correct_transcript(segment.text.strip())

            if not text:
                continue

            srt_entries.append(
                f"{index}\n"
                f"{format_srt_time(segment.start)} --> "
                f"{format_srt_time(segment.end)}\n"
                f"{text}\n"
            )

        if not srt_entries:
            raise RuntimeError(
                "Tidak ada ucapan yang terdeteksi untuk dijadikan subtitle."
            )

        srt_path.write_text(
            "\n".join(srt_entries),
            encoding="utf-8-sig",
        )

        # 5. Ubah video menjadi vertikal dan tanamkan subtitle.
        subtitle_filter_path = (
            str(srt_path.resolve())
            .replace("\\", "/")
            .replace(":", r"\:")
        )

        face_crop_filter = build_face_crop_filter(
            temp_path,
            start,
            end,
            width,
            height,
        )

        video_filter = (
            f"{face_crop_filter},"
            f"subtitles=filename='{subtitle_path}':"
            f"force_style='FontName=Arial,FontSize={font_size},"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BorderStyle=1,Outline=2,Shadow=1,"
            f"Alignment=2,MarginV={margin_v}'"
        )

        render_command = [
            "ffmpeg", "-y",
            "-i", str(temp_clip),
            "-vf", video_filter,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path),
        ]

        render_result = subprocess.run(
            render_command,
            capture_output=True,
            text=True,
        )

        if render_result.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                f"Gagal menambahkan subtitle: "
                f"{render_result.stderr[-2500:]}"
            )

        return {
            "status": "success",
            "message": "Klip dengan subtitle berhasil dibuat.",
            "filename": output_path.name,
            "duration_seconds": round(end - start, 2),
            "resolution": "1080x1920",
            "subtitle_file": srt_path.name,
            "download_url": f"/videos/download/{output_path.name}",
        }

    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:
        # File sementara tidak diperlukan setelah proses selesai.
        if temp_clip.exists():
            temp_clip.unlink()


@app.post("/videos/transcribe-segment/{filename}")
def transcribe_segment(
    filename: str,
    start: float = Form(...),
    end: float = Form(...),
):
    global whisper_model

    source_path = UPLOAD_DIR / Path(filename).name

    if not source_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video sumber tidak ditemukan."
        )

    if start < 0 or end <= start:
        raise HTTPException(
            status_code=400,
            detail="Waktu mulai dan selesai tidak valid."
        )

    temp_path = OUTPUT_DIR / f"preview_{uuid4().hex}.mp4"

    try:
        cut_command = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", str(source_path),
            "-t", str(end - start),
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "ultrafast",
            str(temp_path),
        ]

        cut_result = subprocess.run(
            cut_command,
            capture_output=True,
            text=True,
        )

        if cut_result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail="Gagal menyiapkan segmen video."
            )

        if whisper_model is None:
            whisper_model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8",
                download_root=str(MODEL_DIR),
            )

        segments, info = whisper_model.transcribe(
            str(temp_path),
            language="id",
            beam_size=5,
            initial_prompt=(
                "Transkrip percakapan santai dalam bahasa Indonesia "
                "yang dapat bercampur dengan bahasa Inggris. "
                "Pertahankan frasa bahasa Inggris sesuai ucapan asli, "
                "jangan menerjemahkannya. Pertahankan nama orang "
                "dan istilah khusus secara konsisten. "
                "Contoh istilah: mindset, turning point, "
                "outcome focused, process focused, "
                "sense of achievement."
            ),
            vad_filter=True,
            word_timestamps=True,
        )

        transcript_segments = []

        for segment in segments:
            words = []

            if segment.words:
                for word in segment.words:
                    words.append({
                        "start": round(float(word.start), 3),
                        "end": round(float(word.end), 3),
                        "word": word.word,
                        "probability": round(float(word.probability), 3),
                    })

            transcript_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": correct_transcript(segment.text.strip()),
                "words": words,
            })

        return {
            "status": "success",
            "message": "Transkrip berhasil dibuat.",
            "source_filename": filename,
            "clip_start": start,
            "clip_end": end,
            "language": info.language,
            "segments": transcript_segments,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal membuat transkrip: {str(exc)}",
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/videos/create-edited-subtitled-clip/{filename}")
def create_edited_subtitled_clip(
    filename: str,
    start: float = Form(...),
    end: float = Form(...),
    transcript_json: str = Form(...),
    resolution: int = Form(1080),
):
    import json

    source_path = UPLOAD_DIR / Path(filename).name

    if not source_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video sumber tidak ditemukan."
        )

    if start < 0 or end <= start:
        raise HTTPException(
            status_code=400,
            detail="Rentang waktu klip tidak valid."
        )

    if resolution not in (720, 1080, 1440, 2160):
        raise HTTPException(
            status_code=400,
            detail="Resolusi harus 720, 1080, 1440, atau 2160."
        )

    width = resolution
    height = resolution * 16 // 9
    font_size = 9
    margin_v = 45


    try:
        segments = json.loads(transcript_json)

        if not isinstance(segments, list) or not segments:
            raise ValueError("Transkrip harus berupa daftar segmen.")

        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("Format segmen tidak valid.")

            segment_start = float(segment["start"])
            segment_end = float(segment["end"])
            segment_text = segment["text"]

            # Batasi waktu subtitle agar tidak melewati durasi klip
            clip_duration = end - start
            segment_end = min(segment_end, clip_duration)

            print("DEBUG SEGMENT:", segment)
            print("DEBUG DURASI KLIP:", end - start)
            print("DEBUG WAKTU AKHIR SEGMENT:", segment_end)

            if (
                segment_start < 0
                or segment_end <= segment_start
                or segment_end > end - start
                or not isinstance(segment_text, str)
                or not segment_text.strip()
            ):
                raise ValueError(
                    "Waktu atau teks segmen tidak valid."
                )

    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Data transkrip tidak valid: {exc}",
        )

    clip_id = uuid4().hex
    srt_path = OUTPUT_DIR / f"edited_{clip_id}.srt"
    output_path = OUTPUT_DIR / f"edited_{clip_id}.mp4"
    temp_path = OUTPUT_DIR / f"temp_{clip_id}.mp4"

    try:
        cut_command = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(source_path),
            "-t", str(end - start),
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "ultrafast",
            str(temp_path),
        ]

        cut_result = subprocess.run(
            cut_command,
            capture_output=True,
            text=True,
        )

        if cut_result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail="Gagal memotong video sumber."
            )

        def to_srt_time(seconds):
            total_ms = round(seconds * 1000)
            hours, remainder = divmod(total_ms, 3_600_000)
            minutes, remainder = divmod(remainder, 60_000)
            secs, milliseconds = divmod(remainder, 1000)

            return (
                f"{hours:02}:{minutes:02}:"
                f"{secs:02},{milliseconds:03}"
            )

        # mengubah cara backend menulis subtitle ke file .srt,
        # supaya teks panjang bisa dipecah menjadi beberapa frasa
        # dengan waktu tampil masing-masing.
        with open(srt_path, "w", encoding="utf-8-sig") as srt_file:
            subtitle_index = 1

            for segment in segments:
                text = segment["text"].strip()
                segment_start = float(segment["start"])
                segment_end = float(segment["end"])

                if not text:
                    continue

                word_timestamps = segment.get("words", [])

                # Gunakan timestamp kata jika tersedia dan teksnya
                # masih cocok dengan hasil transkripsi asli.
                original_words = [
                    item.get("word", "").strip()
                    for item in word_timestamps
                ]

                edited_words = text.split()

                use_word_timestamps = (
                    len(word_timestamps) > 0
                    and len(original_words) == len(edited_words)
                    and all(
                        original.lower().strip(".,!?;:")
                        == edited.lower().strip(".,!?;:")
                        for original, edited in zip(
                            original_words, edited_words
                        )
                    )
                )

                if use_word_timestamps:
                    timed_words = [
                        {
                            "text": edited_words[i],
                            "start": max(
                                segment_start,
                                float(word_timestamps[i]["start"])
                            ),
                            "end": min(
                                segment_end,
                                float(word_timestamps[i]["end"])
                            ),
                        }
                        for i in range(len(edited_words))
                    ]
                else:
                    # Fallback untuk teks yang diedit atau timestamp
                    # yang tidak lagi cocok.
                    timed_words = []
                    duration = segment_end - segment_start
                    total_words = len(edited_words)

                    for i, word in enumerate(edited_words):
                        timed_words.append({
                            "text": word,
                            "start": (
                                segment_start
                                + duration * i / total_words
                            ),
                            "end": (
                                segment_start
                                + duration * (i + 1) / total_words
                            ),
                        })

                # Gabungkan maksimal 5 kata yang waktunya berdekatan.
                phrase = []
                phrase_start = None
                phrase_end = None
                previous_end = None

                def write_phrase():
                    nonlocal subtitle_index

                    if not phrase:
                        return

                    srt_file.write(
                        f"{subtitle_index}\n"
                        f"{to_srt_time(phrase_start)} --> "
                        f"{to_srt_time(phrase_end)}\n"
                        f"{' '.join(phrase)}\n\n"
                    )
                    subtitle_index += 1

                for word in timed_words:
                    # Pisahkan subtitle jika jeda ucapan cukup panjang.
                    has_long_pause = (
                        previous_end is not None
                        and word["start"] - previous_end > 0.45
                    )

                    if len(phrase) >= 5 or has_long_pause:
                        write_phrase()
                        phrase = []
                        phrase_start = None

                    if not phrase:
                        phrase_start = word["start"]

                    phrase.append(word["text"])
                    phrase_end = max(word["end"], phrase_start + 0.05)
                    previous_end = word["end"]

                write_phrase()

        subtitle_path = str(srt_path.resolve())
        subtitle_path = subtitle_path.replace("\\", "/")
        subtitle_path = subtitle_path.replace(":", r"\:")

        face_crop_filter = build_face_crop_filter(
            Path(source_path),
            start,
            end,
            width,
            height,
        )

        video_filter = (
            f"{face_crop_filter},"
            f"subtitles=filename='{subtitle_path}':"
            f"force_style='FontName=Arial,FontSize={font_size},"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BorderStyle=1,Outline=2,Shadow=1,"
            f"Alignment=2,MarginV={margin_v}'"
        )

        render_command = [
            "ffmpeg", "-y",
            "-i", str(temp_path),
            "-vf", video_filter,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path),
        ]

        render_result = subprocess.run(
            render_command,
            capture_output=True,
            text=True,
        )

        if render_result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail="Gagal membuat video dengan subtitle."
            )

        return {
            "status": "success",
            "message": "Klip dengan transkrip edit berhasil dibuat.",
            "filename": output_path.name,
            "subtitle_file": srt_path.name,
            "duration_seconds": round(end - start, 2),
            "resolution": f"{width}x{height}",
            "download_url": f"/videos/download/{output_path.name}",
        }

    finally:
        for temporary_file in (temp_path,):
            if temporary_file.exists():
                temporary_file.unlink()

from fastapi.staticfiles import StaticFiles

# untuk menerima URL YouTube,
# mengunduh video dan audio menggunakan pytubefix,
# lalu menggabungkannya menggunakan FFmpeg.
# Setelah itu, frontend akan kita sambungkan ke endpoint tersebut.
@app.post("/videos/import-youtube")
def import_youtube(
    url: str = Form(...),
    resolution: int = Form(720),
):
    from urllib.parse import urlparse

    if resolution not in (720, 1080, 1440, 2160):
        raise HTTPException(
            status_code=400,
            detail="Resolusi harus 720, 1080, 1440, atau 2160.",
        )

    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()

    if hostname not in (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtube-nocookie.com",
    ):
        raise HTTPException(
            status_code=400,
            detail="Masukkan URL video YouTube yang valid.",
        )

    job_dir = None

    try:
        yt = YouTube(url.strip())

        # Pilih video MP4 tanpa audio sesuai resolusi yang diminta.
        video_streams = [
            stream for stream in yt.streams
            if stream.type == "video"
            and stream.mime_type == "video/mp4"
            and stream.resolution
        ]

        if not video_streams:
            raise RuntimeError("Stream video MP4 tidak tersedia.")

        suitable = [
            stream for stream in video_streams
            if int(stream.resolution[:-1]) <= resolution
        ]

        if suitable:
            video_stream = max(
                suitable,
                key=lambda stream: int(stream.resolution[:-1]),
            )
        else:
            video_stream = min(
                video_streams,
                key=lambda stream: int(stream.resolution[:-1]),
            )

        # Pilih audio MP4 dengan bitrate tertinggi.
        audio_streams = [
            stream for stream in yt.streams
            if stream.type == "audio"
            and stream.mime_type == "audio/mp4"
        ]

        if not audio_streams:
            raise RuntimeError("Stream audio MP4 tidak tersedia.")

        audio_stream = max(
            audio_streams,
            key=lambda stream: int(
                (stream.abr or "0kbps").replace("kbps", "")
            ),
        )

        job_dir = tempfile.mkdtemp(prefix="youtube_import_")
        temp_dir = Path(job_dir)

        video_path = Path(
            video_stream.download(
                output_path=str(temp_dir),
                filename="video.mp4",
            )
        )

        audio_path = Path(
            audio_stream.download(
                output_path=str(temp_dir),
                filename="audio.mp4",
            )
        )

        output_filename = f"youtube_{uuid4().hex}.mp4"
        output_path = UPLOAD_DIR / output_filename

        command = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=7200,
            check=False,
        )

        if result.returncode != 0 or not output_path.is_file():
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Gagal menggabungkan video dan audio: "
                f"{result.stderr[-1500:]}"
            )

        return {
            "status": "success",
            "message": "Video YouTube berhasil diimpor.",
            "title": yt.title,
            "filename": output_filename,
            "resolution_downloaded": video_stream.resolution,
            "duration_seconds": yt.length,
            "download_url": f"/videos/download/{output_filename}",
            "next_step": f"/videos/transcribe/{output_filename}",
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengimpor video YouTube: {str(exc)}",
        ) from exc

    finally:
        if job_dir:
            shutil.rmtree(job_dir, ignore_errors=True)


FRONTEND_DIR = BASE_DIR / "frontend"   # untuk mount file frontend index.html
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
