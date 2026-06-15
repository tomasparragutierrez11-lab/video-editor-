#!/usr/bin/env python3
"""
Dental Podcast Viral Edit Pipeline
Formats: 9:16 vertical, 1080x1920
"""

import os, sys, json, re, math, subprocess, textwrap, requests, time
from pathlib import Path
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
PEXELS_API_KEY  = os.environ["PEXELS_API_KEY"]
RUNWAY_API_KEY  = os.environ.get("RUNWAY_API_KEY", "")
OPUS_API_KEY    = os.environ.get("OPUS_API_KEY", "")

INPUT_VIDEO     = Path("/home/user/video-editor-/workspace/input/IMG_3745.mov")
OUTPUT_DIR      = Path("/home/user/video-editor-/workspace/output")
BROLL_DIR       = Path("/home/user/video-editor-/workspace/broll")
SUBTITLE_FILE   = Path("/home/user/video-editor-/workspace/subtitles/subs.srt")
TRANSCRIPT_FILE = Path("/home/user/video-editor-/workspace/subtitles/transcript.json")

OUTPUT_WIDTH    = 1080
OUTPUT_HEIGHT   = 1920
FPS             = 30

client = OpenAI(api_key=OPENAI_API_KEY)


# ── Helpers ───────────────────────────────────────────────────────────────────
def run(cmd, **kw):
    print(f"  $ {cmd[:100]}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
    return result


def ts(seconds):
    """seconds -> HH:MM:SS,mmm  (SRT format)"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── Step 1: Probe input video ─────────────────────────────────────────────────
def probe_video():
    print("\n[1/8] Probing input video...")
    r = run(f'ffprobe -v quiet -print_format json -show_streams "{INPUT_VIDEO}"')
    info = json.loads(r.stdout)
    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    duration = float(video_stream.get("duration", 0))
    width    = int(video_stream["width"])
    height   = int(video_stream["height"])
    print(f"  Duration: {duration:.1f}s  |  Size: {width}x{height}")
    return {"duration": duration, "width": width, "height": height}


# ── Step 2: Extract & clean audio ────────────────────────────────────────────
def extract_audio():
    print("\n[2/8] Extracting and cleaning audio...")
    raw_audio  = Path("/home/user/video-editor-/workspace/audio/raw.wav")
    clean_audio = Path("/home/user/video-editor-/workspace/audio/clean.wav")

    # Extract
    run(f'ffmpeg -y -i "{INPUT_VIDEO}" -vn -ar 16000 -ac 1 "{raw_audio}"')

    # Clean: highpass, lowpass, afftdn noise reduction, dynamic normalization
    run(
        f'ffmpeg -y -i "{raw_audio}" '
        f'-af "highpass=f=80, lowpass=f=8000, afftdn=nf=-25, '
        f'loudnorm=I=-16:TP=-1.5:LRA=11" '
        f'"{clean_audio}"'
    )
    print("  Audio cleaned and normalized.")
    return clean_audio


# ── Step 3: Transcribe with Whisper ──────────────────────────────────────────
def transcribe(clean_audio):
    print("\n[3/8] Transcribing with Whisper...")
    if TRANSCRIPT_FILE.exists():
        print("  Using cached transcript.")
        return json.loads(TRANSCRIPT_FILE.read_text())

    with open(clean_audio, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="es",
        )
    data = result.model_dump()
    TRANSCRIPT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"  Transcribed {len(data.get('words', []))} words.")
    return data


# ── Step 4: Find hook ─────────────────────────────────────────────────────────
def find_hook(transcript_data):
    print("\n[4/8] Identifying hook phrase with GPT-4o...")
    full_text = transcript_data.get("text", "")

    prompt = f"""Eres un editor de video viral experto en TikTok e Instagram Reels.

Analiza esta transcripción de un podcast de periodoncia/salud oral grabado en un café:

---
{full_text[:4000]}
---

Identifica LA ÚNICA frase más impactante, sorprendente o que genere más curiosidad.
Debe ser una frase corta (5-15 palabras) que haga que alguien quiera ver el video completo.
Responde SOLO con la frase exacta, tal como aparece en la transcripción, sin comillas ni explicaciones."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.3,
    )
    hook_phrase = response.choices[0].message.content.strip()
    print(f"  Hook: '{hook_phrase}'")

    # Find timestamp of hook in words
    words = transcript_data.get("words", [])
    hook_words = hook_phrase.lower().split()
    hook_start = 0.0
    hook_end   = 3.0

    for i, w in enumerate(words):
        word_clean = re.sub(r"[^\w]", "", w.get("word", "").lower())
        if word_clean == re.sub(r"[^\w]", "", hook_words[0]):
            # Check sequence
            match = True
            for j, hw in enumerate(hook_words[:5]):
                if i + j >= len(words):
                    match = False; break
                wc = re.sub(r"[^\w]", "", words[i+j].get("word","").lower())
                if wc != re.sub(r"[^\w]", "", hw):
                    match = False; break
            if match:
                hook_start = float(w.get("start", 0))
                end_word = words[min(i + len(hook_words) - 1, len(words)-1)]
                hook_end = float(end_word.get("end", hook_start + 3))
                break

    return {"phrase": hook_phrase, "start": hook_start, "end": hook_end}


# ── Step 5: Generate SRT subtitles ───────────────────────────────────────────
IMPORTANT_KEYWORDS = [
    "encía","encías","bacteria","bacterias","periodontitis","periodoncia",
    "gingivitis","inflamación","sangrado","hueso","implante","implantes",
    "enfermedad","prevención","limpieza","importante","clave","crítico",
    "jamás","nunca","siempre","primer","mejor","peor","riesgo","grave",
    "cura","cuidado","salud","oral","dental","dientes","boca",
]

def generate_srt(transcript_data):
    print("\n[5/8] Generating SRT subtitles...")
    words = transcript_data.get("words", [])
    if not words:
        print("  No word-level data, using segments.")
        segments = transcript_data.get("segments", [])
        srt = ""
        for i, seg in enumerate(segments, 1):
            srt += f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{seg['text'].strip()}\n\n"
        SUBTITLE_FILE.write_text(srt)
        return srt

    # Group into subtitle chunks (~5-7 words or natural pause)
    chunks = []
    chunk_words = []
    chunk_start = None

    for w in words:
        word = w.get("word", "").strip()
        start = float(w.get("start", 0))
        end   = float(w.get("end", 0))

        if chunk_start is None:
            chunk_start = start

        chunk_words.append({"word": word, "start": start, "end": end})

        # Break chunk at punctuation or every 6 words
        has_punct = bool(re.search(r"[.,;:!?]", word))
        if len(chunk_words) >= 6 or has_punct:
            chunks.append({
                "words": chunk_words.copy(),
                "start": chunk_start,
                "end": end,
            })
            chunk_words = []
            chunk_start = None

    if chunk_words:
        chunks.append({
            "words": chunk_words,
            "start": chunk_start or 0,
            "end": chunk_words[-1]["end"],
        })

    # Build SRT with word highlighting via markup
    srt_lines = []
    for i, chunk in enumerate(chunks, 1):
        text_parts = []
        for w in chunk["words"]:
            clean = re.sub(r"[^\w]", "", w["word"].lower())
            if any(clean == kw or clean.startswith(kw) for kw in IMPORTANT_KEYWORDS):
                text_parts.append(w["word"].upper())
            else:
                text_parts.append(w["word"])
        line = " ".join(text_parts).strip()
        srt_lines.append(f"{i}\n{ts(chunk['start'])} --> {ts(chunk['end'])}\n{line}\n")

    srt_content = "\n".join(srt_lines)
    SUBTITLE_FILE.write_text(srt_content)
    print(f"  Generated {len(chunks)} subtitle chunks.")
    return srt_content


# ── Step 6: Fetch B-roll from Pexels ─────────────────────────────────────────
BROLL_QUERIES = [
    "dental clinic teeth", "gum health oral", "dentist examination",
    "tooth brushing", "dental procedure", "healthy smile teeth",
]

def fetch_broll():
    print("\n[6/8] Fetching B-roll from Pexels...")
    BROLL_DIR.mkdir(exist_ok=True)
    downloaded = []
    headers = {"Authorization": PEXELS_API_KEY}

    for q in BROLL_QUERIES[:3]:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params={"query": q, "per_page": 2, "min_width": 720, "min_duration": 3, "max_duration": 10},
            timeout=15,
        )
        if r.status_code != 200:
            print(f"  Pexels error for '{q}': {r.status_code}")
            continue

        for video in r.json().get("videos", []):
            # Pick HD file
            files = sorted(video["video_files"], key=lambda x: x.get("width", 0), reverse=True)
            hd = next((f for f in files if f.get("width", 0) >= 720), None)
            if not hd:
                continue
            dest = BROLL_DIR / f"broll_{video['id']}.mp4"
            if dest.exists():
                downloaded.append(dest)
                continue
            try:
                dl = requests.get(hd["link"], timeout=30, stream=True)
                with open(dest, "wb") as fp:
                    for chunk in dl.iter_content(8192):
                        fp.write(chunk)
                downloaded.append(dest)
                print(f"  Downloaded: {dest.name}")
            except Exception as e:
                print(f"  Download failed: {e}")

    print(f"  {len(downloaded)} B-roll clips ready.")
    return downloaded


# ── Step 7: Build final video ─────────────────────────────────────────────────
def build_video(video_info, hook_info, broll_clips):
    print("\n[7/8] Building final video...")
    duration  = video_info["duration"]
    src_w     = video_info["width"]
    src_h     = video_info["height"]

    intermediate = OUTPUT_DIR / "intermediate.mp4"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Crop/scale to 9:16 ──
    # Smart crop: center on speaker (assume center-horizontal)
    target_ar  = OUTPUT_WIDTH / OUTPUT_HEIGHT   # 0.5625
    source_ar  = src_w / src_h

    if source_ar > target_ar:
        # Wider than 9:16 — crop width
        crop_h = src_h
        crop_w = int(src_h * target_ar)
        crop_x = (src_w - crop_w) // 2
        crop_y = 0
    else:
        # Taller or square — crop height
        crop_w = src_w
        crop_h = int(src_w / target_ar)
        crop_x = 0
        crop_y = max(0, (src_h - crop_h) // 3)  # Keep face in upper third

    # ── Build complex filter ──
    # Layers:
    # 1. Main video: crop → scale → gentle zoom curve → subtle bottom gradient
    # 2. Subtitles via drawtext (burned in)
    # 3. Hook intro section (first 3s): zoom in + title text

    hook_dur  = min(hook_info["end"] - hook_info["start"], 4.0)
    hook_text = hook_info["phrase"].replace("'", "\\'").replace(":", "\\:").replace(",", "\\,")

    # Escape subtitle path
    sub_path_esc = str(SUBTITLE_FILE).replace(":", "\\:")

    # ── Zoom keyframe expression ──
    # Gentle zoom: 1.0 → 1.08 over full duration, with micro-oscillation
    zoom_expr = (
        "1.0 + 0.08*sin(PI*t/{dur}) + "
        "0.015*sin(2*PI*t/8)"
    ).format(dur=max(duration, 1))

    # Pan: slight horizontal drift
    x_expr = f"(iw-iw/zoom)/2 + 20*sin(2*PI*t/12)"
    y_expr = f"(ih-ih/zoom)/2 + 10*sin(2*PI*t/9)"

    filter_complex = (
        # Input crop + scale
        f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos,"
        # Gentle animated zoom
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
        f":d=1:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:fps={FPS},"
        # Very subtle bottom gradient vignette
        f"vignette=PI/4,"
        # Subtitles
        f"subtitles='{sub_path_esc}':force_style='"
        f"FontName=Arial Bold,"
        f"FontSize=18,"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BackColour=&H60000000,"
        f"Outline=2,"
        f"Shadow=1,"
        f"Alignment=2,"
        f"MarginV=80',"
        # Hook text overlay (first hook_dur seconds)
        f"drawtext=text='{hook_text}':"
        f"fontcolor=white:fontsize=22:x=(w-text_w)/2:y=h*0.15:"
        f"shadowcolor=black:shadowx=2:shadowy=2:"
        f"box=1:boxcolor=black@0.45:boxborderw=12:"
        f"enable='between(t,0,{hook_dur:.1f})',"
        # Fade in at start, fade out at end
        f"fade=t=in:st=0:d=0.5,fade=t=out:st={duration-1.5:.1f}:d=1.5"
        f"[vout]"
    )

    # Outro black + text (3 seconds)
    outro_text = "Próximamente: nuevo capítulo con nuestra periodoncista"

    cmd = (
        f'ffmpeg -y '
        f'-i "{INPUT_VIDEO}" '
        f'-filter_complex "{filter_complex}" '
        f'-map "[vout]" '
        f'-map 0:a '
        f'-af "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11" '
        f'-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p '
        f'-c:a aac -b:a 192k '
        f'-movflags +faststart '
        f'"{intermediate}"'
    )

    result = run(cmd)
    if result.returncode != 0:
        print("  Filter complex failed, trying simplified version...")
        return build_video_simple(video_info, hook_info)

    # ── Append outro ──
    outro = OUTPUT_DIR / "outro.mp4"
    final_out = OUTPUT_DIR / "dental_podcast_viral_FINAL.mp4"

    run(
        f'ffmpeg -y -f lavfi -i "color=c=black:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:d=3:r={FPS}" '
        f'-vf "drawtext=text=\'{outro_text}\':fontcolor=white:fontsize=20:'
        f'x=(w-text_w)/2:y=(h-text_h)/2:shadowcolor=black@0.8:shadowx=2:shadowy=2:'
        f'box=1:boxcolor=black@0.3:boxborderw=20:line_spacing=10:'
        f'fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf,'
        f'fade=t=in:st=0:d=0.8,fade=t=out:st=2:d=1" '
        f'-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p '
        f'-an "{outro}"'
    )

    # Concat
    concat_list = OUTPUT_DIR / "concat.txt"
    concat_list.write_text(f"file '{intermediate}'\nfile '{outro}'\n")
    run(
        f'ffmpeg -y -f concat -safe 0 -i "{concat_list}" '
        f'-c copy "{final_out}"'
    )

    size_mb = final_out.stat().st_size / 1_000_000
    print(f"\n  Final video: {final_out}")
    print(f"  Size: {size_mb:.1f} MB")
    return final_out


def build_video_simple(video_info, hook_info):
    """Fallback: simpler filter without zoompan (faster)."""
    print("  Using simplified pipeline...")
    src_w = video_info["width"]
    src_h = video_info["height"]
    duration = video_info["duration"]
    target_ar = OUTPUT_WIDTH / OUTPUT_HEIGHT
    source_ar = src_w / src_h

    if source_ar > target_ar:
        crop_h = src_h
        crop_w = int(src_h * target_ar)
        crop_x = (src_w - crop_w) // 2
        crop_y = 0
    else:
        crop_w = src_w
        crop_h = int(src_w / target_ar)
        crop_x = 0
        crop_y = max(0, (src_h - crop_h) // 3)

    sub_path_esc = str(SUBTITLE_FILE).replace(":", "\\:")
    hook_text = hook_info["phrase"].replace("'", "\\'").replace(":", "\\:").replace(",","\\,")
    hook_dur = min(hook_info["end"] - hook_info["start"] + 1, 4.0)
    intermediate = OUTPUT_DIR / "intermediate.mp4"
    outro = OUTPUT_DIR / "outro.mp4"
    final_out = OUTPUT_DIR / "dental_podcast_viral_FINAL.mp4"

    run(
        f'ffmpeg -y -i "{INPUT_VIDEO}" '
        f'-vf "crop={crop_w}:{crop_h}:{crop_x}:{crop_y},'
        f'scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos,'
        f'subtitles=\'{sub_path_esc}\':force_style=\'FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Alignment=2,MarginV=80\','
        f'drawtext=text=\'{hook_text}\':fontcolor=white:fontsize=22:'
        f'x=(w-text_w)/2:y=h*0.15:shadowcolor=black:shadowx=2:shadowy=2:'
        f'box=1:boxcolor=black@0.5:boxborderw=10:enable=\'between(t,0,{hook_dur:.1f})\','
        f'fade=t=in:st=0:d=0.5,fade=t=out:st={duration-1.5:.1f}:d=1.5" '
        f'-af "loudnorm=I=-16:TP=-1.5:LRA=11" '
        f'-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p '
        f'-c:a aac -b:a 192k -movflags +faststart '
        f'"{intermediate}"'
    )

    outro_text = "Próximamente\\: nuevo capítulo con nuestra periodoncista"
    run(
        f'ffmpeg -y -f lavfi -i "color=c=black:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:d=3:r={FPS}" '
        f'-vf "drawtext=text=\'{outro_text}\':fontcolor=white:fontsize=20:'
        f'x=(w-text_w)/2:y=(h-text_h)/2:shadowcolor=black@0.8:shadowx=2:shadowy=2,'
        f'fade=t=in:st=0:d=0.8,fade=t=out:st=2:d=1" '
        f'-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -an "{outro}"'
    )

    concat_list = OUTPUT_DIR / "concat.txt"
    concat_list.write_text(f"file '{intermediate}'\nfile '{outro}'\n")
    run(
        f'ffmpeg -y -f concat -safe 0 -i "{concat_list}" '
        f'-c copy "{final_out}"'
    )

    return final_out


# ── Step 8: Run ───────────────────────────────────────────────────────────────
def main():
    if not INPUT_VIDEO.exists():
        print(f"ERROR: Input video not found at {INPUT_VIDEO}")
        sys.exit(1)

    print("=" * 60)
    print("  DENTAL PODCAST VIRAL EDIT - PIPELINE START")
    print("=" * 60)

    video_info  = probe_video()
    clean_audio = extract_audio()
    transcript  = transcribe(clean_audio)
    hook_info   = find_hook(transcript)
    generate_srt(transcript)
    broll_clips = fetch_broll()
    final       = build_video(video_info, hook_info, broll_clips)

    print("\n" + "=" * 60)
    print(f"  DONE! Output: {final}")
    print("=" * 60)


if __name__ == "__main__":
    main()
