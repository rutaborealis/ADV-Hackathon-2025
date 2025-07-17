import os
import sys
from pymediainfo import MediaInfo
import openai
import whisper
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import easyocr
import numpy as np
from pyzbar.pyzbar import decode as decode_qr
from PIL import Image
import zipfile
import json
from datetime import datetime
import warnings

# Подавление warnings
warnings.filterwarnings('ignore')

# Подавление специфических warnings
import logging
logging.getLogger('moviepy').setLevel(logging.ERROR)
logging.getLogger('whisper').setLevel(logging.ERROR)
logging.getLogger('easyocr').setLevel(logging.ERROR)

# Отключение предупреждений OpenCV
cv2.setUseOptimized(True)
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Создаем папку report, если её нет
os.makedirs('report', exist_ok=True)

VIDEO_REQUIREMENTS = {
    'container_format': 'MXF',
    'video_codec': 'MPEG Video',
    'width': 1920,
    'height': 1080,
    'frame_rate': 25,
    'aspect_ratio': '16:9',
    'scan_type': 'Interlaced',
    'bit_depth': 8,
    'color_space': 'YUV',
    'chroma_subsampling': '4:2:2',
    'bitrate': 50000000,  # 50 Mbps
}
AUDIO_REQUIREMENTS = {
    'format': 'PCM',
    'channels': 2,
    'sample_rate': 48000,
    'bit_depth': 24,
}

def validate_video(file_path):
    report = []
    media_info = MediaInfo.parse(file_path)
    video_tracks = [t for t in media_info.tracks if t.track_type == 'Video']
    audio_tracks = [t for t in media_info.tracks if t.track_type == 'Audio']
    general_track = next((t for t in media_info.tracks if t.track_type == 'General'), None)

    # Container
    if general_track:
        container = general_track.format
        if VIDEO_REQUIREMENTS['container_format'] in container:
            report.append(f"Container: OK ({container})")
        else:
            report.append(f"Container: FAIL ({container})")
    else:
        report.append("Container: FAIL (not found)")

    # Video
    if video_tracks:
        v = video_tracks[0]
        # Codec
        if VIDEO_REQUIREMENTS['video_codec'] in (v.format or ''):
            report.append(f"Video Codec: OK ({v.format})")
        else:
            report.append(f"Video Codec: FAIL ({v.format})")
        # Resolution
        if v.width == VIDEO_REQUIREMENTS['width'] and v.height == VIDEO_REQUIREMENTS['height']:
            report.append(f"Resolution: OK ({v.width}x{v.height})")
        else:
            report.append(f"Resolution: FAIL ({v.width}x{v.height})")
        # Frame rate
        try:
            fr = float(v.frame_rate)
        except:
            fr = None
        if fr and abs(fr - VIDEO_REQUIREMENTS['frame_rate']) < 0.1:
            report.append(f"Frame Rate: OK ({fr} fps)")
        else:
            report.append(f"Frame Rate: FAIL ({v.frame_rate})")
        # Aspect ratio
        if v.display_aspect_ratio == VIDEO_REQUIREMENTS['aspect_ratio']:
            report.append(f"Aspect Ratio: OK ({v.display_aspect_ratio})")
        else:
            report.append(f"Aspect Ratio: FAIL ({v.display_aspect_ratio})")
        # Scan type
        if v.scan_type and VIDEO_REQUIREMENTS['scan_type'].lower() in v.scan_type.lower():
            report.append(f"Scan Type: OK ({v.scan_type})")
        else:
            report.append(f"Scan Type: FAIL ({v.scan_type})")
        # Bit depth
        if v.bit_depth and int(v.bit_depth) == VIDEO_REQUIREMENTS['bit_depth']:
            report.append(f"Bit Depth: OK ({v.bit_depth})")
        else:
            report.append(f"Bit Depth: FAIL ({v.bit_depth})")
        # Color space
        if v.color_space and VIDEO_REQUIREMENTS['color_space'] in v.color_space:
            report.append(f"Color Space: OK ({v.color_space})")
        else:
            report.append(f"Color Space: FAIL ({v.color_space})")
        # Chroma subsampling
        if v.chroma_subsampling and VIDEO_REQUIREMENTS['chroma_subsampling'] in v.chroma_subsampling:
            report.append(f"Chroma Subsampling: OK ({v.chroma_subsampling})")
        else:
            report.append(f"Chroma Subsampling: FAIL ({v.chroma_subsampling})")
        # Bitrate
        try:
            br = int(v.bit_rate)
        except:
            br = None
        if br and abs(br - VIDEO_REQUIREMENTS['bitrate']) < 2000000:
            report.append(f"Bitrate: OK ({br} bps)")
        else:
            report.append(f"Bitrate: FAIL ({v.bit_rate})")
    else:
        report.append("Video Track: FAIL (not found)")

    # Audio
    if len(audio_tracks) == AUDIO_REQUIREMENTS['channels']:
        report.append(f"Audio Tracks: OK ({len(audio_tracks)})")
    else:
        report.append(f"Audio Tracks: FAIL ({len(audio_tracks)})")
    for idx, a in enumerate(audio_tracks):
        # Format
        if AUDIO_REQUIREMENTS['format'] in (a.format or ''):
            report.append(f"Audio {idx+1} Format: OK ({a.format})")
        else:
            report.append(f"Audio {idx+1} Format: FAIL ({a.format})")
        # Sample rate
        if a.sampling_rate and int(a.sampling_rate) == AUDIO_REQUIREMENTS['sample_rate']:
            report.append(f"Audio {idx+1} Sample Rate: OK ({a.sampling_rate})")
        else:
            report.append(f"Audio {idx+1} Sample Rate: FAIL ({a.sampling_rate})")
        # Bit depth
        if a.bit_depth and int(a.bit_depth) == AUDIO_REQUIREMENTS['bit_depth']:
            report.append(f"Audio {idx+1} Bit Depth: OK ({a.bit_depth})")
        else:
            report.append(f"Audio {idx+1} Bit Depth: FAIL ({a.bit_depth})")
    return report

def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language='ru')
    return result['text']

def extract_frames(video_path, interval_sec=1):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0
    success, image = vidcap.read()
    while success:
        if int(count % (fps * interval_sec)) == 0:
            frames.append(image.copy())
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return frames

def ocr_text_from_frames(frames):
    reader = easyocr.Reader(['ru', 'en'])
    texts = []
    for frame in frames:
        result = reader.readtext(frame)
        texts.extend([r[1] for r in result])
    return texts

def detect_qr_codes(frames):
    qr_results = []
    for idx, frame in enumerate(frames):
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        decoded = decode_qr(pil_img)
        for d in decoded:
            qr_results.append({'frame': idx, 'data': d.data.decode('utf-8')})
    return qr_results



def legal_ai_check(transcript, ocr_texts, qr_results, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)
    # Read legal requirements from file
    with open('legal_requirements.txt', 'r', encoding='utf-8') as f:
        legal_requirements = f.read()
    prompt = f"""
Ниже приведены требования к ролику (на русском языке):
{legal_requirements}

Транскрипция аудио:
{transcript}

Текст с экрана (OCR):
{ocr_texts}

QR-коды:
{qr_results}

Проверьте, соблюдены ли требования. Для каждого требования укажите: OK/FAIL и комментарий.
"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000
    )
    return response.choices[0].message.content

def ai_legal_validate(video_path, openai_api_key):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        print("[AI] Extracting audio from video...")
        extract_audio(video_path, audio_path)
        print("[AI] Transcribing audio with Whisper...")
        transcript = transcribe_audio(audio_path)
        print("[AI] Extracting frames from video...")
        frames = extract_frames(video_path, interval_sec=1)
        print(f"[AI] Extracted {len(frames)} frames.")
        print("[AI] Running OCR on frames...")
        ocr_texts = ocr_text_from_frames(frames)
        print(f"[AI] OCR found {len(ocr_texts)} text snippets.")
        print("[AI] Detecting QR codes in frames...")
        qr_results = detect_qr_codes(frames)
        print(f"[AI] Detected {len(qr_results)} QR codes.")
        print("[AI] Sending data to OpenAI for legal compliance check...")
        report = legal_ai_check(transcript, ocr_texts, qr_results, openai_api_key)
        
        # Save AI report to report folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ai_report_filename = os.path.join("report", f"openai_analysis_{timestamp}.txt")
        with open(ai_report_filename, "w", encoding="utf-8") as f:
            f.write(f"OpenAI Legal Compliance Analysis\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video file: {video_path}\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        
        print(f"[AI] Report saved to: {ai_report_filename}")
    print("[AI] Legal compliance check complete.")
    return report

def save_technical_validation_report(file_path, validation_report):
    """Save technical validation report to report folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tech_report_filename = os.path.join("report", f"technical_validation_{timestamp}.txt")
    
    with open(tech_report_filename, "w", encoding="utf-8") as f:
        f.write(f"Technical Validation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video file: {file_path}\n")
        f.write("=" * 50 + "\n\n")
        
        # Add technical requirements reference
        f.write("TECHNICAL REQUIREMENTS REFERENCE:\n")
        f.write("- Container format: MXF\n")
        f.write("- Video codec: MPEG Video\n")
        f.write("- Resolution: 1920x1080\n")
        f.write("- Frame rate: 25 fps\n")
        f.write("- Aspect ratio: 16:9\n")
        f.write("- Scan type: Interlaced\n")
        f.write("- Bit depth: 8 bit\n")
        f.write("- Color space: YUV\n")
        f.write("- Chroma subsampling: 4:2:2\n")
        f.write("- Bitrate: 50 Mbps\n")
        f.write("- Audio format: PCM\n")
        f.write("- Audio channels: 2\n")
        f.write("- Audio sample rate: 48 kHz\n")
        f.write("- Audio bit depth: 24 bit\n")
        f.write("\n" + "=" * 50 + "\n\n")
        
        # Add validation results
        f.write("VALIDATION RESULTS:\n")
        for line in validation_report:
            f.write(line + "\n")
        
        # Add summary
        f.write("\n" + "=" * 50 + "\n")
        f.write("SUMMARY:\n")
        ok_count = sum(1 for line in validation_report if "OK" in line)
        fail_count = sum(1 for line in validation_report if "FAIL" in line)
        total_count = len(validation_report)
        f.write(f"Total checks: {total_count}\n")
        f.write(f"Passed: {ok_count}\n")
        f.write(f"Failed: {fail_count}\n")
        f.write(f"Success rate: {(ok_count/total_count)*100:.1f}%\n")
        
        if fail_count == 0:
            f.write("\n✅ ALL TECHNICAL REQUIREMENTS MET\n")
        else:
            f.write(f"\n❌ {fail_count} TECHNICAL REQUIREMENT(S) FAILED\n")
    
    print(f"[TECH] Technical validation report saved to: {tech_report_filename}")
    return tech_report_filename

def create_chatgpt_archive(file_path, validation_report, transcript, ocr_texts, qr_results):
    """Create a zip file with data for ChatGPT analysis in report folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = os.path.join("report", f"chatgpt_analysis_{timestamp}.zip")
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Add validation report
        zipf.writestr("validation_report.txt", "\n".join(validation_report))
        
        # Add transcript
        zipf.writestr("transcript.txt", transcript)
        
        # Add OCR results
        zipf.writestr("ocr_results.txt", "\n".join(ocr_texts))
        
        # Add QR codes
        qr_data = json.dumps(qr_results, indent=2, ensure_ascii=False)
        zipf.writestr("qr_codes.json", qr_data)
    
    # Save prompt to report folder
    prompt = f"""Проанализируйте данные из видеофайла {file_path}.

ДАННЫЕ ДЛЯ АНАЛИЗА:

1. ТЕХНИЧЕСКАЯ ВАЛИДАЦИЯ:
{chr(10).join(validation_report)}

2. ТРАНСКРИПЦИЯ АУДИО:
{transcript}

3. ТЕКСТ С ЭКРАНА (OCR):
{chr(10).join(ocr_texts)}

4. QR-КОДЫ:
{json.dumps(qr_results, indent=2, ensure_ascii=False)}

ЗАДАЧА:
Проверьте соответствие видео требованиям из файла legal_requirements.txt.
Для каждого требования укажите: OK/FAIL и подробный комментарий.
"""
    
    prompt_filename = os.path.join("report", f"chatgpt_prompt_{timestamp}.txt")
    with open(prompt_filename, "w", encoding="utf-8") as f:
        f.write(prompt)
    
    return zip_filename, prompt_filename

def main():
    file_path = "sovkombank_10sec_clcut_240918_16x9_opt1_tv.mxf"
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    # Run validation
    print("[TECH] Starting technical validation...")
    validation_report = validate_video(file_path)
    print("\nValidation Report:")
    for line in validation_report:
        print(line)
    
    # Save technical validation report to file
    tech_report_filename = save_technical_validation_report(file_path, validation_report)
    
    # Collect data for analysis
    print("\nCollecting data for analysis...")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(file_path, audio_path)
        transcript = transcribe_audio(audio_path)
        frames = extract_frames(file_path, interval_sec=1)
        ocr_texts = ocr_text_from_frames(frames)
        qr_results = detect_qr_codes(frames)
    
    # Try OpenAI analysis (commented out for now)
    ai_report = None
    try:
        ai_report = ai_legal_validate(file_path, "")
        print(ai_report)
    except Exception as e:
        print(f"AI analysis failed: {e}")
    
    # Create ChatGPT archive (always works)
    zip_filename, prompt_filename = create_chatgpt_archive(file_path, validation_report, transcript, ocr_texts, qr_results)
    print(f"\nChatGPT archive created: {zip_filename}")
    print(f"ChatGPT prompt saved: {prompt_filename}")
    print("Upload this archive to ChatGPT along with the prompt")
    
    # Summary of generated files
    print(f"\n📁 Generated files in report/ folder:")
    print(f"  • Technical validation: {os.path.basename(tech_report_filename)}")
    print(f"  • ChatGPT archive: {os.path.basename(zip_filename)}")
    print(f"  • ChatGPT prompt: {os.path.basename(prompt_filename)}")
    if ai_report:
        print(f"  • AI analysis: openai_analysis_*.txt")

if __name__ == "__main__":
    main() 
    