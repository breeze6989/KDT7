import os, uuid, subprocess; from flask import current_app
FFMPEG = "ffmpeg"

def save_clip(rtsp_url: str, seconds: int = 10) -> str:
    """RTSP → seconds초 mp4 저장, 파일명 반환"""
    out_dir = os.path.join(current_app.config["MEDIA_ROOT"], "clips"); os.makedirs(out_dir, exist_ok=True)
    fname = f"clip_{uuid.uuid4().hex}.mp4"; path = os.path.join(out_dir, fname)
    cmd = [FFMPEG,"-y","-rtsp_transport","tcp","-i", rtsp_url,"-t", str(seconds),"-vcodec","copy","-acodec","copy", path]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return fname