import streamlit as st
import os
import subprocess
import tempfile
import requests
import json
from pytube import YouTube
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import uuid

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# ---------- Helper Functions ----------
def download_youtube_video(url, output_path):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
    stream.download(output_path=output_path, filename="video.mp4")
    return os.path.join(output_path, "video.mp4")

def generate_subtitles(video_path):
    audio_path = video_path.replace(".mp4", ".wav")
    subprocess.run(["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path])

    whisper_url = "https://api-inference.huggingface.co/models/openai/whisper-base"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    with open(audio_path, "rb") as f:
        response = requests.post(whisper_url, headers=headers, data=f.read())

    try:
        return response.json()["text"]
    except:
        return "Transcript not available."

def analyze_video_with_groq(transcript):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    prompt = f"""
    Analyze this transcript and return 10-20 most viral segments with:
    - start time
    - end time
    - viral_score (1-100)
    Format response as JSON list: [{{"start": seconds, "end": seconds, "viral_score": score}}]
    Transcript: {transcript}
    """
    payload = {"model": "llama3-70b-8192", "input": prompt}
    response = requests.post("https://api.groq.com/v1/completions", headers=headers, json=payload)
    try:
        return json.loads(response.json()["choices"][0]["text"])
    except:
        return []

def process_segment(video_path, seg, transcript, idx):
    start, end = seg["start"], seg["end"]
    clip_file = f"short_{idx+1}.mp4"
    subprocess.run(["ffmpeg", "-i", video_path, "-ss", str(start), "-to", str(end), "-c", "copy", clip_file])
    caption_text = transcript[:200].replace("\n", " ")
    subtitled_file = f"short_{idx+1}_subtitled.mp4"
    subprocess.run([
        "ffmpeg", "-i", clip_file, "-vf",
        f"drawtext=text='{caption_text}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=h-50",
        "-codec:a", "copy", subtitled_file
    ])
    return {"file": subtitled_file, "viral_score": seg["viral_score"]}

def create_shorts_parallel(video_path, segments, transcript, update_progress):
    shorts = []
    total = len(segments)
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_segment, video_path, seg, transcript, idx): idx for idx, seg in enumerate(segments)}
        for count, future in enumerate(as_completed(futures), start=1):
            shorts.append(future.result())
            update_progress(count / total)
    return sorted(shorts, key=lambda x: x["viral_score"], reverse=True)

def create_zip(shorts):
    zip_filename = "shorts_bundle.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for short in shorts:
            zipf.write(short["file"])
    return zip_filename

# ---------- Streamlit App with Resumable + Queue ----------
st.title("AI-Powered YouTube Shorts Generator (Resumable + Multi-User Queue)")

# Initialize session state
if "queue" not in st.session_state:
    st.session_state.queue = []
if "results" not in st.session_state:
    st.session_state.results = {}
if "processing" not in st.session_state:
    st.session_state.processing = None

youtube_url = st.text_input("Enter YouTube URL")
job_id = str(uuid.uuid4())

if st.button("Submit Job") and youtube_url:
    st.session_state.queue.append({"id": job_id, "url": youtube_url})
    st.info(f"Job added to queue. Job ID: {job_id}")

# Process queue if idle
if st.session_state.processing is None and st.session_state.queue:
    current_job = st.session_state.queue.pop(0)
    st.session_state.processing = current_job["id"]
    st.info(f"Processing Job: {current_job['id']}")

    with st.spinner("Downloading, transcribing, and generating shorts..."):
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)
        video_path = download_youtube_video(current_job["url"], temp_dir)
        transcript = generate_subtitles(video_path)
        segments = analyze_video_with_groq(transcript)

        if segments:
            progress_bar = st.progress(0)
            shorts = create_shorts_parallel(video_path, segments, transcript, lambda x: progress_bar.progress(x))
            zip_file = create_zip(shorts)
            st.session_state.results[current_job["id"]] = {"shorts": shorts, "zip": zip_file}
            progress_bar.empty()
            st.success("Shorts Generated!")
        else:
            st.session_state.results[current_job["id"]] = None
            st.error("Failed to generate shorts.")
        
        st.session_state.processing = None  # Free worker for next job

# Show results if available
for job, data in st.session_state.results.items():
    if data:
        st.write(f"### Results for Job {job}")
        for short in data["shorts"]:
            st.video(short["file"])
            st.write(f"Viral Score: {short['viral_score']}")
        with open(data["zip"], "rb") as f:
            st.download_button("Download All Shorts as ZIP", f, file_name=f"shorts_{job}.zip")
