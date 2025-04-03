# app/core/data_collection/video_collector.py
import os
import requests
import youtube_dl
from pytube import YouTube
from moviepy.editor import VideoFileClip

class VideoCollector:
    def __init__(self, output_dir="./data/raw"):
        """Initialize the video collector with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def download_from_youtube(self, url, output_filename=None):
        """Download a video from YouTube."""
        try:
            yt = YouTube(url)
            if output_filename is None:
                output_filename = f"{yt.title.replace(' ', '_')}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(output_path=self.output_dir, filename=output_filename)
            return output_path
        except Exception as e:
            print(f"Error downloading from YouTube: {e}")
            return None
            
    def extract_audio(self, video_path, output_filename=None):
        """Extract audio from a video file."""
        try:
            if output_filename is None:
                output_filename = os.path.splitext(os.path.basename(video_path))[0] + ".mp3"
            output_path = os.path.join(self.output_dir, output_filename)
            
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(output_path)
            return output_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
