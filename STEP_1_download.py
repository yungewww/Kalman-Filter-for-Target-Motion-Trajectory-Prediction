'''
STEP 1
- Set up Docker environment
- Download videos from YouTube
'''

import ssl
import os
from pytube import YouTube

urls = [
    "https://www.youtube.com/watch?v=WeF4wpw7w9k",
    "https://www.youtube.com/watch?v=2NFwY15tRtA",
    "https://www.youtube.com/watch?v=5dRramZVu2Q",
    "https://www.youtube.com/watch?v=2hQx48U1L-Y"
]

ssl._create_default_https_context = ssl._create_unverified_context

def download_videos(urls):
    download_path = 'video'
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for url in urls:
        try:
            yt = YouTube(url)

            stream = yt.streams.get_highest_resolution()
            print(f"Downloading {yt.title}...")
            stream.download(output_path=download_path)
            print("Download completed!")
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")

download_videos(urls)

