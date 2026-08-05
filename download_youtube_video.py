import yt_dlp

def download_video(url):
    # Options for yt-dlp
    ydl_opts = {
        # bestvideo+bestaudio merges separate streams (needed for >720p on YouTube).
        # Falls back to best progressive file if merge isn't possible.
        #'format': 'bestvideo+bestaudio/best',
        'format': 'bestvideo[height<=1080]+bestaudio/best',
        'merge_output_format': 'mp4',  # requires ffmpeg
        'outtmpl': '%(title)s.%(ext)s',  # Set filename to video title
        'noplaylist': True,  # Download single video if URL is part of a playlist
        'quiet': False,  # Show download progress in the console
        'ignoreerrors': True,  # Continue even if an error is encountered
        'no_warnings': True,  # Suppress warnings
    }
    try:
        # Use yt-dlp to download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with the desired YouTube video URL
download_video('https://www.youtube.com/watch?v=SegaZRr8t_8')
