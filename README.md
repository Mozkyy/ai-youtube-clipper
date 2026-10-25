# AI YouTube Clipper

A locally hosted web application that helps content creators turn YouTube videos and local video files into short, vertical clips with automatic speech transcription, editable subtitles, and AI-assisted video cropping.

## Application Preview

### Dashboard

![AI YouTube Clipper Dashboard](assets/Preview%201%20AI%20Clipper.png)

### Video Preview and Clip Settings

![AI YouTube Clipper Video Preview and Clip Settings](assets/Preview%202%20AI%20Clipper.png)

### Subtitle Editing and Video Export

![AI YouTube Clipper Subtitle Editing and Export](assets/Preview%203%20AI%20Clipper.png)

The application provides a single workspace for importing videos, previewing footage, selecting clip timestamps, choosing output resolutions, editing transcripts, and rendering subtitled videos.

## Features

- **YouTube Video Import** — Import videos using a YouTube URL.
- **Local Video Upload** — Upload video files directly from your computer.
- **Clip Selection** — Specify the start and end times of the desired segment.
- **Automatic Speech Transcription** — Generate transcripts using Faster-Whisper.
- **Editable Transcripts** — Review and correct transcription results before exporting.
- **Burned-in Subtitles** — Render subtitles directly into the exported video.
- **Vertical Video Cropping** — Convert landscape footage into vertical video suitable for short-form content.
- **Face-Aware Cropping** — Track detected face positions to help keep the speaker in frame, with smoothing and movement constraints to reduce unnecessary crop movement.
- **Multiple Output Resolutions** — Choose 720p, 1080p, 1440p, or 2160p, depending on the selected export settings and source footage.
- **Video Export and Download** — Render and download the finished clip.
- **Local Processing** — Run the application on your own computer without requiring a publicly hosted web service.

## Tech Stack

| Technology            | Purpose                                                                      |
| --------------------- | ---------------------------------------------------------------------------- |
| Python                | Backend application                                                          |
| FastAPI               | REST API and web server                                                      |
| Faster-Whisper        | Speech recognition and transcription                                         |
| OpenCV                | Face detection for face-aware cropping                                       |
| FFmpeg                | Video trimming, cropping, resizing, audio processing, and subtitle rendering |
| pytubefix             | YouTube video importing                                                      |
| HTML, CSS, JavaScript | Frontend interface                                                           |

## Requirements

- Windows
- Python 3.11
- FFmpeg installed and available in the system `PATH`
- Internet connection for YouTube importing and initial model downloads
- Sufficient storage for source videos, temporary files, exported clips, and speech recognition models

## Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/Mozkyy/ai-youtube-clipper.git
cd ai-youtube-clipper
```

### 2. Create a Virtual Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, review your local PowerShell execution policy before changing any security settings.

### 3. Install Dependencies

Install the required Python packages using the project's dependency file:

```powershell
python -m pip install -r requirements.txt
```

### 4. Verify FFmpeg

```powershell
ffmpeg -version
```

Make sure FFmpeg is installed correctly and accessible from your terminal.

### 5. Start the Application

From the project root, with the virtual environment activated, run:

```powershell
python -m uvicorn backend.main:app --reload
```

## Running the Application

Open the following URLs in your browser:

- **Web Application:** http://127.0.0.1:8000/app/
- **API Documentation:** http://127.0.0.1:8000/docs

Keep the terminal running while using the application. Press `Ctrl+C` to stop the server.

## How to Use

1. Open the web application.
2. Import a video using a YouTube URL or upload a local video file.
3. Preview the video and select the desired start and end times.
4. Generate a transcript for the selected segment.
5. Review and edit the transcript as needed.
6. Select the desired output resolution.
7. Render the clip with burned-in subtitles.
8. Download the finished video.

## Project Structure

```text
ai-youtube-clipper/
├── backend/
│   └── main.py
├── frontend/
│   └── index.html
├── assets/
│   ├── Preview 1 AI Clipper.png
│   ├── Preview 2 AI Clipper.png
│   └── Preview 3 AI Clipper.png
├── .gitignore
├── README.md
└── requirements.txt
```

The structure above describes the main project files and preview assets. Additional directories may be created at runtime for uploaded videos, temporary files, models, and exported clips.

## Limitations

- Transcription accuracy depends on audio quality, background noise, accents, speaking speed, and language.
- Indonesian and mixed Indonesian-English transcription may require manual corrections.
- Face-aware cropping depends on face detection quality and may occasionally select an incorrect face or lag behind rapid movement.
- Videos with multiple speakers, scene changes, or subtitles embedded in the source footage may affect cropping behavior.
- YouTube importing may be affected by changes to YouTube, video availability, or access restrictions.
- Rendering time and output quality depend on video duration, source footage, output resolution, and available hardware.
- Higher output resolutions do not restore detail that is absent from the source video.
- The application currently runs locally and is not a publicly hosted service.

## Roadmap

- Improve transcription accuracy for Indonesian and mixed Indonesian-English speech.
- Develop more context-aware transcript correction.
- Improve face tracking and crop stability across scene changes.
- Expand subtitle customization options.
- Improve error handling and automated testing.
- Improve installation documentation and dependency verification.

## Author

**Mozkyy**

- GitHub: [@Mozkyy](https://github.com/Mozkyy)
- Repository: [AI YouTube Clipper](https://github.com/Mozkyy/ai-youtube-clipper)

## License

No license has been specified yet. Until a license is added to the repository, reuse and redistribution permissions should not be assumed.

---

_Built as a portfolio project exploring AI-assisted video editing, speech transcription, and automated short-form video creation._
