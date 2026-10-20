# AI YouTube Clipper

A locally hosted web application that helps creators turn YouTube videos and local video files into short clips with automatic transcription and editable subtitles.

## Application Preview

### Dashboard

![AI YouTube Clipper Dashboard](<assets/Preview 1 AI Clipper.png>)

### Video Preview and Clip Settings

![AI YouTube Clipper Video Preview and Clip Settings](<assets/Preview 2 AI Clipper.png>)

### Subtitle Editing and Video Export

![AI YouTube Clipper Subtitle Editing and Export](<assets/Preview 3 AI Clipper.png>)

The interface provides a single workspace for importing videos, previewing footage, selecting clip timestamps, configuring output resolution, and preparing editable subtitles.

## Features

- **YouTube Video Import** — Import videos using a YouTube URL.
- **Local Video Upload** — Upload a video file directly from your computer.
- **Clip Selection** — Select the start and end times of the desired segment.
- **Automatic Transcription** — Generate speech transcripts using Faster-Whisper.
- **Editable Transcripts** — Review and correct transcription results before rendering.
- **Burned-in Subtitles** — Render subtitles directly into the exported video.
- **Multiple Output Resolutions** — Choose 720p, 1080p, 1440p, or 2160p.
- **Video Export** — Render and download the finished clip.

## Tech Stack

| Technology            | Purpose                                            |
| --------------------- | -------------------------------------------------- |
| Python                | Backend application                                |
| FastAPI               | REST API and web server                            |
| Faster-Whisper        | Speech recognition and transcription               |
| FFmpeg                | Video processing, resizing, and subtitle rendering |
| pytubefix             | YouTube video importing                            |
| HTML, CSS, JavaScript | Frontend interface                                 |

## Requirements

- Windows
- Python 3.11
- FFmpeg installed and available in the system `PATH`
- Internet connection for YouTube importing and model downloads

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/Mozkyy/ai-youtube-clipper.git
cd ai-youtube-clipper
```

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the required Python dependencies using the `requirements.txt` file:

```powershell
python -m pip install -r requirements.txt
```

### 4. Verify FFmpeg

```powershell
ffmpeg -version
```

Make sure FFmpeg is available in your system `PATH`.

## Running the Application

From the project root, with the virtual environment activated, run:

```powershell
python -m uvicorn backend.main:app --reload
```

Open the following URLs in your browser:

- **Web Application:** http://127.0.0.1:8000/app/
- **API Documentation:** http://127.0.0.1:8000/docs

Keep the terminal running while using the application. Press `Ctrl+C` to stop the server.

## How to Use

1. Open the web application.
2. Import a YouTube video or upload a local video.
3. Select the start and end times for your clip.
4. Generate the transcript.
5. Review and edit the transcript if necessary.
6. Select the output resolution.
7. Render the subtitled clip and download the result.

## Project Structure

```text
ai-youtube-clipper/
├── backend/
│   └── main.py
├── frontend/
│   └── index.html
├── .gitignore
└── README.md
```

## Limitations

- Transcription accuracy depends on audio quality, background noise, accents, and language.
- YouTube importing may be affected by changes to YouTube or restrictions on individual videos.
- Rendering time depends on video duration, output resolution, and hardware.
- The application currently runs locally and is not a publicly hosted service.

## Roadmap

- Improve Indonesian and mixed-language transcription accuracy.
- Add a reproducible dependency file and setup checks.
- Expand subtitle customization options.
- Improve error handling and automated testing.

## Author

**Mozkyy**

- GitHub: [@Mozkyy](https://github.com/Mozkyy)
- Repository: [AI YouTube Clipper](https://github.com/Mozkyy/ai-youtube-clipper)

## License

No license has been specified yet. Until a license is added to the repository, reuse and redistribution permissions should not be assumed.

---

_Built as a portfolio project exploring AI-assisted video editing and speech transcription._
