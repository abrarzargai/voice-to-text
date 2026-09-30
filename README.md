# Audio to Text

Turn audio or video files (mp3, mp4, m4a, aac, ogg, wav, …) into text. You can keep the original language or translate the speech to English.
Use it from a web page or the command line.

## Online or offline?

**It runs offline.** The AI model is OpenAI's **Whisper**, run on your own machine through the
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) library. Your audio is never
uploaded to any cloud service.

- **Internet is needed only once per model.** The first time you use a model, it is downloaded
  from Hugging Face and cached in `~/.cache/huggingface/`.
- **After that, it works with no internet at all.**
- The "upload" on the web page only sends the file to the local server on your own computer
  (`127.0.0.1`).

## Flow

```
 Browser (static/index.html)
   │  1. "New file": choose a file, a unique name and options (task, language, format...)
   ▼
 Flask server (app.py) ── creates a record, keeps the audio in data/audio/
   │  2. starts a background job (one heavy job runs at a time; others queue)
   ▼
 Whisper (faster-whisper, in its own process so Stop is instant)
   │  3. detects language, then transcribes/translates chunk by chunk
   ▼
 Browser polls /api/jobs/<id> every 5 seconds ── progress bar + text as it appears
   ▼
 data/db.json ── 4. saved as transcript v1 of the record
   │
   ├─ "Transcribe again" on the saved audio  → transcript v2, v3, ... (older versions are kept)
   └─ "Generate minutes" (local LLM via Ollama) → minutes v1, v2, ... (always English, editable)
```

Each record keeps its full history. You can rename a record, or delete a single version or the
whole record (with its audio).

## Setup from scratch

You need:

- **Python 3.9 or newer.** It is required.
- **Ollama.** It is optional and only needed for "Generate minutes".
- **An NVIDIA GPU with CUDA.** It is optional; without one, everything runs on the CPU.

You don't need to install ffmpeg separately, because faster-whisper includes the audio decoding.

### Linux

**1. Install Python and venv**

```bash
# Debian / Ubuntu / Mint
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

# Fedora
sudo dnf install -y python3 python3-pip

# Arch
sudo pacman -S --needed python python-pip

python3 --version                        # must print 3.9 or newer
```

**2. Get the project and install the dependencies**

```bash
cd ~/path/to/audio-to-text               # the folder that contains app.py
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**3. (Optional) Install Ollama for meeting minutes**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:0.5b                 # or pick and download a model later in Settings
```

The installer starts Ollama as a service on `http://127.0.0.1:11434`. If it is not running, start it with `ollama serve`.

**4. Start the app**

```bash
source venv/bin/activate                 # every time you open a new terminal
python app.py
```

Then open <http://127.0.0.1:5000> in your browser.

### Windows 10 / 11

**1. Install Python**

Download Python from <https://www.python.org/downloads/windows/>. In the installer, tick **"Add python.exe to PATH"**. Or install it from a terminal:

```powershell
winget install -e --id Python.Python.3.12
```

Close and reopen the terminal, then check the version:

```powershell
py --version                             # must print 3.9 or newer
```

**2. Get the project and install the dependencies** (PowerShell)

```powershell
cd C:\path\to\audio-to-text              # the folder that contains app.py
py -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell says *"running scripts is disabled on this system"*, run this once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

In Command Prompt (cmd.exe), activate with `venv\Scripts\activate.bat` instead.

**3. (Optional) Install Ollama for meeting minutes**

Download Ollama from <https://ollama.com/download/windows>, or install it with `winget install -e --id Ollama.Ollama`. Ollama runs in the system tray. Then run:

```powershell
ollama pull qwen2.5:0.5b                 # or pick and download a model later in Settings
```

**4. Start the app**

```powershell
venv\Scripts\Activate.ps1                # every time you open a new terminal
python app.py
```

Then open <http://127.0.0.1:5000> in your browser.

### (Optional) NVIDIA GPU

The app works on the CPU without any extra setup. To use **Device: GPU (CUDA)**, you need an up-to-date
NVIDIA driver plus the CUDA 12 cuBLAS and cuDNN 9 libraries:

- **Linux:** install the libraries into the venv and tell the loader where they are:

  ```bash
  pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
  export LD_LIBRARY_PATH=$(python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
  ```

  Run the `export` line in every new terminal before `python app.py`.
- **Windows:** install the [CUDA 12 Toolkit](https://developer.nvidia.com/cuda-downloads) and
  [cuDNN 9](https://developer.nvidia.com/cudnn-downloads), or copy the cuBLAS/cuDNN DLLs from
  [Purfview/whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs)
  into a folder on your `PATH`.

If the GPU cannot be used, choose **Device: CPU**.

### First run

- The first time you use a Whisper model, it is downloaded, which needs internet. Every run after that works offline.
- The minutes model is downloaded by Ollama with `ollama pull` or from the Settings page.
- The app keeps its data in `data/`. Set the `APP_DATA_DIR` environment variable to store it somewhere else.
- If Ollama runs on another address, set `OLLAMA_URL` (default `http://127.0.0.1:11434`).

### Command line (optional)

With the venv activated (on Linux or Windows):

```bash
python transcribe.py voice.mp4 --language hi --translate
python transcribe.py a.mp3 b.ogg --format srt -o outputs/
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `python3: command not found` / `'python' is not recognized` | Python is not installed or not on PATH. Reinstall it with "Add to PATH" ticked, then open a new terminal. |
| `No module named venv` / `ensurepip is not available` (Linux) | Run `sudo apt install python3-venv`. |
| `ModuleNotFoundError: No module named 'flask'` | The venv is not activated. Activate it, then run the command again. |
| "Cannot reach Ollama" when generating minutes | Start Ollama (`ollama serve`, or open the Ollama app on Windows). |
| `Library libcublas.so.12 is not found` / `cudnn` errors | Follow the GPU steps above, or set Device to CPU. |
| Port 5000 is already in use | Stop the other program that uses port 5000. The port is set at the bottom of `app.py`. |

## Options

| Option     | Choices                                   | Notes                                              |
|------------|-------------------------------------------|----------------------------------------------------|
| Task       | Transcribe / Translate to English         | Transcribe keeps the spoken language               |
| Language   | Auto-detect, en, hi, ur, …                | Choosing the language yourself is more reliable than auto-detect |
| Model      | tiny, base, small, medium, large-v3, large-v3-turbo | Bigger models are more accurate but slower. Use **large-v3-turbo**, because `small` is unreliable on Hindi/Urdu |
| Format     | Text with timestamps / plain text / .srt  | The .srt format makes subtitles                    |
| Device     | Auto / CPU / GPU (CUDA)                   | A GPU is much faster                               |
| Beam size  | 1–10 (default 5)                          | A higher value is slightly more accurate but slower |
| Skip silence (VAD) | on / off                          | Reduces made-up text in quiet parts                |

## Files

- `app.py` is the web server: records, background jobs, progress, stopping tasks and downloads.
- `store.py` reads and writes `data/db.json`.
- `whisper_worker.py` runs Whisper in a separate process that Stop can end immediately.
- `catalog.py` lists the Whisper and Ollama models shown in Settings.
- `static/index.html` is the web page.
- `transcribe.py` is the command-line tool. It also has the shared helpers: the model list and output formatting.
- `data/db.json` holds the settings (chosen models, options, theme) and every record with all its
  transcript and minutes versions. `data/audio/` holds the uploaded recordings.
- `outputs/` is only read once: the first time the app starts without `data/db.json`, the
  transcripts and minutes in it are imported as records. The files themselves are left in place.
