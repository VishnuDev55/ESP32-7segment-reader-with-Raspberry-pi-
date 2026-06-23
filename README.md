# 7-Segment Display Reader

A Python tool that reads numeric values off a 7-segment digital display in real time, using an ESP32-CAM as the camera source. It watches an MJPEG stream, finds the display automatically, decodes the digits using two independent recognition engines, and logs every confirmed reading to a CSV file.

---

## What It Does

- Connects to an ESP32-CAM's live MJPEG stream over WiFi
- Automatically detects the display region in the frame (or accepts a manual region)
- Runs **two recognition engines in parallel** on every digit and cross-checks them
- Uses **majority voting across multiple frames** to filter out one-off misreads
- Automatically re-checks the display position if reads start failing (camera/display drift)
- Logs every confirmed reading to a timestamped CSV file, with a numeric confidence score
- Works fully headless (no display needed) — or with a live debug GUI for setup
- Falls back to headless mode automatically if no display is available, instead of crashing

---

## Screenshots

<!--
  Add your screenshots to a `screenshots/` folder in this repo, then update
  the filenames below (or just keep these names and drop your images in).
-->

 Live GUI Debug View 

<img width="1918" height="1198" alt="gui_debug" src="https://github.com/user-attachments/assets/a88f7016-fce3-4d90-92fd-743ee5d7eca8" />

---

## Hardware Setup — ESP32-CAM

This project does **not** use any custom ESP32-CAM firmware. It relies entirely on the **stock `CameraWebServer` example sketch** that ships with the ESP32 Arduino core — no modifications needed.

1. In Arduino IDE: **File → Examples → ESP32 → Camera → CameraWebServer**
2. Uncomment the line matching your board, e.g.:
   ```cpp
   #define CAMERA_MODEL_AI_THINKER
   ```
3. Enter your WiFi credentials:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
4. Upload the sketch to the ESP32-CAM
5. Open the Serial Monitor and note the IP address it prints once connected, e.g.:
   ```
   Camera Ready! Use 'http://192.168.1.50' to connect
   ```
6. The MJPEG stream is available at:
   ```
   http://<that-ip>:81/stream
   ```
7. Point this tool at it using `--url` (see [Running It](#running-it) below)

This tool simply connects to the raw MJPEG stream that the example sketch already serves — the ESP32-CAM side stays completely stock.

---

## How It Works

### 1. Stream Capture
A background thread reads raw bytes directly from the ESP32-CAM's stream and manually parses JPEG frame boundaries. This avoids the buffering/lag issues that come from using OpenCV's built-in stream reader.

### 2. Region of Interest (ROI)
The tool needs to know *where* the digits are in the frame. It resolves this in order of priority:
1. A manually supplied ROI (`--roi`)
2. A previously cached ROI (saved to a JSON file, drawn once via the GUI and remembered on every future run)
3. Automatic detection — it looks for the largest bright, wide rectangle in the frame, since a lit display is usually the most prominent object of that shape

If reads start failing repeatedly (e.g. the camera or display physically shifted), it automatically re-runs detection rather than staying stuck on a stale region.

### 3. Preprocessing
Each cropped region is cleaned up before reading:

- Upscaled 3× for sharper edges
- Contrast-enhanced (CLAHE) to counter glare and uneven lighting
- Blurred slightly, then converted to a clean black-and-white image
- Small gaps and noise specks are smoothed out morphologically

### 4. Digit Detection
The binary image is scanned for individual digit-shaped blobs (and separately, for small decimal-point dots), then sorted left to right.

### 5. Dual-Engine Decoding
Each digit is read by **two independent methods**, and the results are cross-checked:

| Engine | How it works |
|---|---|
| **Segment-State Engine** | Splits the digit into the 7 classic segments (a–g) and checks pixel density in each zone to see if it's "lit" |
| **Template-Matching Engine** | Compares the digit against synthetically drawn reference digits (0–9) generated entirely in code |

- If both engines agree → high-confidence result
- If they disagree → the template engine wins (it's more resistant to threshold noise)
- If both are uncertain → the reading is discarded rather than guessed

### 6. Voting & Logging
A single misread frame won't get logged. The tool buffers several consecutive readings and only commits a value once enough of them agree — then writes it to CSV with a timestamp and confidence score.

---

## Prerequisites

- Python 3.10+
- An ESP32-CAM running the stock `CameraWebServer` sketch, streaming over WiFi on the same network

### Install dependencies

```bash
pip install -r requirements.txt
```

Or directly:

```bash
pip install opencv-python numpy
```

---

## Running It

### Basic usage (headless, auto-detect everything)

```bash
python display_reader.py
```

### With a live debug window (recommended for first-time setup)

```bash
python display_reader.py --gui
```

Inside the GUI, press **`1`** and drag a box around the digits — it's saved automatically and remembered on every future run, even headless.

### Point it at your ESP32-CAM's IP

```bash
python display_reader.py --url http://192.168.1.50:81/stream
```

### Manually set the display region instead of auto-detecting

```bash
python display_reader.py --roi 100,150,200,80
```

### Take a single test reading and exit

```bash
python display_reader.py --once
```

### Limit the expected digit count

```bash
python display_reader.py --max-digits 4
```

### Change how often it reads (in seconds)

```bash
python display_reader.py --interval 5
```

### Log to a custom file

```bash
python display_reader.py --log readings.csv
```

### Put it all together

```bash
python display_reader.py --url http://192.168.1.50:81/stream --gui --interval 3 --log session1.csv
```

---

## All Command-Line Options

| Flag | Default | Description |
|---|---|---|
| `--url` | `http://192.168.137.219:81/stream` | ESP32-CAM stream URL |
| `--log` | `display_log.csv` | Output CSV path |
| `--interval` | `5.0` | Seconds between reads |
| `--roi` | auto | Manual region as `x,y,w,h` |
| `--roi-file` | `display_roi.json` | Where the detected ROI is cached between runs |
| `--gui` | off | Show live debug windows |
| `--debug-dir` | `.` | Folder to save debug snapshots into |
| `--value-min` | `0.0` | Minimum value considered valid |
| `--value-max` | `95.0` | Maximum value considered valid |
| `--max-digits` | `5` | Maximum number of digits expected on the display |
| `--once` | off | Take a single reading (no voting) and exit |

---

## GUI Controls (when running with `--gui`)

| Key | Action |
|---|---|
| `1` | Draw a region around the digits manually |
| `r` | Re-run automatic region detection |
| `s` | Save a debug snapshot (raw + processed image) |
| `q` | Quit |

---

## Output Format

Every confirmed reading is appended to the CSV log as:

```
Timestamp,            Value,  Confidence,            Score
2026-06-22 14:02:31,   42.5,   2/3 frames agree,      0.67
```

`Score` is left blank for single readings taken with `--once`, since there's no multi-frame vote to score.

---

## Notes

- Readings outside the `--value-min` / `--value-max` range are automatically discarded — useful for filtering out obvious misreads.
- The ROI is cached after the first successful detection (auto or manual), so the camera doesn't need to re-search the frame on every run.
- If auto-detection fails (e.g. unusual lighting or camera angle), run once with `--gui` and press `1` to draw the region manually — it'll be remembered for next time.
- If `--gui` is passed but no display is available (e.g. SSH into a headless Raspberry Pi), the tool automatically falls back to headless mode instead of crashing.

---

## License

MIT — see [LICENSE](LICENSE) for details.
