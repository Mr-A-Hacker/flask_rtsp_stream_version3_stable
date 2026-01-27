📡 Flask RTSP Stream — Version 3 (Stable)
A fast, modular, and production‑ready Flask application for streaming RTSP camera feeds with motion detection, alarms, and a fully interactive control dashboard.
Designed for LAN‑only setups, forensic testing, and real‑time monitoring.

🚀 Features
🎥 Live RTSP Streaming
Streams any RTSP camera using OpenCV

Low‑latency MJPEG output

Auto‑reconnect logic for unstable cameras

🧠 Motion Detection
Adjustable sensitivity

Frame‑difference–based detection

Triggers alarms and logs events

🔔 Alarm System
Play local MP3 alarms

Includes sample alarms:

allahu-akbar_1E2DAiw.mp3

sad-meow-song.mp3

Triggered automatically or manually from the dashboard

🖥️ Interactive Web Dashboard
Start/stop stream

Toggle motion detection

Trigger alarms

View real‑time logs

Clean Bootstrap UI

🛠️ Modular Codebase
app.py handles routes and logic

camera.py handles RTSP capture

static/ for JS/CSS

templates/ for UI

📦 Installation
1. Clone the repository
bash
git clone https://github.com/Mr-A-Hacker/flask_rtsp_stream_version3_stable.git
cd flask_rtsp_stream_version3_stable
2. Create a virtual environment
bash
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
▶️ Running the App
bash
python app.py
Then open:

Code
http://localhost:5000
⚙️ Configuration
Edit your RTSP URL inside app.py:

python
RTSP_URL = "rtsp://192.168.x.x:554/stream1?tcp"
📁 Project Structure
Code
flask_rtsp_stream_version3_stable/
│── app.py
│── camera.py
│── requirements.txt
│── templates/
│   └── index.html
│── static/
│   ├── script.js
│   └── style.css
└── alarms/
    ├── allahu-akbar_1E2DAiw.mp3
    └── sad-meow-song.mp3
🛡️ Notes
This project is intended for local network use only

No cloud services, no external logging

All data stays on your machine

🤝 Contributing
Pull requests are welcome.
For major changes, open an issue first to discuss what you’d like to modify.

📜 License
MIT License — free to use, modify, and distribute.
