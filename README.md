# MemeSight AI — Real-Time Gesture-Controlled Meme Engine 

**MemeSight AI** is an interactive computer-vision application that tracks facial expressions and hand gestures in real time using your webcam, instantly triggering contextual memes (GIFs, videos, or images) based on your live reactions.

Powered by **MediaPipe**, **OpenCV**, and **Python**, MemeSight AI brings computer vision and real-time interaction together into a seamless split-screen experience.

##  Key Features

###  Face Gesture Tracking

Tracks up to 468 facial mesh landmarks to trigger instant media responses:

| Gesture | Movement / Trigger Description | Triggered Meme | 
| ----- | ----- | ----- | 
|  **Tongue Out + Head Shake** | Sideways head oscillation while sticking tongue out | `freaky-orca.gif` | 
|  **Head Nod** | Rapid vertical head movement | `ishowspeed.gif` | 
|  **Idle Stare** | Neutral face looking directly at camera | `monkeytruth.jpg` | 

###  Hand Gesture Recognition

Tracks 21 key points per hand to recognize multi-hand dynamics:

| Gesture | Movement / Trigger Description | Triggered Meme | 
| ----- | ----- | ----- | 
|  **Palms Rubbing** | Rapid horizontal oscillation with close proximity | `freaky-sonic.mp4` | 
|  **Hands on Head** | Dual hand placement near top of head ("Oh no" reaction) | `ishowspeed-wow.gif` | 
|  **Index Up** | Raised index finger gesture | `monkeyrealize.jpeg` | 
|  **Hand on Chin** | Hand proximity to chin / thinking stance | `monkeythink.jpg` | 
|  **Thumbs Up** | Positive thumb elevation gesture | `thumbsupmonkey.png` | 

##  System Architecture & How It Works

```
+-----------------------------------+-----------------------------------+
|            Webcam Feed            |      Dynamic Meme / Reaction      |
|    (MediaPipe Mesh Overlay)       |       (GIF / Video / Image)       |
+-----------------------------------+-----------------------------------+
```

1. **MediaPipe FaceMesh:** Evaluates 468 3D facial landmarks for mouth open ratio, eye direction, and head pose angles.
2. **MediaPipe Hands:** Tracks 21 3D hand landmarks per hand to compute spatial proximity, finger extension, and rapid hand convergence.
3. **OpenCV Rendering:** Captures webcam frames, processes dual inputs, and renders split-screen synchronized video playback.
4. **Custom Gesture Priority Engine:** Built-in cooldown, gesture persistence verification, and priority queuing prevent erratic meme switching.

## 📂 Project Structure

```
memesight-ai/
├── memesight.py           # Main application loop & gesture engine
├── README.md              # Project documentation
└── memes/                 # Reaction media directory
    ├── freaky-orca.gif
    ├── freaky-sonic.mp4
    ├── ishowspeed.gif
    ├── ishowSpeed-wow.gif
    ├── monkeyrealize.jpeg
    ├── monkeythink.jpg
    ├── monkeytruth.jpg
    └── thumbsupmonkey.png
```

##  Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/AkashChitti/memesight-ai.git
cd memesight-ai
```

### 2️⃣ Install Dependencies

```bash
pip install opencv-python mediapipe pillow numpy
```

*(If on older OpenCV versions, install `pip install opencv-contrib-python`)*

### 3️⃣ Run MemeSight AI

```bash
python memesight.py
```

> **Note:** Ensure your webcam is enabled. Perform gestures in clear lighting for optimal tracking!

##  Customization

MemeSight AI is built modularly, making it easy to extend:

* **Add Gestures:** Define new landmark logic in `memesight.py`.
* **Custom Meme Packs:** Drop your own `.gif`, `.mp4`, or `.jpg` files into `/memes/` and update filepath mapping.
* **Sensitivity Tweaks:** Adjust cooldown thresholds and landmark distance tolerances in the configuration section.

##  Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for:

* Additional face/hand gestures
* Audio feedback / sound effects
* Custom GUI interface
* Multi-person tracking support


