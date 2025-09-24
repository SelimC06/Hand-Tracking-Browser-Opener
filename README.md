# Hand Gesture Web Browser Opener

A simple computer vision project that detects hand gestures from a webcam feed and maps them to browser actions, such as opening a specific website or navigating tabs.

## ✨ Features
- Real-time hand tracking with **MediaPipe Hands**  
- Gesture recognition (e.g., **open palm**, **fist**, **point**)  
- Map gestures to browser actions (e.g., open YouTube, Google, or GitHub)  
- Lightweight, runs locally with just Python + webcam  

---

## 📦 Tech Stack
- **Python 3.9+**  
- **MediaPipe** – hand tracking  
- **OpenCV** – video capture & visualization  
- **Webbrowser (Python stdlib)** – open websites  
- **PyAutoGUI** (optional) – simulate keyboard/mouse actions  

---

## 🚀 Usage

1. **Clone the repository**  
   ```
   git clone https://github.com/your-username/hand-gesture-web-browser.git
   cd hand-gesture-web-browser

2. **Install dependencies**
    ```
    pip install -r requirements.txt

3. **Run the script**
    ```
    python Test-Model.py

4. **Use gestures to control the browser**
    * Open Palm → Opens Google
    * Fist → Opens YouTube
    * Point → Opens GitHub