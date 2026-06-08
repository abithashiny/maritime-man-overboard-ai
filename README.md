# Automated Maritime Man Overboard (MOB) AI Detection System 🌊🤖

An automated maritime safety application that leverages a fine-tuned YOLOv8 computer vision model to detect individuals falling into the water. The system instantly processes live video telemetry, captures synchronized timestamps, logs simulated GPS coordinates, and triggers real-time visual and auditory dashboard alerts for immediate rescue.

## 🚀 Key Features
- **Real-Time Inference:** Low-latency video analysis loop (<50ms per frame).
- **Automated Event Logging:** Captures exact detection timestamps mapped with mock marine GPS coordinates.
- **Dynamic Alert System:** Streamlit-powered dashboard that triggers immediate sirens and visual alert states upon detection confidence ≥60%.

## 🛠️ Tech Stack
- **AI/ML:** Python, YOLOv8 (Ultralytics), OpenCV, Roboflow, Google Colab
- **Frontend/Dashboard:** Streamlit

## 📦 Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/maritime-man-overboard-ai.git](https://github.com/your-username/maritime-man-overboard-ai.git)
