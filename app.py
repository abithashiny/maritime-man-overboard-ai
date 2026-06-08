import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# 1. Web Page Configuration
st.set_page_config(page_title="Maritime MOB Detection", layout="centered")
st.title("⚓ Maritime Man Overboard (MOB) Intelligent Detection System")
st.write("Upload a video or image feed to monitor the marine environment in real-time.")

# 2. Load the AI model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 3. File Uploader Layout
uploaded_file = st.file_uploader("Choose an ocean video/image stream...", type=["mp4", "jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    # Handle image reading safely across formats
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Run the AI engine
    results = model.predict(image, conf=0.40) # Lowered base confidence to 40% to be completely safe
    
    trigger_alarm = False
    max_confidence = 0.0
    
    # 4. Check detections
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Get the class ID number and confidence rating
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # In your dataset dashboard, class 1 is explicitly 'human'
                # We also check if the text string contains 'human' just in case
                class_name = model.names[class_id].lower() if class_id in model.names else ""
                
                if class_id == 1 or "human" in class_name:
                    trigger_alarm = True
                    if confidence > max_confidence:
                        max_confidence = confidence

    # 5. Force UI Emergency Render if a human is detected
    if trigger_alarm:
        st.error(f"🚨 [EMERGENCY ALERT]: MAN OVERBOARD DETECTED! Max Confidence: {max_confidence*100:.1f}%")
        st.warning("📍 Generating GPS coordinates... Activating bridge sirens.")
        
        # Audio Siren Element (Requires your laptop volume to be turned up!)
        siren_html = """
            <audio autoplay loop>
                <source src="https://www.soundjay.com/buttons/sounds/alarm-clock-01.mp3" type="audio/mp3">
            </audio>
        """
        st.components.v1.html(siren_html, height=0)
    else:
        st.success("✅ System Monitoring Active: No immediate hazards detected in current frame.")

    # 6. Always render the image with the boxes drawn on it
    res_plotted = results[0].plot()
    st.image(res_plotted, channels="BGR", caption="Processed Live Feed Dashboard")