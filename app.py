import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
model = YOLO("yolo11n.pt") 
st.title("Real-Time YOLO Object Detection on Traffic Watch Stream- Grand Bassin Parking")
STREAM_URL = "https://stream.myt.mu/prod/gbassinexch1.stream_720p/playlist.m3u8"
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    st.error("Error: Cannot open livestream")
    st.stop()
video_placeholder = st.empty()
def process_frame(frame):
    results = model(frame)
    processed_frame = results[0].plot()  
    return processed_frame

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Can't read frames from livestream")
        break
    processed_frame = process_frame(frame)
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(processed_frame_rgb)
    video_placeholder.image(image, channels="RGB", use_container_width=True)   

cap.release()
