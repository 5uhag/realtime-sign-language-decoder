import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import streamlit.components.v1 as components
import pickle
import warnings
import os

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Live Sign Language")

# --- LOAD MODEL ---
model = None
load_status = "Starting..."
try:
    if os.path.exists('./model.p'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']
        load_status = "Success"
    else:
        load_status = "File Missing"
except Exception as e:
    load_status = f"Error: {str(e)[:20]}"

st.title("🤝 Two-Way Sign Language Translator")
tab1, tab2 = st.tabs(["📷 Live Sign Detector", "🔤 Text to Sign"])

# ==========================
# TAB 1: LIVE AI PREDICTION
# ==========================
with tab1:
    st.header("Real-Time Hand Tracking")
    if load_status == "Success":
        st.success("Brain Loaded! System Ready.")
    else:
        st.error(f"Brain Status: {load_status}")
        
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        class HandDetectorProcessor(VideoProcessorBase):
            def __init__(self):
                self.mp_hands = mp.solutions.hands
                self.mp_draw = mp.solutions.drawing_utils
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5
                )

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                
                predicted_char = ""
                status_text = "Wait..."
                color = (0, 0, 255)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        data_aux = []
                        x_ = []
                        y_ = []
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            y_.append(y)
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                        if model:
                            try:
                                prediction = model.predict([data_aux])
                                predicted_char = prediction[0]
                                cv2.rectangle(img, (0, 0), (160, 60), (0, 0, 0), -1)
                                cv2.putText(img, predicted_char, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                                status_text = "Active"
                                color = (0, 255, 0)
                            except:
                                status_text = "Shape Err"
                                color = (0, 255, 255)

                cv2.putText(img, status_text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(key="sign-language", mode=WebRtcMode.SENDRECV, video_processor_factory=HandDetectorProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)

    components.html(
        """<script>
        document.addEventListener('keydown', function(e) {
            if (e.key === 'f' || e.key === 'F') {
                const video = parent.document.querySelector('video');
                if (video) { video.requestFullscreen(); }
            }
        });
        </script>""",
        height=0, width=0
    )

# ==========================
# TAB 2: LOCAL SVG FILES
# ==========================
with tab2:
    st.header("Text to Sign Language")
    st.write("Enter a word to translate it into sign language.")
    
    user_input = st.text_input("Type here (A-Z):", "").lower()
    
    if user_input:
        cols = st.columns(6)
        for i, char in enumerate(user_input):
            if 'a' <= char <= 'z':
                # LOOK FOR .svg FILES IN THE 'asl_images' FOLDER
                img_path = f"asl_images/{char}.svg"
                
                if os.path.exists(img_path):
                    cols[i % 6].image(img_path, caption=char.upper(), width=100)
                else:
                    cols[i % 6].error(f"Missing: {char.upper()}")
            elif char == " ":
                cols[i % 6].write("   ")
