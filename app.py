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

# --- LOAD MODEL WITH DEBUGGING ---
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
    load_status = f"Error: {str(e)[:20]}" # Show first 20 chars of error

st.title("🤝 Two-Way Sign Language Translator")
tab1, tab2 = st.tabs(["📷 Live Sign Detector", "🔤 Text to Sign"])

with tab1:
    st.header("Real-Time Hand Tracking")
    
    # Show status at the top
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
                
                # Default status text on video
                status_text = f"Model: {load_status}"
                color = (0, 0, 255) # Red

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Data Prep
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

                        # DEBUGGING LOGIC
                        if model:
                            try:
                                # We try predicting regardless of shape to see the error
                                prediction = model.predict([data_aux])
                                predicted_char = prediction[0]
                                
                                # If successful, draw Green Box
                                cv2.rectangle(img, (0, 0), (160, 60), (0, 0, 0), -1)
                                cv2.putText(img, predicted_char, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                                status_text = "Predicting"
                                color = (0, 255, 0)
                            except Exception as e:
                                # If shape doesn't match, print the required shape
                                status_text = f"Shape Err: Needs {getattr(model, 'n_features_in_', '?')}"
                                color = (0, 255, 255) # Yellow
                        else:
                            status_text = "No Model"

                # Draw Debug Text on bottom of video
                cv2.putText(img, status_text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(key="sign-language", mode=WebRtcMode.SENDRECV, video_processor_factory=HandDetectorProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)

# ==========================
# TAB 2: TEXT TO SIGN (HTML HACK)
# ==========================
with tab2:
    st.header("Text to Sign Language")
    st.write("Enter a word to translate it into sign language.")
    
    user_input = st.text_input("Type here (A-Z):", "").lower()
    
    if user_input:
        # Start HTML Container
        html_code = '<div style="display: flex; flex-wrap: wrap; gap: 15px; justify-content: center;">'
        
        for char in user_input:
            if char in ASL_IMAGES:
                # Direct HTML Image Injection
                html_code += f'''
                <div style="text-align: center; margin: 5px;">
                    <img src="{ASL_IMAGES[char]}" width="120" style="border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.3);">
                    <br><b style="color: white; font-size: 20px;">{char.upper()}</b>
                </div>
                '''
            elif char == " ":
                html_code += '<div style="width: 50px;"></div>' # Spacer
        
        html_code += '</div>'
        
        # RENDER HTML (This flag is the key!)
        st.markdown(html_code, unsafe_allow_html=True)
