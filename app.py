import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import streamlit.components.v1 as components

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Live Sign Language")

st.title("🤝 Two-Way Sign Language Translator")
tab1, tab2 = st.tabs(["📷 Live Sign Detector", "🔤 Text to Sign"])

# ==========================
# TAB 1: LIVE VIDEO (Resized + Fullscreen)
# ==========================
with tab1:
    st.header("Real-Time Hand Tracking")
    
    # Layout: [Spacer, Video, Spacer] to center and resize the video
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.caption("Press 'f' for Fullscreen | 'Esc' to Exit")

        class HandDetectorProcessor(VideoProcessorBase):
            def __init__(self):
                self.mp_hands = mp.solutions.hands
                self.mp_draw = mp.solutions.drawing_utils
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5
                )

            def detect_gesture(self, landmarks):
                # Finger states (1 = Open, 0 = Closed)
                thumb_tip = landmarks[4].x
                thumb_ip = landmarks[3].x
                index_tip = landmarks[8].y
                index_pip = landmarks[6].y
                middle_tip = landmarks[12].y
                middle_pip = landmarks[10].y
                ring_tip = landmarks[16].y
                ring_pip = landmarks[14].y
                pinky_tip = landmarks[20].y
                pinky_pip = landmarks[18].y

                fingers = []
                # Thumb logic (Simple X-axis check for right hand)
                fingers.append(1 if thumb_tip < thumb_ip else 0)
                fingers.append(1 if index_tip < index_pip else 0)
                fingers.append(1 if middle_tip < middle_pip else 0)
                fingers.append(1 if ring_tip < ring_pip else 0)
                fingers.append(1 if pinky_tip < pinky_pip else 0)

                # Gesture Rules
                if fingers == [0, 1, 1, 0, 0]:
                    return "VICTORY (V)"
                elif fingers == [1, 1, 1, 1, 1]:
                    return "HELLO / HIGH FIVE"
                elif fingers == [0, 0, 0, 0, 0]:
                    return "FIST / ROCK"
                elif fingers == [1, 1, 0, 0, 1]:
                    return "I LOVE YOU"
                elif fingers == [0, 1, 0, 0, 0]:
                    return "ONE"
                else:
                    return ""

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                
                gesture_text = ""
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        gesture_text = self.detect_gesture(hand_landmarks.landmark)
                        
                        # Draw Text (Black Border + Green Text)
                        cv2.putText(img, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                        cv2.putText(img, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Start the video streamer inside the centered column
        webrtc_streamer(
            key="sign-language",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=HandDetectorProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # Javascript for Fullscreen ('f' key)
    components.html(
        """
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.key === 'f' || e.key === 'F') {
                const video = parent.document.querySelector('video');
                if (video) {
                    if (video.requestFullscreen) { video.requestFullscreen(); }
                    else if (video.webkitRequestFullscreen) { video.webkitRequestFullscreen(); }
                    else if (video.msRequestFullscreen) { video.msRequestFullscreen(); }
                }
            }
        });
        </script>
        """,
        height=0, width=0
    )

# ==========================
# TAB 2: TEXT TO SIGN
# ==========================
with tab2:
    st.header("Text to Sign Language")
    st.write("Enter a word to translate it into sign language.")
    
    # Use reliable GitHub-hosted images to avoid 403 errors
    BASE_URL = "https://raw.githubusercontent.com/cloud-computer/ASL/master/data/asl_alphabet_train/asl_alphabet_train"
    
    user_input = st.text_input("Type here (A-Z):", "").upper()
    
    if user_input:
        cols = st.columns(6) # Grid layout
        for i, char in enumerate(user_input):
            if 'A' <= char <= 'Z':
                img_url = f"{BASE_URL}/{char}/{char}1.jpg"
                cols[i % 6].image(img_url, caption=char, width=100)
            elif char == " ":
                cols[i % 6].write("   ")
