import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- CONFIG & ASSETS ---
st.set_page_config(layout="wide", page_title="Live Sign Language")

# Dictionary for Text-to-Sign (Same as before)
ASL_IMAGES = {
    'a': 'https://upload.wikimedia.org/wikipedia/commons/2/27/Sign_language_a.svg',
    'b': 'https://upload.wikimedia.org/wikipedia/commons/1/18/Sign_language_b.svg',
    'c': 'https://upload.wikimedia.org/wikipedia/commons/e/e3/Sign_language_c.svg',
    'd': 'https://upload.wikimedia.org/wikipedia/commons/0/06/Sign_language_d.svg',
    'e': 'https://upload.wikimedia.org/wikipedia/commons/c/cd/Sign_language_e.svg',
    'f': 'https://upload.wikimedia.org/wikipedia/commons/8/8f/Sign_language_f.svg',
    'g': 'https://upload.wikimedia.org/wikipedia/commons/d/d9/Sign_language_g.svg',
    'h': 'https://upload.wikimedia.org/wikipedia/commons/9/97/Sign_language_h.svg',
    'i': 'https://upload.wikimedia.org/wikipedia/commons/6/6a/Sign_language_i.svg',
    'j': 'https://upload.wikimedia.org/wikipedia/commons/b/b1/Sign_language_j.svg',
    'k': 'https://upload.wikimedia.org/wikipedia/commons/9/97/Sign_language_k.svg',
    'l': 'https://upload.wikimedia.org/wikipedia/commons/d/d2/Sign_language_l.svg',
    'm': 'https://upload.wikimedia.org/wikipedia/commons/c/c4/Sign_language_m.svg',
    'n': 'https://upload.wikimedia.org/wikipedia/commons/e/e6/Sign_language_n.svg',
    'o': 'https://upload.wikimedia.org/wikipedia/commons/e/e0/Sign_language_o.svg',
    'p': 'https://upload.wikimedia.org/wikipedia/commons/0/08/Sign_language_p.svg',
    'q': 'https://upload.wikimedia.org/wikipedia/commons/d/d4/Sign_language_q.svg',
    'r': 'https://upload.wikimedia.org/wikipedia/commons/1/14/Sign_language_r.svg',
    's': 'https://upload.wikimedia.org/wikipedia/commons/3/33/Sign_language_s.svg',
    't': 'https://upload.wikimedia.org/wikipedia/commons/1/13/Sign_language_t.svg',
    'u': 'https://upload.wikimedia.org/wikipedia/commons/d/d6/Sign_language_u.svg',
    'v': 'https://upload.wikimedia.org/wikipedia/commons/3/3d/Sign_language_v.svg',
    'w': 'https://upload.wikimedia.org/wikipedia/commons/5/5c/Sign_language_w.svg',
    'x': 'https://upload.wikimedia.org/wikipedia/commons/3/30/Sign_language_x.svg',
    'y': 'https://upload.wikimedia.org/wikipedia/commons/1/1d/Sign_language_y.svg',
    'z': 'https://upload.wikimedia.org/wikipedia/commons/8/88/Sign_language_z.svg',
    ' ': 'https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg'
}

st.title("🤝 Two-Way Sign Language Translator")

tab1, tab2 = st.tabs(["📷 Live Sign Detector", "🔤 Text to Sign"])

# ==========================
# TAB 1: LIVE VIDEO (WebRTC)
# ==========================
with tab1:
    st.header("Real-Time Hand Tracking")
    st.markdown("Wait for the video to load. It might take a few seconds to connect.")

    # 1. Define the Processor Class
    class HandDetectorProcessor(VideoProcessorBase):
        def __init__(self):
            # Initialize MediaPipe Hands once
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        def recv(self, frame):
            # 2. Get the frame from the webcam
            img = frame.to_ndarray(format="bgr24")
            
            # 3. Process the frame
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            # 4. Draw Landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            # 5. Return the processed frame
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # 6. Start the Streamer
    webrtc_streamer(
        key="sign-language",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=HandDetectorProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ==========================
# TAB 2: TEXT TO SIGN
# ==========================
with tab2:
    st.header("Text to Sign Language")
    user_input = st.text_input("Enter text (A-Z only):", "").lower()
    
    if user_input:
        st.write(f"Translating: **{user_input.upper()}**")
        
        # Create a container for the images
        html_code = '<div style="display: flex; flex-wrap: wrap; gap: 10px;">'
        
        for char in user_input:
            if char in ASL_IMAGES:
                # We use HTML <img> tags directly so YOUR browser fetches them, not the server
                html_code += f'''
                <div style="text-align: center;">
                    <img src="{ASL_IMAGES[char]}" width="100" style="border-radius: 10px; border: 2px solid #333;">
                    <br><b>{char.upper()}</b>
                </div>
                '''
            elif char == " ":
                # Add a spacer for spaces
                html_code += '<div style="width: 50px;"></div>'
        
        html_code += '</div>'
        
        # Render the HTML
        st.markdown(html_code, unsafe_allow_html=True)
