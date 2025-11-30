import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# --- CONFIG & ASSETS ---
st.set_page_config(layout="wide", page_title="Two-Way Sign Language")

# Publicly available ASL images (Using a reliable GitHub source)
# You can replace these URLs with your own images later if you want specific styles
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
    ' ': 'https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg' # Placeholder for space
}

st.title("🤝 Two-Way Sign Language Translator")

# --- TABS FOR NAVIGATION ---
tab1, tab2 = st.tabs(["📷 Detect Signs (Camera)", "🔤 Text to Sign"])

# ==========================
# TAB 1: CAMERA DETECTION
# ==========================
with tab1:
    st.header("Sign Language Detector")
    st.text("Show your hand to the camera to see the skeleton tracking.")

    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            st.success("Hand Detected! Processing skeleton...")
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            st.warning("No hand detected. Try moving closer or checking lighting.")

        st.image(img, channels="BGR")

# ==========================
# TAB 2: TEXT TO SIGN
# ==========================
with tab2:
    st.header("Text to Sign Language")
    st.text("Type a word below to see how to sign it.")
    
    user_input = st.text_input("Enter text (A-Z only):", "").lower()
    
    if user_input:
        st.write(f"Translating: **{user_input.upper()}**")
        
        # Display images in a row
        cols = st.columns(len(user_input))
        
        for i, char in enumerate(user_input):
            if char in ASL_IMAGES:
                with cols[i]:
                    st.image(ASL_IMAGES[char], caption=char.upper(), width=100)
            elif char == " ":
                with cols[i]:
                    st.write("  ") # Empty space
            else:
                st.error(f"Character '{char}' not found.")
