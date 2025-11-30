# 🤝 Two-Way Sign Language Translator

<div align="center">

  ![Status](https://img.shields.io/badge/Status-Under_Construction-yellow?style=for-the-badge)
  ![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

  <h3>🚧 Project in Active Development 🚧</h3>
  
  <p>
    <b>Current State:</b> Functional Prototype.<br>
    The core logic works, but the UI and AI model are currently being polished.
  </p>

  [**🔴 Live Demo**](https://share.streamlit.io/5uhag/realtime-sign-language-decoder/main/app.py)

</div>

---

## 🧐 What is this?
A two-way communication tool designed to bridge the gap between Sign Language users and non-users.
1.  **Sign-to-Text:** Uses your camera to detect Hand Sign Language (ASL) and translates it to text in real-time.
2.  **Text-to-Sign:** You type a word, and it translates it into Sign Language images.

## 🛠 Features (Current Status)

| Feature | Status | Notes |
| :--- | :---: | :--- |
| **Hand Tracking** | ✅ Working | Uses MediaPipe for skeleton detection. |
| **Alphabet Recognition** | ⚠️ Basic | Uses a Random Forest classifier. Works best with Right Hand + Good Lighting. |
| **Text-to-Sign** | ✅ Working | Instant translation using local SVG assets. |
| **UI / UX** | 🚧 Polishing | Dark mode enabled, mobile-responsive layout in progress. |

## 💻 Tech Stack
* **Frontend:** Streamlit (Cloud)
* **Computer Vision:** OpenCV, MediaPipe
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Deployment:** Streamlit Community Cloud

## 🚀 How to Run Locally
If you want to test this on your own machine:

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/5uhag/realtime-sign-language-decoder.git](https://github.com/5uhag/realtime-sign-language-decoder.git)
    cd realtime-sign-language-decoder
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## 📝 To-Do List
- [ ] Improve Model Accuracy (Train on larger dataset)
- [ ] Add "Dynamic Gesture" support (Moving signs like 'J' or 'Z')
- [ ] Better UI styling for mobile users
- [ ] Add Sentence Formation logic (Auto-correct)

---

<div align="center">
  <b>Built by <a href="https://github.com/5uhag">@5uhag</b>
href="https://github.com/ningaraj-mw">@Ningaraj</a></b>
</div>
