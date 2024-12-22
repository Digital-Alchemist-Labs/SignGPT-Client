import json
import streamlit as st
from modules.text_to_sign import TextToSign
from modules.llm_chains_v2 import SignGPT_API
import yaml
import time
import numpy as np
import cv2
import mediapipe as mp
from openvino.runtime import Core

save = ['아']

# Initialize APIs and services
api = SignGPT_API(base_url="http://0.0.0.0:8000")
_TexttoSign = TextToSign(
    mapping_path="dictionary/kr-dict-mapping.json",
    url_path="dictionary/kr-dict-urls.json",
    paths_path="dictionary/kr-dict-paths.json",
    paths_path1="dictionary/kr-dict-paths1.json",
    mode="path"
)

# 전체 페이지 설정
st.set_page_config(
    page_title="Sign GPT",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일 정의
st.markdown(
    """
    <style>
    /* 전체 페이지 배경 그라데이션 */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        background: linear-gradient(to bottom, #2a3a7c, #000118);
        width: 100%;
        height: 100%;
    }

    /* 사이드바 배경 */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, rgb(116, 138, 224), rgb(51, 53, 116));
    }

    /* 사이드바 확장 버튼 스타일 */
    [data-testid="stSidebar"] #stSidebarCollapsedControl {
        color: #fff !important;
        padding: 5px;
    }

    /* 개별 박스 스타일 */
    .custom-box {
        border: 2px solid black;
        border-radius: 10px;
        padding: 10px;
        padding-bottom: 300px;
        background: white;
    }

    /* 하단 빈 공간 스타일 */
    .custom-footer {
        height: 50px;
        border: 1px solid #000;
        border-radius: 10px;
        padding-bottom: 200px;
        margin-top: 25px;
        background: #fff7e6;
    }

    [data-testid="stSidebar"] h1 {
        font-size: 40px;
        color: #fff;
        text-align: center;
    }

    .stButton > button {
        width: 100%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 4px;
        border: none;
        cursor: pointer;
        text-align: center;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .output-text {
        text-align: center;
        min-height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .spacer {
        min-height: 20px;
    }
    .timer {
        font-size: 24px;
        font-weight: bold;
        color: red;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "translated_words" not in st.session_state:
    st.session_state.translated_words = []
if "current_output_index" not in st.session_state:
    st.session_state.current_output_index = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "hands_keypoints" not in st.session_state:
    st.session_state.hands_keypoints = []
if "translation_flag" not in st.session_state:
    st.session_state.translation_flag = False
if "timer_completed" not in st.session_state:
    st.session_state.timer_completed = False
if "app_state" not in st.session_state:
    st.session_state.app_state = "waiting"
if "video_sources" not in st.session_state:
    st.session_state.video_sources = []
if "current_video_index" not in st.session_state:
    st.session_state.current_video_index = 0
if "video_frame" not in st.session_state:
    st.session_state.video_frame = None
if "camera" not in st.session_state:
    st.session_state.camera = None
if "recognized_words" not in st.session_state:
    st.session_state.recognized_words = []
if "is_recognizing" not in st.session_state:
    st.session_state.is_recognizing = False
if "recognition_interval" not in st.session_state:
    st.session_state.recognition_interval = 2

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load configuration
with open("configs/default.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

num_classes = cfg["num_classes"]
ckpt_name = cfg["ckpt_name"]

korean = [
    "", "안녕하세요", "서울", "부산", "거리", "무엇", "너"
]

# OpenVINO model loading
@st.cache_resource
def load_openvino_model():
    ie = Core()
    model_xml = "ckpts/openvino_ir/model.xml"
    model_bin = "ckpts/openvino_ir/model.bin"
    model = ie.read_model(model=model_xml, weights=model_bin)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    return compiled_model

compiled_model = load_openvino_model()
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def landmarkxy2list(landmark_list):
    keypoints = []
    for i in range(21):
        keypoints.extend([
            landmark_list.landmark[i].x,
            landmark_list.landmark[i].y,
        ])
    return keypoints

def process_video(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.right_hand_landmarks and results.left_hand_landmarks:
        if st.session_state.is_recognizing:
            keypoints_on_frame = []
            keypoints_on_frame.extend(landmarkxy2list(results.left_hand_landmarks))
            keypoints_on_frame.extend(landmarkxy2list(results.right_hand_landmarks))
            st.session_state.hands_keypoints.append(keypoints_on_frame)

        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    return image, results

def recognize_sign():
    if len(st.session_state.hands_keypoints) > 0:
        hands_keypoints = np.array(st.session_state.hands_keypoints)
        frames_len = hands_keypoints.shape[0]
        ids = np.round(np.linspace(0, frames_len - 1, 60)).astype(int)
        keypoint_sequence = hands_keypoints[ids, ...]
        input_data = keypoint_sequence.reshape(1,60,42,2).astype(np.float32)

        output = compiled_model([input_data])[output_layer]
        exp_out = np.exp(output - np.max(output))
        softmax_out = exp_out / np.sum(exp_out)

        label_index = np.argmax(softmax_out)
        confidence = softmax_out[0, label_index]
        label = korean[label_index]
        return label, confidence
    return None, 0

def play_next_video():
    if st.session_state.current_video_index < len(st.session_state.video_sources):
        word, source = st.session_state.video_sources[st.session_state.current_video_index]
        if source == "Not found":
            st.write(f"No video found for the word: {word}")
            st.session_state.current_video_index += 1
            play_next_video()
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            st.write(f"Error opening video file: {source}")
            st.session_state.current_video_index += 1
            play_next_video()
            return

        output_text_container.markdown(
            f"<div class='output-text'>수어 단어: {word}</div>",
            unsafe_allow_html=True
        )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            st.session_state.video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_output_right.image(
                st.session_state.video_frame, channels="RGB", use_container_width=True)
            time.sleep(0.006)

        cap.release()

        st.session_state.current_video_index += 1
        play_next_video()
    else:
        
        output_text_container.markdown(
            f"<div class='output-text'>수어 응답: {api.cmc_result}</div>",
            unsafe_allow_html=True
        )

def maintain_frame_rate(start_time, fps):
    frame_time = 1 / fps
    elapsed_time = time.time() - start_time
    if elapsed_time < frame_time:
        time.sleep(frame_time - elapsed_time)

def main_loop():
    st.session_state.camera = cv2.VideoCapture(0)
    st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps = 360
    last_recognition_time = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while st.session_state.camera.isOpened() and st.session_state.is_recognizing:
            start_time = time.time()

            ret, frame = st.session_state.camera.read()
            if not ret:
                st.write("Unable to read from the camera.")
                break

            frame, results = process_video(frame, holistic)
            video_output_left.image(frame, channels="BGR", use_container_width=True)

            current_time = time.time()
            if current_time - last_recognition_time >= st.session_state.recognition_interval:
                label, confidence = recognize_sign()
                left_text_container.markdown(
                    "<div class='output-text'>수어를 인식 중입니다...</div>",
                    unsafe_allow_html=True
                )
                if label:
                    
                    st.session_state.recognized_words.append(label)
                    left_text_container.markdown(
                        f"<div class='output-text'>인식된 수어: {label}</div>",
                        unsafe_allow_html=True
                    )
                st.session_state.hands_keypoints = []
                last_recognition_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            maintain_frame_rate(start_time, fps)

    st.session_state.camera.release()
    video_output_left.empty()

# Sidebar
with st.sidebar:
    st.markdown('<h1>Sign GPT</h1>', unsafe_allow_html=True)
    menu_option = st.selectbox(
        "Menu Option",
        ["Sign GPT", "Menu 1", "Menu 2", "Menu 3"]
    )

# Main container
with st.container():
    st.title("SignGPT")
    st.markdown("##### by Digital Alchemist")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("수어 입력")
        video_output_left = st.empty()
        timer_placeholder = st.empty()
        left_text_container = st.empty()
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        start_button = st.button("수어 인식 시작/종료")

    with col_right:
        st.subheader("수어 응답")
        video_output_right = st.empty()
        output_text_container = st.empty()
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

            # log.json 파일에서 데이터 읽기
    log_file_path = "log.json"
    try:
        with open(log_file_path, "r", encoding="utf-8") as log_file:
            existing_data = json.load(log_file)  # 기존 JSON 데이터를 읽어옴
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []  # 파일이 없거나 JSON 파싱 오류가 있으면 빈 리스트로 초기화

    # 로그 표시
    if existing_data:
        st.subheader("질문 로그")
        for entry in existing_data:
            question_complete = entry.get("질문완성", "데이터 없음")
            question_response = entry.get("질문응답", "데이터 없음")
            
            st.markdown(f"""
            - **질문완성:** {question_complete}
            - **질문응답:** {question_response}
            """, unsafe_allow_html=True)
    else:
        st.write("저장된 로그가 없습니다.")


# Handle start button
if start_button:
    if not st.session_state.is_recognizing:
        st.session_state.is_recognizing = True
        st.session_state.recognized_words = []
        st.session_state.hands_keypoints = []
        video_output_right.empty()
        output_text_container.empty()
    else:
        st.session_state.is_recognizing = False
        if st.session_state.recognized_words:
            recognized_text = ", ".join(st.session_state.recognized_words)
            save.append(recognized_text)

       

            words = api.sgc2(words=recognized_text)
            paths = _TexttoSign.find_videos(words)
            st.session_state.video_sources = paths
            st.session_state.current_video_index = 0
            play_next_video()
            question_data = {
            "수어입력": recognized_text,
            "질문완성": api.sfc_result,
            "질문응답": api.cmc_result,
            "수어변환": api.ssc_result
        }

        # log.json 파일에서 기존 데이터 읽기
        log_file_path = "log.json"
        try:
            with open(log_file_path, "r", encoding="utf-8") as log_file:
                existing_data = json.load(log_file)  # 기존 JSON 데이터를 읽어옴
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []  # 파일이 없거나 JSON 파싱 오류가 있으면 빈 리스트로 초기화

        # 새로운 데이터 추가
        existing_data.append(question_data)

        # JSON 파일에 저장
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            json.dump(existing_data, log_file, ensure_ascii=False, indent=4)

        if st.session_state.camera:
            st.session_state.camera.release()
        timer_placeholder.empty()

# Start the main loop if recognition is active
if st.session_state.is_recognizing:
    main_loop()