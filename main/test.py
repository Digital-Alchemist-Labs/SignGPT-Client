import streamlit as st
import time
import numpy as np
import cv2
import mediapipe as mp
import yaml
from openvino.runtime import Core  # 오픈비노 런타임
from modules.text_to_sign import TextToSign
from modules.llm_chains_v2 import SignGPT_API

# -----------------------
# 페이지 전역 설정
# -----------------------
st.set_page_config(
    page_title="Sign GPT",  # 페이지 제목
    page_icon="🌟",         # 페이지 아이콘
    layout="wide"           # 레이아웃 설정
)

# -----------------------
# CSS 스타일 정의 (배경 및 추가 스타일)
# -----------------------
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

    /* 버튼 스타일 */
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

# -----------------------
# 사이드바 메뉴 구현
# -----------------------
with st.sidebar:
    st.markdown('<h1>Menu</h1>', unsafe_allow_html=True)
    menu_option = st.selectbox(
        "Menu Option",
        ["옵션을 선택하세요", "Menu 1", "Menu 2", "Menu 3"]
    )


# -----------------------
# 공용 세션 상태 정의
# -----------------------
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

# -----------------------
# 기능적 부분에 필요한 로직 (모델, API, 함수)
# -----------------------
api = SignGPT_API(base_url="http://0.0.0.0:8000")
_TexttoSign = TextToSign(
    mapping_path="dictionary/kr-dict-mapping.json",
    url_path="dictionary/kr-dict-urls.json",
    paths_path="dictionary/kr-dict-paths.json",
    paths_path1="dictionary/kr-dict-paths1.json",
    mode="path"
)

with open("configs/default.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

num_classes = cfg["num_classes"]

korean = [
    "", "안녕하세요", "서울", "부산", "거리", "무엇", "너"
]

# OpenVINO 모델 로드
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

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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
        if frames_len < 60:
            return None, 0

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

def play_next_video(output_text_container, video_output_placeholder):
    if st.session_state.current_video_index < len(st.session_state.video_sources):
        word, source = st.session_state.video_sources[st.session_state.current_video_index]
        if source == "Not found":
            st.write(f"No video found for the word: {word}")
            st.session_state.current_video_index += 1
            play_next_video(output_text_container, video_output_placeholder)
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            st.write(f"Error opening video file: {source}")
            st.session_state.current_video_index += 1
            play_next_video(output_text_container, video_output_placeholder)
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
            video_output_placeholder.image(
                st.session_state.video_frame, channels="RGB", use_container_width=True)
            time.sleep(0.006)

        cap.release()

        st.session_state.current_video_index += 1
        play_next_video(output_text_container, video_output_placeholder)
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

def main_loop(video_placeholder, left_text_container, fps=360):
    st.session_state.camera = cv2.VideoCapture(0)
    st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_recognition_time = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while st.session_state.camera.isOpened() and st.session_state.is_recognizing:
            start_time = time.time()

            ret, frame = st.session_state.camera.read()
            if not ret:
                st.write("Unable to read from the camera.")
                break

            frame, results = process_video(frame, holistic)
            video_placeholder.image(frame, channels="BGR", use_container_width=True)

            current_time = time.time()
            # 일정 간격마다 인식 시도
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
    video_placeholder.empty()

# -----------------------
# 메인 컨테이너
# -----------------------
with st.container():
    # 상단 영역
    col_left, col_right = st.columns(2)

    if menu_option == "Menu 1":
        # 여기서 기능적 코드 수행
        st.title("SignGPT")
        st.markdown("##### by Digital Alchemist")

        with col_left:
            st.header("수어 입력")
            video_placeholder = st.empty()
            timer_placeholder = st.empty()
            left_text_container = st.empty()
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            start_button = st.button("수어 인식 시작/종료")

        with col_right:
            st.header("수어 응답")
            video_output_placeholder = st.empty()
            output_text_container = st.empty()
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        if start_button:
            if not st.session_state.is_recognizing:
                st.session_state.is_recognizing = True
                st.session_state.recognized_words = []
                st.session_state.hands_keypoints = []
                video_output_placeholder.empty()
                output_text_container.empty()
            else:
                st.session_state.is_recognizing = False
                if st.session_state.recognized_words:
                    recognized_text = ", ".join(st.session_state.recognized_words)
                    # API 호출
                    words = api.sgc2(words=recognized_text)
                    paths = _TexttoSign.find_videos(words)
                    st.session_state.video_sources = paths
                    st.session_state.current_video_index = 0
                    play_next_video(output_text_container, video_output_placeholder)

                if st.session_state.camera:
                    st.session_state.camera.release()
                timer_placeholder.empty()

        if st.session_state.is_recognizing:
            main_loop(video_placeholder, left_text_container)

    else:
        # Menu 1 이외의 메뉴에서는 원래의 레이아웃 박스만 표시
        with col_left:
            st.markdown('<div class="custom-box">왼쪽 화면</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="custom-box">오른쪽 화면</div>', unsafe_allow_html=True)

    # 하단 빈 공간
    with st.container():
        st.markdown('<div class="custom-footer"></div>', unsafe_allow_html=True)