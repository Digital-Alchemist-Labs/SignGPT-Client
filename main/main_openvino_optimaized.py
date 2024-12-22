from modules.text_to_sign import TextToSign
from modules.llm_chains_v2 import SignGPT_API
# from model_dim import InferModel  # 모델을 openvino로 최적화 하였기 때문에 필요하지 않음
import yaml
# import torch
import time
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
from openvino.runtime import Core  # 오픈비노 런타임

api = SignGPT_API(base_url="http://0.0.0.0:8000")
_TexttoSign = TextToSign(
    mapping_path="dictionary/kr-dict-mapping.json",
    url_path="dictionary/kr-dict-urls.json",
    paths_path="dictionary/kr-dict-paths.json",
    paths_path1="dictionary/kr-dict-paths1.json",
    mode="path"
)

st.set_page_config(layout="wide")

# CSS 스타일
st.markdown("""
<style>
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
""", unsafe_allow_html=True)

st.title("SignGPT")
st.markdown("##### by Digital Alchemist")

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

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with open("configs/default.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

num_classes = cfg["num_classes"]
ckpt_name = cfg["ckpt_name"]

korean = [
    "", "안녕하세요", "서울", "부산", "거리", "무엇", "너"
]

# OpenVINO 모델 로드
@st.cache_resource
def load_openvino_model():
    ie = Core()
    # IR 모델 파일 경로 (pytorch -> ONNX -> IR 변환)
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
        # OpenVINO 입력을 위해 numpy float32로 변환
        input_data = keypoint_sequence.reshape(1,60,42,2).astype(np.float32)

        # OpenVINO 추론
        output = compiled_model([input_data])[output_layer]
        # 소프트맥스 적용 (모델 구조에 따라 필요 없을 수도 있음)
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
            video_output_placeholder.image(
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

left_column, right_column = st.columns(2)

with left_column:
    st.header("수어 입력")
    video_placeholder = st.empty()
    timer_placeholder = st.empty()
    left_text_container = st.empty()
    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
    start_button = st.button("수어 인식 시작/종료")

with right_column:
    st.header("수어 응답")
    video_output_placeholder = st.empty()
    output_text_container = st.empty()
    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

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
            video_placeholder.image(frame, channels="BGR", use_container_width=True)

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
    video_placeholder.empty()

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
            play_next_video()

        if st.session_state.camera:
            st.session_state.camera.release()
        timer_placeholder.empty()

if st.session_state.is_recognizing:
    main_loop()