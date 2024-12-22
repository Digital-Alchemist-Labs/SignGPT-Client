import streamlit as st
from modules.text_to_sign import TextToSign
from modules.llm_chains_v2 import SignGPT_API
import cv2
import time
import yaml
import mediapipe as mp
import numpy as np

# API 및 TextToSign 초기화
api = SignGPT_API(base_url="http://0.0.0.0:8000")
_TexttoSign = TextToSign(
    mapping_path="dictionary/kr-dict-mapping.json",
    url_path="dictionary/kr-dict-urls.json",
    paths_path="dictionary/kr-dict-paths.json",
    paths_path1="dictionary/kr-dict-paths1.json",
    mode="path"
)

# Streamlit 앱의 기본 설정
st.set_page_config(layout="wide")

# CSS 스타일
st.markdown("""
<style>
		.big-font {
				font-size:30px !important;
		}
    .stButton > button {
        width: 100%;
        height: 100px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: #007BFF;
        color: #fff;
        font-size: 24px;
        padding: 12px 28px;
        border-radius: 6px;
        border: none;
        cursor: pointer;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    .output-text {
        text-align: center;
        min-height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size : 25px;
    }
    .spacer {
        min-height: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("SignGPT")
st.markdown("##### by Digital Alchemist")

# MediaPipe Holistic 모델 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 세션 상태 초기화
if "recognized_words" not in st.session_state:
    st.session_state.recognized_words = []
if "current_video_index" not in st.session_state:
    st.session_state.current_video_index = 0
if "video_sources" not in st.session_state:
    st.session_state.video_sources = []
if "video_frame" not in st.session_state:
    st.session_state.video_frame = None

# 설정 파일 로드
with open("configs/default.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# 한국어 레이블 (수어 단어 목록)
korean = [
    "행복", "안녕하세요", "슬픔", "눈", "당신", "식사하셨어요?", "이름이 뭐예요?",
    "사랑", "수어", "만나서 반가워요!", "다시 한번 수어를 입력해주세요!"
]

# 키포인트 출력 함수 (입력용)
def process_video_with_keypoints(video_file, video_placeholder, holistic):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        video_placeholder.image(image, channels="RGB", use_column_width=True)
        time.sleep(0.0001)
    cap.release()

# 비디오 재생 함수 (키포인트 없이, 응답용)
def play_video_without_keypoints(video_file, video_placeholder):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(image, channels="RGB", use_column_width=True)
        time.sleep(0.006)
    cap.release()

# 입력 비디오 재생 함수 (키포인트 포함)
def play_input_video(word, output_placeholder):
    paths = _TexttoSign.find_videos1(word)
    if paths and paths[0][1] != "Not found":
        word, source = paths[0]
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            process_video_with_keypoints(source, output_placeholder, holistic)
    else:
        st.write(f"No video found for the word: {word}")

# 메인 레이아웃
st.write("수어 단어를 선택하고 '질문 하기' 버튼을 눌러주세요.")

left_column, right_column = st.columns(2)

with left_column:
    st.header("인식된 수어")
    left_video_placeholder = st.empty()
    left_text_container = st.empty()

with right_column:
    st.header("수어 응답")
    right_video_placeholder = st.empty()
    right_text_container = st.empty()

# 선택된 단어 표시 및 비디오 재생
if st.session_state.recognized_words:
    left_text_container.markdown(
        f"<div class='output-text'>인식된 수어: {', '.join(st.session_state.recognized_words)}</div>",
        unsafe_allow_html=True
    )
    play_input_video(st.session_state.recognized_words[-1], left_video_placeholder)

# 다음 비디오 재생 함수 (응답용, 키포인트 없음)
def play_next_video(output_placeholder):
    if st.session_state.current_video_index < len(st.session_state.video_sources):
        word, source = st.session_state.video_sources[st.session_state.current_video_index]
        if source == "Not found":
            st.write(f"No video found for the word: {word}")
            st.session_state.current_video_index += 1
            play_next_video(output_placeholder)
            return

        right_text_container.markdown(
            f"<div class='output-text'>수어 단어: {word}</div>",
            unsafe_allow_html=True
        )

        play_video_without_keypoints(source, output_placeholder)

        st.session_state.current_video_index += 1
        play_next_video(output_placeholder)
    else:
        right_text_container.markdown(
            f"<div class='output-text'>수어 응답: {api.cmc_result}</div>",
            unsafe_allow_html=True
        )

# '질문 하기' 버튼
if st.button("질문 하기"):
    if st.session_state.recognized_words:
        recognized_text = ", ".join(st.session_state.recognized_words)
        words = api.sgc2(words=recognized_text)
        paths = _TexttoSign.find_videos(words)
        st.session_state.video_sources = paths
        st.session_state.current_video_index = 0
        st.session_state.recognized_words = []
        play_next_video(right_video_placeholder)
    else:
        st.write("선택된 수어 단어가 없습니다.")

# 버튼 생성 (페이지 맨 아래)
st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
st.markdown("### 수어 단어 선택")
cols = st.columns(4)
for i, word in enumerate(korean):
    with cols[i % 4]:
        # if st.button(f"<h1>{word}</h1>", key=word):
        if st.button(word):
            st.session_state.recognized_words.append(word)
            play_input_video(word, left_video_placeholder)
            left_text_container.markdown(
                f"<div class='output-text'>인식된 수어: {', '.join(st.session_state.recognized_words)}</div>",
                unsafe_allow_html=True
            )