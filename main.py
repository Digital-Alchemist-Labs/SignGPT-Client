from modules.text_to_sign import TextToSign
from modules.llm_chains_v2 import SignGPT_API
from model_dim import InferModel
import yaml
import torch
import time
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st

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

# 세션 상태 초기화
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
  st.session_state.app_state = "waiting"  # 'waiting', 'recognizing', 'completed'
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
  st.session_state.recognition_interval = 2  # 2초 간격으로 인식

# MediaPipe 모델 초기화
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 설정 파일 로드
with open("configs/default.yaml") as f:
  cfg = yaml.load(f, Loader=yaml.FullLoader)

num_classes = cfg["num_classes"]
ckpt_name = cfg["ckpt_name"]

# 모델 로드


@st.cache_resource
def load_model():
  model = InferModel.load_from_checkpoint(checkpoint_path=ckpt_name)
  model.eval().to("cpu")
  return model


model = load_model()

# 한국어 레이블
# korean = [
#     "행복", "안녕", "슬픔", "눈", "당신", "식사하셨어요?", "이름이 뭐예요?",
#     "사랑", "수어", "만나서 반가워요!", "다시 한번 수어를 입력해주세요!"
# ]
korean = [
    "", "안녕하세요", "서울", "부산", "거리", "무엇", "너"
]

# korean = [
#     "공수",
#     "안녕",
#     "서울",
#     "부산",
#     "거리",
#     "무엇",
#     "당신",
#     "이름",
#     "날씨",
#     "미세먼지",
#     "오늘"
# ]

# 비디오 처리 함수


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

# 랜드마크를 리스트로 변환하는 함수


def landmarkxy2list(landmark_list):
  keypoints = []
  for i in range(21):
    keypoints.extend([
        landmark_list.landmark[i].x,
        landmark_list.landmark[i].y,
    ])
  return keypoints

# 수어 인식 및 번역 함수


def recognize_sign():
  if len(st.session_state.hands_keypoints) > 0:
    hands_keypoints = torch.tensor(st.session_state.hands_keypoints)
    frames_len = hands_keypoints.shape[0]
    ids = np.round(np.linspace(0, frames_len - 1, 60)).astype(int)
    keypoint_sequence = hands_keypoints[ids]

    input_data = keypoint_sequence.view(1, 60, 42, 2).to("cpu")

    output = model(input_data)
    output = torch.softmax(output, dim=1)
    label_index = torch.max(output, dim=1)[1][0]
    confidence = torch.max(output, dim=1)[0][0]
    confidence = confidence / torch.sum(confidence)

    label = korean[label_index]
    return label, confidence.item()
  return None, 0

# 비디오 재생 함수


def play_next_video():
  if st.session_state.current_video_index < len(st.session_state.video_sources):
    word, source = st.session_state.video_sources[st.session_state.current_video_index]
    if source == "Not found":
      st.write(f"No video found for the word: {word}")
      st.session_state.current_video_index += 1
      play_next_video()  # 다음 비디오로 넘어감
      return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
      st.write(f"Error opening video file: {source}")
      st.session_state.current_video_index += 1
      play_next_video()  # 다음 비디오로 넘어감
      return

    # 영상에 해당하는 수어 단어 출력
    output_text_container.markdown(
        f"<div class='output-text'>수어 단어: {word}</div>",
        unsafe_allow_html=True
    )

    # 영상 크기를 640x480으로 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      st.session_state.video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      video_output_placeholder.image(
          st.session_state.video_frame, channels="RGB", use_container_width=True)
      time.sleep(0.006)  # 약 166 FPS로 재생

    cap.release()

    st.session_state.current_video_index += 1
    play_next_video()  # 다음 비디오 재생
  else:
    # 모든 비디오 재생이 끝났을 때
    output_text_container.markdown(
        f"<div class='output-text'>수어 응답: {api.cmc_result}</div>",
        unsafe_allow_html=True
    )


# Streamlit 앱 레이아웃
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

# 프레임 레이트 제어를 위한 함수


def maintain_frame_rate(start_time, fps):
  frame_time = 1 / fps
  elapsed_time = time.time() - start_time
  if elapsed_time < frame_time:
    time.sleep(frame_time - elapsed_time)

# 메인 루프


def main_loop():
  # st.session_state.camera = cv2.VideoCapture(0)
  st.session_state.camera = cv2.VideoCapture(0)
  st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  fps = 360  # 목표 프레임 레이트 설정
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


# 버튼 기능
if start_button:
  if not st.session_state.is_recognizing:
    st.session_state.is_recognizing = True
    st.session_state.recognized_words = []
    st.session_state.hands_keypoints = []
    # 새로운 인식 세션을 시작할 때 이전 출력을 지웁니다
    video_output_placeholder.empty()
    output_text_container.empty()
  else:
    st.session_state.is_recognizing = False
    if st.session_state.recognized_words:
      # 인식된 단어들을 문자열로 변환
      recognized_text = ", ".join(st.session_state.recognized_words)
      print(recognized_text)
      # API 호출
      words = api.sgc2(words=recognized_text)
      paths = _TexttoSign.find_videos(words)
      st.session_state.video_sources = paths
      st.session_state.current_video_index = 0
      play_next_video()

    if st.session_state.camera:
      st.session_state.camera.release()
    timer_placeholder.empty()
    # output_text_container와 video_output_placeholder를 비우지 않습니다

if st.session_state.is_recognizing:
  main_loop()
