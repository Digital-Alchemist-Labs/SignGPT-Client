import streamlit as st

# 전체 페이지 설정
st.set_page_config(
    page_title="Sign GPT",  # 페이지 제목
    page_icon="🌟",                   # 페이지 아이콘
    layout="wide"                     # 레이아웃 설정
)

# CSS 스타일 정의 (배경 포함)
st.markdown(
    """
    <style>
    /* 전체 페이지 배경 그라데이션 */

    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        background: linear-gradient(to bottom, #2a3a7c, #000118); /* 배경 그라데이션 */
        width: 100%;
        height: 100%;
    }

    /* 사이드바 배경 */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, rgb(116, 138, 224), rgb(51, 53, 116));
    }

    /* 사이드바 확장 버튼 스타일 */
    [data-testid="stSidebar"] #stSidebarCollapsedControl {
        color: #fff !important; /* 아이콘 색상 */
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
    </style>
    """,
    unsafe_allow_html=True,
)

# 사이드바 메뉴 구현
with st.sidebar:
    st.markdown('<h1>Menu</h1>', unsafe_allow_html=True)
    menu_option = st.selectbox(
        "Menu Option",
        ["옵션을 선택하세요", "Menu 1", "Menu 2", "Menu 3"]
    )

# 메인 컨테이너 구현
with st.container():

    # 상단 영역
    with st.container():
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="custom-box">왼쪽 화면</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="custom-box">오른쪽 화면</div>', unsafe_allow_html=True)

    # 하단 빈 공간
    with st.container():
        st.markdown('<div class="custom-footer"></div>', unsafe_allow_html=True)
