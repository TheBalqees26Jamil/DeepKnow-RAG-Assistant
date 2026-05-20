import streamlit as st
import base64
from pathlib import Path
from retrieval.retriever import load_embeddings, build_faiss_index, search
from llm.llm_client import generate_answer
from safety.guardrails import is_safe_query


st.set_page_config(
    page_title="Deep Learning Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


if "page" not in st.session_state:
    st.session_state.page = "home"

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_path = Path(r"C:\Users\DELL\Desktop\Develop_myself\Projects\DeepKnow_RAG_Assistant\lucid.jpg")

if st.session_state.page == "home":

    st.markdown('''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
        
        #MainMenu, header, footer, .stDeployButton {display: none !important;}
        
        .stApp {
            background: #000000;
            font-family: 'Inter', sans-serif;
            overflow: hidden;
        }

        .main-wrapper {
            position: relative;
            height: 100vh;
            width: 100%;
            display: flex;
            align-items: center;
            padding: 0 8%;
        }

        .main-title {
            font-family: 'Inter', sans-serif;
            font-size: clamp(5rem, 12vw, 10rem);
            font-weight: 900;
            color: #FFFFFF;
            line-height: 0.95;
            letter-spacing: -4px;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
            text-shadow: 
                0 0 5px #8B5CF6,
                0 0 10px #8B5CF6,
                0 0 20px #8B5CF6,
                0 0 40px #8B5CF6,
                0 0 80px rgba(139, 92, 246, 0.5),
                0 4px 30px rgba(0,0,0,0.3);
            position: relative;
            z-index: 10;
            display: block;
        }

        div[data-testid="stButton"] {
            position: fixed !important;
            left: 8% !important;
            top: 70% !important;
            z-index: 100 !important;
        }

        div[data-testid="stButton"] > button {
            opacity: 0 !important;
            width: 280px !important;
            height: 75px !important;
            cursor: pointer !important;
        }

        .custom-start-btn {
            display: inline-flex;
            align-items: center;
            background: #FFFFFF;
            border: none;
            border-radius: 50px;
            padding: 8px 8px 8px 45px;
            transition: all 0.3s ease;
            box-shadow: 
                0 0 25px rgba(139, 92, 246, 0.6),
                0 0 50px rgba(139, 92, 246, 0.4),
                0 8px 32px rgba(0,0,0,0.4);
            position: fixed;
            left: 8%;
            top: 70%;
            z-index: 99;
            pointer-events: none;
        }

        .custom-start-btn:hover {
            transform: translateY(-3px);
            box-shadow: 
                0 0 35px rgba(139, 92, 246, 0.9),
                0 0 70px rgba(139, 92, 246, 0.6),
                0 12px 40px rgba(0,0,0,0.5);
        }

        .custom-start-btn-text {
            color: #333;
            font-family: 'Inter', sans-serif;
            font-size: 1.4rem;
            font-weight: 600;
            margin-right: 20px;
        }

        .custom-start-btn-arrow {
            background: #2D2D3A;
            color: #FFFFFF;
            padding: 18px 40px;
            border-radius: 50px;
            font-family: 'Inter', sans-serif;
            font-size: 1.3rem;
            font-weight: 600;
        }

        .neon-border {
            position: fixed;
            left: 8%;
            top: 70%;
            width: 280px;
            height: 75px;
            border-radius: 50px;
            border: 2px solid rgba(139, 92, 246, 0.6);
            box-shadow: 
                0 0 10px rgba(139, 92, 246, 0.8),
                0 0 20px rgba(139, 92, 246, 0.4),
                inset 0 0 10px rgba(139, 92, 246, 0.2);
            z-index: 98;
            pointer-events: none;
            animation: neonPulse 2s ease-in-out infinite;
        }

        @keyframes neonPulse {
            0%, 100% { 
                box-shadow: 
                    0 0 10px rgba(139, 92, 246, 0.8),
                    0 0 20px rgba(139, 92, 246, 0.4),
                    inset 0 0 10px rgba(139, 92, 246, 0.2);
            }
            50% { 
                box-shadow: 
                    0 0 20px rgba(139, 92, 246, 1),
                    0 0 40px rgba(139, 92, 246, 0.6),
                    inset 0 0 20px rgba(139, 92, 246, 0.4);
            }
        }

        .decor-circle-1 {
            position: fixed;
            top: 8%;
            right: 12%;
            width: 50px;
            height: 50px;
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 12px;
            box-shadow: 0 0 15px rgba(139, 92, 246, 0.2);
        }

        .decor-circle-2 {
            position: fixed;
            top: 30%;
            right: 8%;
            width: 60px;
            height: 60px;
            border: 2px solid rgba(139, 92, 246, 0.2);
            border-radius: 50%;
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.15);
        }

        .decor-circle-3 {
            position: fixed;
            top: 55%;
            right: 15%;
            width: 20px;
            height: 20px;
            background: rgba(139, 92, 246, 0.3);
            border-radius: 50%;
            box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
        }

        .robot-image {
            position: fixed;
            right: -5%;
            bottom: -10%;
            height: 110vh;
            width: auto;
            object-fit: contain;
            pointer-events: none;
            z-index: 1;
        }

        .dark-overlay {
            position: fixed;
            top: 0;
            right: 0;
            width: 50%;
            height: 100vh;
            background: linear-gradient(180deg, 
                rgba(0,0,0,0.2) 0%, 
                rgba(0,0,0,0.5) 50%, 
                rgba(0,0,0,0.9) 100%
            );
            z-index: 0;
        }

        .content-section {
            position: relative;
            z-index: 10;
            max-width: 900px;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }

        .float-anim {
            animation: float 4s ease-in-out infinite;
        }
    </style>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="decor-circle-1"></div>', unsafe_allow_html=True)
    st.markdown('<div class="decor-circle-2"></div>', unsafe_allow_html=True)
    st.markdown('<div class="decor-circle-3"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dark-overlay"></div>', unsafe_allow_html=True)

    img_base64 = get_image_base64(image_path)
    st.markdown(f'''<img src="data:image/jpeg;base64,{img_base64}" class="robot-image float-anim" alt="AI Robot">''', unsafe_allow_html=True)

    st.markdown('<div class="neon-border"></div>', unsafe_allow_html=True)

    #
    col1, col2, col3 = st.columns([1, 3, 6])
    with col1:
        if st.button("Get Started", key="start_btn"):
            st.session_state.page = "main"
            st.rerun()

    st.markdown('''
    <div class="main-wrapper">
        <div class="content-section">
            <div class="main-title">DEEP LEARNING</div>
            <div class="main-title">ASSISTANT</div>
            <div class="custom-start-btn">
                <span class="custom-start-btn-text">Get Started</span>
                <div class="custom-start-btn-arrow">→</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)



elif st.session_state.page == "main":

    
    st.markdown("""
    <style>

    body {
        background-color: #000000;
        color: white;
    }

    .stApp {
        background-color: #000000 !important;
    }

    /* ========== CONTAINER NARROW ========== */
    /* Main content block - make it narrower */
    .block-container {
        max-width: 700px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* ========== CENTER TITLE ========== */
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
        text-shadow: 
            0 0 5px #a855f7,
            0 0 10px #a855f7,
            0 0 20px #a855f7,
            0 0 40px #a855f7,
            0 0 80px rgba(168, 85, 247, 0.5);
    }

    /* ========== SUBTITLE CENTERED ========== */
    .subtitle,
    .stApp .subtitle,
    .stMarkdown .subtitle,
    div[data-testid="stMarkdown"] .subtitle {
        text-align: center;
        font-size: 16px;
        opacity: 0.7;
        margin-bottom: 20px;
        color: #ffffff !important;
    }

    /* ========== HEADINGS ========== */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    /* ========== LABELS LEFT-ALIGNED ========== */
    label, .stTextInput label {
        color: white !important;
        text-align: left !important;
    }

    /* ========== CHECKBOX LABEL ========== */
    .stCheckbox label,
    .stApp .stCheckbox label,
    .stCheckbox > label,
    .stCheckbox div[data-testid="stMarkdownContainer"] p,
    .stCheckbox span {
        color: #ffffff !important;
        text-shadow: 
            0 0 5px #a855f7,
            0 0 10px #a855f7,
            0 0 20px rgba(168, 85, 247, 0.5);
    }

    .stCheckbox input[type="checkbox"] {
        accent-color: #a855f7 !important;
        background-color: white !important;
        border: 2px solid #a855f7 !important;
        cursor: pointer;
    }

    .stCheckbox input[type="checkbox"]:checked {
        accent-color: #a855f7 !important;
        background-color: #a855f7 !important;
    }

    /* ========== SPINNER TEXT ========== */
    .stSpinner > div {
        color: white !important;
    }

    /* ========== INPUT FIELD (SHORT) ========== */
    .stTextInput input {
        background-color: #ffffff !important;
        color: black !important;
        border: 2px solid #a855f7 !important;
        border-radius: 10px;
        box-shadow: 0 0 10px #a855f7;
        max-width: 100% !important;
    }

    /* ========== BUTTON ========== */
    .stButton > button {
        background-color: white;
        color: black;
        border-radius: 10px;
        transition: 0.3s;
        border: 2px solid #a855f7;
        box-shadow: 0 0 10px #a855f7;
    }

    .stButton > button:hover {
        background-color: #a855f7;
        color: white;
        box-shadow: 0 0 20px #a855f7;
    }

    /* ========== ANSWER BOX (SHORT) ========== */
    .answer-box {
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #a855f7;
        box-shadow: 0 0 15px #a855f7;
        margin-top: 10px;
        background-color: #ffffff;
        color: black;
        max-width: 100%;
    }

    /* ========== CHUNKS BOX (SHORT) ========== */
    .chunk-box {
        padding: 12px;
        border-radius: 10px;
        border: 2px solid #a855f7;
        margin-bottom: 10px;
        box-shadow: 0 0 10px #a855f7;
        background-color: #ffffff;
        color: black;
        max-width: 100%;
    }

    /* ========== LEFT ALIGN SECTION HEADINGS ========== */
    .section-heading {
        color: white !important;
        text-align: left !important;
        margin-left: 0 !important;
        padding-left: 0 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    
    @st.cache_resource
    def load_system():
        data = load_embeddings()
        index, _ = build_faiss_index(data)
        return data, index

    data, index = load_system()

   
    st.markdown("<div class='title'>Deep Learning RAG Assistant</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='subtitle'>Ask anything about your knowledge base</div>",
        unsafe_allow_html=True
    )

   
    query = st.text_input("Enter your question:")

    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        ask = st.button("Ask", use_container_width=True)

 
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        show_chunks = st.checkbox("Show chunks")

    
    if ask and query:

        if not is_safe_query(query):
            st.error("🚫 This query contains blocked content and cannot be processed.")
            st.stop()

        with st.spinner("Thinking..."):

            contexts = search(query, index, data, k=3)
            
            answer = generate_answer(query, contexts)

        
        st.markdown("<h3 class='section-heading'>Answer</h3>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='answer-box'>{answer}</div>",
            unsafe_allow_html=True
        )

        
        if show_chunks:
            st.markdown("<h3 class='section-heading'>Retrieved Chunks</h3>", unsafe_allow_html=True)

            for i, ctx in enumerate(contexts):
                st.markdown(
                    f"""
                    <div class='chunk-box'>
                    <b>Chunk {i+1}</b><br>
                    <b>File:</b> {ctx['file_name']}<<br><br>
                    {ctx['text']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )