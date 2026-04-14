import os
import streamlit as st

from src.config import GROQ_MODEL, CHROMA_PERSIST_DIR, DOCS_PATH, TOP_K
from src.pdf_loader import load_documents
from src.chunker import split_documents_into_chunks
from src.vector_store import build_vector_store
from src.rag_pipeline import ask_rag


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="✨",
    layout="wide"
)


# =========================================================
# CUSTOM CSS (DARK PREMIUM UI)
# =========================================================
st.markdown("""
<style>
/* ---------- GLOBAL ---------- */
.stApp {
    background:
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.08), transparent 28%),
        radial-gradient(circle at top left, rgba(99, 102, 241, 0.08), transparent 30%),
        linear-gradient(180deg, #0b1020 0%, #111827 45%, #0f172a 100%);
    color: #e5e7eb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(17, 24, 39, 0.92);
    border-right: 1px solid rgba(148, 163, 184, 0.14);
}

/* General text */
h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: #e5e7eb;
}

/* ---------- HERO ---------- */
.hero-card {
    background: rgba(15, 23, 42, 0.78);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 28px;
    padding: 30px 32px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
    margin-bottom: 1.25rem;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
}

.hero-subtitle {
    font-size: 1rem;
    color: #cbd5e1;
    line-height: 1.8;
    max-width: 920px;
}

.pill-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}

.pill {
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid rgba(125, 211, 252, 0.18);
    color: #dbeafe;
    border-radius: 999px;
    padding: 8px 14px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

/* ---------- CARDS ---------- */
.glass-card {
    background: rgba(15, 23, 42, 0.76);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.22);
    margin-bottom: 1rem;
}

.answer-card {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.92) 0%, rgba(17, 24, 39, 0.92) 100%);
    border: 1px solid rgba(59, 130, 246, 0.22);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 18px 42px rgba(30, 64, 175, 0.14);
    margin-bottom: 1rem;
}

.answer-title {
    font-size: 0.95rem;
    font-weight: 800;
    color: #93c5fd;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.answer-text {
    color: #f8fafc;
    font-size: 1.02rem;
    line-height: 1.9;
    white-space: pre-wrap;
}

/* ---------- LABELS ---------- */
.mini-label {
    font-size: 0.8rem;
    font-weight: 800;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}

.helper-text {
    color: #94a3b8;
    font-size: 0.96rem;
    line-height: 1.65;
    margin-bottom: 0.5rem;
}

/* ---------- TEXT AREA ---------- */
[data-testid="stTextArea"] textarea {
    background: rgba(15, 23, 42, 0.95) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(96, 165, 250, 0.22) !important;
    border-radius: 18px !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    caret-color: #93c5fd !important;
    padding: 16px !important;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

[data-testid="stTextArea"] textarea:focus {
    border: 1px solid rgba(96, 165, 250, 0.45) !important;
    box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.25) !important;
}

/* ---------- BUTTONS ---------- */
.stButton > button {
    border-radius: 16px !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    background: rgba(15, 23, 42, 0.88) !important;
    color: #f8fafc !important;
    font-weight: 700 !important;
    padding: 0.72rem 1rem !important;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16) !important;
}

.stButton > button:hover {
    border-color: rgba(96, 165, 250, 0.32) !important;
    background: rgba(30, 41, 59, 0.95) !important;
}

/* ---------- METRICS ---------- */
[data-testid="metric-container"] {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.10);
    padding: 14px;
    border-radius: 18px;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
}

/* ---------- EXPANDERS ---------- */
.streamlit-expanderHeader {
    font-weight: 700 !important;
    color: #f8fafc !important;
    font-size: 0.98rem !important;
}

[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.10);
    border-radius: 18px;
    margin-bottom: 0.6rem;
}

/* ---------- CODE BLOCK ---------- */
code {
    white-space: pre-wrap !important;
}

pre {
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.10) !important;
}

/* ---------- HR ---------- */
hr {
    border-color: rgba(148, 163, 184, 0.12);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def is_vector_db_ready():
    return "vector_store" in st.session_state and st.session_state.vector_store is not None


def rebuild_knowledge_base():
    docs = load_documents()
    chunks = split_documents_into_chunks(docs)
    vector_store = build_vector_store(chunks, reset_db=True)

    # Store in Streamlit session (RAM)
    st.session_state.vector_store = vector_store

    return {
        "total_documents_loaded": len(docs),
        "total_chunks_created": len(chunks),
        "vectors_stored": len(chunks)
    }

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Control Panel")
    st.markdown("Tune retrieval and rebuild the knowledge base when needed.")

    st.markdown("---")

    st.markdown(f"**Model**  \n`{GROQ_MODEL}`")
    st.markdown(f"**Docs Path**  \n`{DOCS_PATH}`")
    st.markdown(f"**Vector Store**  \n`{CHROMA_PERSIST_DIR}`")
    st.markdown(f"**Top-K Retrieval**  \n`{TOP_K}`")

    st.markdown("---")

    if is_vector_db_ready():
        st.success("Knowledge base is ready")
    else:
        st.warning("Knowledge base is not built yet")

    st.markdown("### Answering Mode")

    mode_choice = st.radio(
        "Choose behavior:",
        options=[
            "Hybrid Mode (RAG + General LLM Fallback)",
            "Strict RAG Only"
        ],
        index=0
    )

    strict_mode = mode_choice == "Strict RAG Only"

    score_threshold = st.slider(
        "Retrieval relevance threshold",
        min_value=0.50,
        max_value=1.50,
        value=1.10,
        step=0.05,
        help="Lower score = better retrieval. If the best score is above this threshold, Hybrid Mode uses fallback."
    )

    st.markdown("---")

    if st.button("Build / Rebuild Knowledge Base", use_container_width=True):
        try:
            with st.spinner("Building knowledge base..."):
                stats = rebuild_knowledge_base()

            st.success("Knowledge base built successfully")
            st.markdown(f"**Documents Loaded:** {stats['total_documents_loaded']}")
            st.markdown(f"**Chunks Created:** {stats['total_chunks_created']}")
            st.markdown(f"**Vectors Stored:** {stats['vectors_stored']}")
        except Exception as e:
            st.error(f"Build failed: {str(e)}")

    st.markdown("---")
    st.caption("Hybrid RAG Assistant • Retrieval + Groq")


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero-card">
    <div class="hero-title">Hybrid RAG Assistant</div>
    <div class="hero-subtitle">
        Ask questions grounded in your internal documents, or let the assistant intelligently fall back to general LLM knowledge when retrieval confidence is weak.
        Built for a premium, modern AI product experience.
    </div>
    <div class="pill-row">
        <div class="pill">Groq LLM</div>
        <div class="pill">Chroma Retrieval</div>
        <div class="pill">Hybrid Fallback</div>
        <div class="pill">Source-Aware</div>
        <div class="pill">Deployment Ready</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODE BANNER
# =========================================================
if strict_mode:
    st.warning("Strict RAG Mode is active — answers will only come from your document knowledge base.")
else:
    st.success("Hybrid Mode is active — document retrieval first, then general LLM fallback if needed.")


# =========================================================
# MAIN PAGE KNOWLEDGE BASE SETUP
# =========================================================
st.markdown('<div class="mini-label">Knowledge base setup</div>', unsafe_allow_html=True)

kb_col1, kb_col2 = st.columns([1.8, 4])

with kb_col1:
    build_main_clicked = st.button("🔨 Build / Rebuild Knowledge Base", use_container_width=True)

with kb_col2:
    if is_vector_db_ready():
        st.success("Knowledge base detected and ready.")
    else:
        st.warning("Knowledge base is not ready yet. Click the build button once before asking questions.")

if build_main_clicked:
    try:
        with st.spinner("Building knowledge base... Please wait..."):
            stats = rebuild_knowledge_base()

        st.success(
            f"✅ Knowledge base built successfully! "
            f"Loaded {stats['total_documents_loaded']} docs, "
            f"created {stats['total_chunks_created']} chunks, "
            f"stored {stats['vectors_stored']} vectors."
        )
        st.rerun()

    except Exception as e:
        st.error(f"❌ Failed to build knowledge base: {str(e)}")

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# INPUT AREA
# =========================================================
st.markdown('<div class="mini-label">Ask a question</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="helper-text">Try: “What is the notice period?” • “What is your name?” • “What is machine learning?”</div>',
    unsafe_allow_html=True
)

query = st.text_area(
    "",
    placeholder="Type your question here...",
    height=140,
    label_visibility="collapsed"
)

btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1.2, 5])

with btn_col1:
    ask_clicked = st.button("Ask", use_container_width=True)

with btn_col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# ASK FLOW
# =========================================================
if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question first.")
    elif not is_vector_db_ready():
        st.warning("Knowledge base is not ready yet. Please click the Build / Rebuild Knowledge Base button above.")
    else:
        try:
            with st.spinner("Retrieving context and generating answer..."):
                result = ask_rag(
                    query=query.strip(),
                    strict_mode=strict_mode,
                    score_threshold=score_threshold
                )

            # Diagnostics
            st.markdown('<div class="mini-label">Response diagnostics</div>', unsafe_allow_html=True)

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Mode Used", result.get("mode", "unknown"))

            with metric_col2:
                st.metric("Fallback Used", "Yes" if result.get("used_fallback", False) else "No")

            with metric_col3:
                best_score = result.get("best_score", None)
                st.metric("Best Score", f"{best_score:.4f}" if best_score is not None else "N/A")

            st.markdown("<br>", unsafe_allow_html=True)

            # Answer card
            safe_answer = result["answer"].replace("<", "&lt;").replace(">", "&gt;")

            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-title">Answer</div>
                <div class="answer-text">{safe_answer}</div>
            </div>
            """, unsafe_allow_html=True)

            # Retrieval explanation
            retrieval_status = result.get("retrieval_status", "unknown")

            if retrieval_status == "relevant_context_found":
                st.info("Relevant document context was found, so the answer was generated using retrieval-augmented generation.")
            elif retrieval_status == "weak_retrieval" and result.get("used_fallback", False):
                st.warning("Document retrieval was weak, so Hybrid Mode used the general LLM fallback.")
            elif retrieval_status == "weak_retrieval":
                st.warning("Document retrieval was weak, and Strict RAG mode refused to answer outside the knowledge base.")
            elif retrieval_status == "no_results" and result.get("used_fallback", False):
                st.warning("No relevant document results were found, so Hybrid Mode used the general LLM fallback.")
            else:
                st.warning("No strong document context was found.")

            # Sources
            st.markdown('<div class="mini-label">Retrieved sources</div>', unsafe_allow_html=True)

            if result["sources"]:
                for idx, source in enumerate(result["sources"], start=1):
                    title = f"Source {idx} • {source['source_file']} • chunk {source['chunk_id']}"

                    with st.expander(title, expanded=(idx == 1)):
                        meta_col1, meta_col2, meta_col3 = st.columns(3)

                        with meta_col1:
                            st.markdown(f"**File**  \n`{source['source_file']}`")

                        with meta_col2:
                            st.markdown(f"**Chunk ID**  \n`{source['chunk_id']}`")

                        with meta_col3:
                            st.markdown(f"**Score**  \n`{source['score']:.4f}`")

                        st.markdown("**Content Preview**")
                        st.code(source.get("content_preview", ""), language="text")
            else:
                st.info("No document sources were returned for this answer.")

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")


# =========================================================
# FOOTER
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)
st.caption("Hybrid RAG Assistant • Premium Dark UI • Streamlit + Chroma + Groq")