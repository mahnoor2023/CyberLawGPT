import os
import re
from pathlib import Path

import faiss
import gdown
import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CyberLawGPT
# RAG application for Pakistan cyber-law information
# Stack: Python + Streamlit + FAISS + Sentence Transformers + Groq
# ============================================================

st.set_page_config(
    page_title="CyberLawGPT",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "CyberLawGPT"
DRIVE_FILE_ID = "1zE6ll1wwOX1l6qudvgwp_P3afwDurdkn"
PDF_PATH = Path("cyber_law_source.pdf")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
NCCIA_COMPLAINT_URL = "https://complaint.nccia.gov.pk/"
NCCIA_WEBSITE_URL = "https://www.nccia.gov.pk/"


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 42%, #f8fafc 100%); }
    .block-container { max-width: 1180px; padding-top: 2.1rem; padding-bottom: 2rem; }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #111827 100%); }
    [data-testid="stSidebar"] * { color: #e5e7eb; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background: #1e293b; border-color: #334155; }

    .hero {
        position: relative; overflow: hidden; padding: 30px 32px 28px 32px;
        border-radius: 24px; background: linear-gradient(135deg, #0f172a 0%, #172554 58%, #1e3a8a 100%);
        color: white; box-shadow: 0 18px 45px rgba(15, 23, 42, .16); margin-bottom: 18px;
    }
    .hero:after {
        content: ""; position: absolute; width: 260px; height: 260px; right: -80px; top: -110px;
        border-radius: 50%; background: rgba(96, 165, 250, .16);
    }
    .hero-kicker {
        display: inline-block; padding: 6px 11px; border: 1px solid rgba(255,255,255,.18);
        border-radius: 999px; background: rgba(255,255,255,.08); color: #dbeafe;
        font-size: .78rem; font-weight: 700; letter-spacing: .04em; margin-bottom: 12px;
    }
    .hero-title { font-size: clamp(2rem, 4vw, 3.15rem); line-height: 1.05; font-weight: 850; margin: 0; letter-spacing: -.04em; }
    .hero-title span { color: #93c5fd; }
    .hero-subtitle { max-width: 760px; margin: 12px 0 20px 0; color: #cbd5e1; font-size: 1.02rem; line-height: 1.65; }
    .status-row { display: flex; flex-wrap: wrap; gap: 9px; }
    .status-pill { padding: 7px 11px; border-radius: 999px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.13); color: #e2e8f0; font-size: .78rem; font-weight: 600; }
    .status-dot { color: #4ade80; }

    .section-heading, .try-title { font-size: 1.35rem; font-weight: 800; color: #0f172a; margin: 25px 0 10px 0; }
    .feature-card {
        min-height: 142px; padding: 20px; border: 1px solid #e2e8f0; border-radius: 18px;
        background: rgba(255,255,255,.88); box-shadow: 0 7px 25px rgba(15, 23, 42, .06);
    }
    .feature-icon { font-size: 1.55rem; margin-bottom: 8px; }
    .feature-title { font-size: 1rem; font-weight: 800; color: #0f172a; margin-bottom: 5px; }
    .feature-text { color: #64748b; font-size: .88rem; line-height: 1.5; }

    .complaint-card {
        padding: 18px 20px; border-radius: 18px; border: 1px solid #fecaca;
        background: linear-gradient(135deg, #fff7ed 0%, #fff1f2 100%); margin: 22px 0 8px 0;
    }
    .complaint-title { font-size: 1.05rem; font-weight: 800; color: #7f1d1d; margin-bottom: 4px; }
    .complaint-text { color: #57534e; font-size: .88rem; line-height: 1.5; }
    .try-subtitle { color: #64748b; font-size: .9rem; margin-bottom: 10px; }

    .stButton > button, .stLinkButton > a {
        border-radius: 11px !important; font-weight: 700 !important; border: 1px solid #dbe2ea !important;
        transition: all .18s ease !important;
    }
    .stButton > button:hover, .stLinkButton > a:hover { transform: translateY(-1px); border-color: #93c5fd !important; box-shadow: 0 6px 18px rgba(37,99,235,.10); }

    .legal-note {
        padding: 13px 16px; border-radius: 13px; background: #fffbeb; border: 1px solid #fde68a;
        color: #57534e; font-size: .88rem; line-height: 1.55; margin-top: 16px;
    }
    [data-testid="stChatMessage"] { border-radius: 16px; }
    [data-testid="stChatInput"] { margin-top: 10px; }
    .footer { text-align: center; padding: 20px 8px 4px 8px; color: #94a3b8; font-size: .78rem; line-height: 1.6; }

    @media (max-width: 700px) {
        .block-container { padding-top: 1rem; }
        .hero { padding: 24px 20px; border-radius: 18px; }
        .feature-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Secrets / environment
# -----------------------------
def get_groq_key():
    """Read the Groq key from Streamlit secrets or environment variables."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None

    return key or os.getenv("GROQ_API_KEY")


# -----------------------------
# Download PDF
# -----------------------------
@st.cache_resource(show_spinner="Downloading the Pakistan cyber-law PDF...")
def ensure_pdf():
    """Download the user's Google Drive PDF once per app process."""
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 10_000:
        return PDF_PATH

    try:
        # gdown 5.x accepts a Google Drive file ID directly.
        # Do not use fuzzy=True because newer gdown releases removed that argument.
        gdown.download(
            id=DRIVE_FILE_ID,
            output=str(PDF_PATH),
            quiet=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "The Google Drive PDF could not be downloaded. "
            "Make sure the Drive file is accessible to anyone with the link."
        ) from exc

    if not PDF_PATH.exists() or PDF_PATH.stat().st_size < 10_000:
        raise RuntimeError(
            "The downloaded file appears to be invalid or empty. "
            "Check the Google Drive sharing permission."
        )

    return PDF_PATH


# -----------------------------
# PDF parsing + chunking
# -----------------------------
def normalize_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, chunk_size=1200, overlap=180):
    """Character-based chunking keeps the app simple and lightweight."""
    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Prefer ending near a sentence/paragraph boundary.
        if end < len(text):
            boundary = max(
                chunk.rfind("\n"),
                chunk.rfind(". "),
                chunk.rfind("۔ "),
                chunk.rfind("; "),
            )
            if boundary > chunk_size * 0.60:
                end = start + boundary + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return [c for c in chunks if len(c) >= 80]


@st.cache_resource(show_spinner="Reading the PDF and creating document chunks...")
def load_documents(pdf_path):
    doc = fitz.open(pdf_path)
    records = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")
        for chunk_number, chunk in enumerate(chunk_text(text), start=1):
            records.append(
                {
                    "text": chunk,
                    "page": page_number,
                    "chunk": chunk_number,
                }
            )

    doc.close()

    if not records:
        raise RuntimeError(
            "No readable text was extracted from the PDF. "
            "If the PDF is scanned, OCR should be added before indexing."
        )

    return records


# -----------------------------
# Embeddings + FAISS
# -----------------------------
@st.cache_resource(show_spinner="Loading the embedding model and building FAISS index...")
def build_index(records):
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [item["text"] for item in records]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index


def retrieve(question, model, index, records, top_k=5):
    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(records[idx])
        item["score"] = float(score)
        results.append(item)

    return results


# -----------------------------
# Legal-safety prompt
# -----------------------------
def build_prompt(
    question,
    context,
    technicality,
    response_size,
    language,
    answer_style,
):
    language_instruction = {
        "English": "Answer in clear professional English.",
        "Urdu": "Answer in clear Urdu script.",
        "Roman Urdu": "Answer in clear Roman Urdu.",
    }[language]

    technicality_instruction = {
        "Beginner": "Use simple language. Explain legal and cybersecurity terms briefly.",
        "Intermediate": "Use moderate legal and cybersecurity terminology with short explanations.",
        "Advanced": "Use precise legal terminology, section-level reasoning, and nuanced distinctions.",
    }[technicality]

    size_instruction = {
        "Short": "Keep the answer concise, normally around 150-250 words.",
        "Medium": "Give a balanced answer, normally around 300-500 words.",
        "Detailed": "Give a detailed answer, normally around 600-900 words.",
    }[response_size]

    style_instruction = {
        "Legal Q&A": "Answer directly, identify the relevant provision, and explain why it is relevant.",
        "Case Analysis": "Use: facts/assumptions, applicable provision, analysis, and lawful next steps.",
        "Study Mode": "Teach the concept clearly with definitions, section references, and a simple example.",
        "Compliance Checklist": "Focus on lawful compliance, prevention, documentation, and reporting steps.",
    }[answer_style]

    return f"""
You are CyberLawGPT, a source-grounded Pakistani cyber-law information assistant.
Your purpose is to help users understand Pakistani cyber-law provisions, especially the law contained in the supplied legal PDF.

CORE RAG RULE
- Treat the retrieved PDF context as the primary and controlling source for legal claims.
- Answer the user's actual question using the most relevant retrieved passages.
- Never invent a section number, offence, punishment, legal test, procedure, authority, case, or citation.
- If the retrieved context does not support a legal conclusion, explicitly say: "The supplied source does not provide enough information to confirm this."
- Do not silently use general model memory to fill missing legal details.

PAKISTANI-LAW SCOPE
- Interpret the user's question in the context of Pakistani cyber law and the supplied PECA/cyber-law material.
- If the question concerns hacking, unauthorized access, online fraud, identity misuse, harassment, privacy, cyberstalking, electronic forgery/fraud, data, interception, harmful content, or another cyber-law issue, identify the relevant provision only when it is supported by the retrieved source.
- If the question is outside the supplied legal corpus, clearly distinguish that limitation instead of pretending the answer is authoritative.

REPORTING / COMPLAINT GUIDANCE
- If the user says they are a victim, wants to report cybercrime, asks where to complain, or asks how to file a cybercrime complaint, provide practical lawful reporting guidance.
- Tell the user that the official National Cyber Crime Investigation Agency (NCCIA) is the relevant Pakistani cybercrime reporting authority.
- Provide the official complaint portal: https://complaint.nccia.gov.pk/
- You may also mention the official NCCIA website: https://www.nccia.gov.pk/
- Encourage the user to preserve relevant evidence such as screenshots, URLs, messages, transaction records, account details, and other lawful evidence before deleting anything. Do not ask the user to expose unnecessary sensitive information in the chat.
- Do not claim that CyberLawGPT itself registers, investigates, or submits a complaint.

LEGAL-SAFETY RULE
- This application provides educational legal information, not a lawyer's opinion, legal representation, or a guarantee of legal outcome.
- Do not provide operational instructions that enable unauthorized access, malware, credential theft, evasion, surveillance, fraud, harassment, doxxing, disruption, or other unlawful cyber activity.
- If the user requests instructions to commit or facilitate cyber abuse, refuse those operational details and redirect to the relevant legal-risk, prevention, compliance, or authorized-security perspective.
- Defensive, educational, incident-response, compliance, and authorized security-testing guidance may be discussed at a safe level.

ACCURACY AND RESPONSE RULES
1. Separate source-supported facts from reasonable explanation or inference.
2. Cite section numbers only when they are visible/verifiable in the retrieved context.
3. Never invent court cases, judgments, dates, penalties, authorities, or legal citations.
4. Do not say conduct is definitely legal or illegal unless the retrieved source supports that conclusion.
5. If facts are incomplete, state the assumptions and uncertainty.
6. For case-like questions, explain which additional facts could change the legal analysis.
7. Prefer precise, neutral, non-alarmist language.
8. End with a short "Source basis" line containing the relevant PDF page number(s).
9. For complaint/reporting questions, include the official NCCIA complaint portal when relevant.

USER SETTINGS
Technicality: {technicality}
Response size: {response_size}
Language: {language_instruction}
Answer style: {style_instruction}
Length guidance: {size_instruction}
Technicality guidance: {technicality_instruction}

RETRIEVED SOURCE CONTEXT
=========================
{context}
=========================

USER QUESTION
=============
{question}

Now produce the most accurate, useful, source-grounded legal-information response possible. Do not hallucinate missing law.
"""


def ask_groq(question, retrieved, technicality, response_size, language, answer_style):
    api_key = get_groq_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to Streamlit Secrets "
            "or set the GROQ_API_KEY environment variable."
        )

    client = Groq(api_key=api_key)

    context_parts = []
    for item in retrieved:
        context_parts.append(
            f"[PDF page {item['page']}, retrieval score {item['score']:.3f}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = build_prompt(
        question=question,
        context=context,
        technicality=technicality,
        response_size=response_size,
        language=language,
        answer_style=answer_style,
    )

    max_tokens = {
        "Short": 700,
        "Medium": 1200,
        "Detailed": 1800,
    }[response_size]

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful Pakistani cyber-law information assistant. "
                    "Ground legal claims in the supplied retrieval context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_completion_tokens=max_tokens,
    )

    return completion.choices[0].message.content


# -----------------------------
# App initialization
# -----------------------------
st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">⚖️ PAKISTAN CYBER-LAW • AI LEGAL INFORMATION</div>
        <h1 class="hero-title">CyberLaw<span>GPT</span></h1>
        <p class="hero-subtitle">
            Ask questions about Pakistani cyber law and get source-grounded explanations
            powered by Retrieval-Augmented Generation, FAISS and Groq.
        </p>
        <div class="status-row">
            <span class="status-pill"><span class="status-dot">●</span> Legal knowledge base</span>
            <span class="status-pill">🔎 FAISS retrieval</span>
            <span class="status-pill">🤖 Groq AI</span>
            <span class="status-pill">📚 PDF source-grounded</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">What can CyberLawGPT help with?</div>', unsafe_allow_html=True)
feature_cols = st.columns(3)
features = [
    ("🔍", "Ask about cyber law", "Understand cyber offences, legal concepts, sections and requirements using the retrieved legal source."),
    ("⚖️", "Explore legal scenarios", "Get structured explanations for fraud, harassment, unauthorized access, privacy and other cyber-law questions."),
    ("🛡️", "Learn safely", "Educational, compliance and defensive guidance without providing instructions for unlawful cyber activity."),
]
for col, (icon, title, text) in zip(feature_cols, features):
    with col:
        st.markdown(
            f"""<div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-text">{text}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="complaint-card">
        <div class="complaint-title">🚨 Victim of a cybercrime?</div>
        <div class="complaint-text">
            CyberLawGPT can explain the relevant legal information, but an actual
            complaint should be submitted through the official National Cyber Crime
            Investigation Agency (NCCIA) channel.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
complaint_col, website_col = st.columns([1, 1])
with complaint_col:
    st.link_button("📝 Submit Cybercrime Complaint", NCCIA_COMPLAINT_URL, use_container_width=True)
with website_col:
    st.link_button("🌐 Visit Official NCCIA Website", NCCIA_WEBSITE_URL, use_container_width=True)

st.markdown(
    """
    <div class="legal-note">
        <b>Important:</b> CyberLawGPT provides source-grounded legal information for
        education and general guidance. It is not a lawyer, does not create an
        attorney-client relationship, and should not be treated as a final legal opinion.
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Response Controls")

    technicality = st.selectbox(
        "Technicality level",
        ["Beginner", "Intermediate", "Advanced"],
        index=1,
    )

    response_size = st.selectbox(
        "Response size",
        ["Short", "Medium", "Detailed"],
        index=1,
    )

    language = st.selectbox(
        "Answer language",
        ["English", "Roman Urdu", "Urdu"],
        index=0,
    )

    answer_style = st.selectbox(
        "Answer style",
        ["Legal Q&A", "Case Analysis", "Study Mode", "Compliance Checklist"],
        index=0,
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=3,
        max_value=8,
        value=5,
        help="More passages can improve context but may add noise.",
    )

    show_sources = st.checkbox("Show retrieved source passages", value=True)

    st.divider()
    st.caption(f"LLM: {GROQ_MODEL}")
    st.caption(f"Embeddings: {EMBEDDING_MODEL}")
    st.caption("Vector index: FAISS Inner Product")

# Build RAG resources
try:
    pdf_file = ensure_pdf()
    records = load_documents(str(pdf_file))
    embedding_model, faiss_index = build_index(records)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.markdown(
    f"""
    <div style="margin-top:18px;padding:11px 15px;border-radius:12px;
                background:#f0fdf4;border:1px solid #bbf7d0;color:#166534;
                font-size:.86rem;">
        <b>● Knowledge base ready</b> &nbsp;•&nbsp; {len(records)} indexed passages
        &nbsp;•&nbsp; Source PDF loaded &nbsp;•&nbsp; FAISS index active
    </div>
    """,
    unsafe_allow_html=True,
)

# Session history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown('<div class="try-title">Try asking</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="try-subtitle">Start with an example or type your own question below.</div>',
    unsafe_allow_html=True,
)

suggestion_cols = st.columns(3)
suggestions = [
    "What is unauthorized access under Pakistani cyber law?",
    "What should I do if I become a victim of online fraud?",
    "What cyber law applies to online harassment?",
]
for i, (col, suggestion) in enumerate(zip(suggestion_cols, suggestions)):
    with col:
        if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            st.session_state["pending_question"] = suggestion
            st.rerun()

pending_question = st.session_state.pop("pending_question", None)
chat_question = st.chat_input("Ask anything about Pakistani cyber law...")
question = chat_question or pending_question

if question:
    cleaned_question = question.strip()

    if len(cleaned_question) < 4:
        st.warning("Please enter a more specific question.")
        st.stop()

    # Keep a reasonable limit to avoid accidental huge prompts.
    if len(cleaned_question) > 4000:
        st.warning("Please keep the question under 4,000 characters.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": cleaned_question}
    )

    with st.chat_message("user"):
        st.markdown(cleaned_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the legal knowledge base..."):
            retrieved = retrieve(
                cleaned_question,
                embedding_model,
                faiss_index,
                records,
                top_k=top_k,
            )

        if not retrieved:
            answer = (
                "I could not retrieve relevant material from the supplied legal PDF. "
                "Please rephrase the question."
            )
            st.markdown(answer)
        else:
            try:
                with st.spinner("Generating a source-grounded answer..."):
                    answer = ask_groq(
                        cleaned_question,
                        retrieved,
                        technicality,
                        response_size,
                        language,
                        answer_style,
                    )
                st.markdown(answer)
            except Exception as exc:
                answer = f"Unable to generate the answer: {exc}"
                st.error(answer)

            if show_sources:
                with st.expander("📚 Retrieved source passages"):
                    for i, item in enumerate(retrieved, start=1):
                        st.markdown(
                            f"**Passage {i} — PDF page {item['page']} — "
                            f"similarity {item['score']:.3f}**"
                        )
                        st.write(item["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# Footer
st.divider()
st.markdown(
    """
    <div class="footer">
        <b>CyberLawGPT</b> • Source-grounded Pakistani cyber-law information assistant<br>
        Verify important legal matters against the current official law and obtain
        professional legal advice where appropriate.<br>
        For cybercrime complaints, use the official NCCIA portal.
    </div>
    """,
    unsafe_allow_html=True,
)
