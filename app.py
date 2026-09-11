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


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .source-box {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 10px;
        padding: 12px;
        margin-top: 8px;
    }
    .legal-note {
        padding: 12px;
        border-radius: 10px;
        background: rgba(255,193,7,.10);
        border: 1px solid rgba(255,193,7,.35);
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
        gdown.download(
            id=DRIVE_FILE_ID,
            output=str(PDF_PATH),
            quiet=False,
            fuzzy=True,
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
        "Beginner": "Explain legal concepts in beginner-friendly language and define technical terms.",
        "Intermediate": "Use moderate legal and cybersecurity terminology and explain important terms briefly.",
        "Advanced": "Use precise legal and cybersecurity terminology, section-level reasoning, and nuanced distinctions.",
    }[technicality]

    size_instruction = {
        "Short": "Keep the answer concise, normally around 150-250 words.",
        "Medium": "Give a balanced answer, normally around 300-500 words.",
        "Detailed": "Give a detailed answer, normally around 600-900 words.",
    }[response_size]

    style_instruction = {
        "Legal Q&A": "Answer directly, then identify the relevant legal provision and explain its relevance.",
        "Case Analysis": "Structure the answer as facts/assumptions, applicable provision, analysis, and practical lawful next steps.",
        "Study Mode": "Teach the concept clearly with definitions, section references, and a small example.",
        "Compliance Checklist": "Focus on lawful compliance steps and avoid operational abuse instructions.",
    }[answer_style]

    return f"""
You are CyberLawGPT, an AI legal-information assistant focused on Pakistani cyber-law.

PRIMARY SOURCE RULE
Use ONLY the retrieved source context below for legal claims about the Pakistani cyber-law covered by the supplied PDF.
Do not invent section numbers, punishments, definitions, procedures, authorities, or legal tests.
If the PDF does not contain enough information to answer, say so explicitly.

CURRENT-SOURCE LIMIT
The supplied document is the application's legal knowledge base. It may not contain every Pakistani statute, regulation, notification, court judgment, or later amendment. Do not silently fill gaps from memory.

LEGAL-SAFETY RULE
This is legal information, not a substitute for a licensed Pakistani lawyer or a court's interpretation.
Do not provide instructions for unauthorized access, malware, credential theft, evasion, surveillance, fraud, harassment, doxxing, disruption, or other unlawful cyber activity.
If the user asks how to commit or facilitate such conduct, refuse the operational instructions and instead explain the relevant lawful/legal-risk perspective from the retrieved source.
Defensive, educational, compliance, incident-response, and authorized security-testing guidance may be discussed at a high level.

ACCURACY RULES
1. Distinguish clearly between what the source says and what is an inference.
2. Cite the relevant section number whenever the source provides one.
3. If a section number cannot be verified from the retrieved context, do not guess it.
4. Do not claim that something is definitely a crime or definitely legal when the source does not establish that conclusion.
5. Mention uncertainty when facts are incomplete.
6. Do not manufacture case law or citations.
7. At the end, include a brief "Source basis" line listing the retrieved PDF page numbers.

USER SETTINGS
Technicality: {technicality}
Response size: {response_size}
Language: {language_instruction}
Answer style: {style_instruction}
Length guidance: {size_instruction}
Technicality guidance: {technicality_instruction}

RETRIEVED SOURCE CONTEXT
------------------------
{context}
------------------------

USER QUESTION
{question}

Now provide the safest and most accurate legal-information response supported by the source.
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
st.markdown('<div class="main-title">⚖️ CyberLawGPT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">RAG-powered Pakistani cyber-law information assistant '
    'using FAISS + Sentence Transformers + Groq.</div>',
    unsafe_allow_html=True,
)

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

st.success(
    f"Knowledge base ready: {len(records)} indexed passages from the supplied PDF."
)

# Session history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about Pakistani cyber law..."
)

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
st.caption(
    "CyberLawGPT is a source-grounded educational tool. "
    "Verify important legal matters against the current official Pakistani law "
    "and obtain professional legal advice where appropriate."
)
