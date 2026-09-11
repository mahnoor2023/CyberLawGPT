# ⚖️ CyberLawGPT

CyberLawGPT is a beginner-friendly **Retrieval-Augmented Generation (RAG)** application for Pakistani cyber-law information.

It uses only **three project files**:

```text
CyberLawGPT/
├── app.py
├── requirements.txt
└── readme.md
```

The application automatically downloads the supplied Google Drive PDF at startup, extracts its text, creates embeddings, stores them in a FAISS vector index, retrieves the most relevant passages for a question, and sends those passages to Groq for a source-grounded answer.

## Features

- Python + Streamlit
- FAISS vector similarity search
- Sentence Transformers embeddings
- Groq LLM
- Automatic Google Drive PDF download on startup
- Automatic PDF text extraction
- Automatic chunking and embedding
- Configurable retrieval depth
- Technicality selector:
  - Beginner
  - Intermediate
  - Advanced
- Response size:
  - Short
  - Medium
  - Detailed
- Answer language:
  - English
  - Roman Urdu
  - Urdu
- Answer style:
  - Legal Q&A
  - Case Analysis
  - Study Mode
  - Compliance Checklist
- Retrieved source passages with PDF page numbers
- Low-temperature generation for more consistent legal answers
- Legal-safety prompt that avoids operational cyber-abuse instructions
- Works with Google Colab and Streamlit Community Cloud

## Important legal scope

The application is deliberately **source-grounded**.

It should not pretend that one PDF contains every Pakistani cyber-related law, regulation, notification, court judgment, or later amendment.

The supplied knowledge base is the Google Drive document specified by the project owner. If the PDF is an older version of a Pakistani cyber law, CyberLawGPT will not silently invent later amendments.

For important legal matters, verify the current law from an authoritative Pakistani government source and consult a qualified lawyer.

## RAG architecture

```text
Google Drive PDF
      │
      ▼
Download on startup
      │
      ▼
PyMuPDF text extraction
      │
      ▼
Text cleaning + chunking
      │
      ▼
Sentence Transformer embeddings
      │
      ▼
FAISS IndexFlatIP
      │
      │      User Question
      │           │
      │           ▼
      └────► Query Embedding
                  │
                  ▼
          Similarity Retrieval
                  │
                  ▼
       Top relevant PDF passages
                  │
                  ▼
             Groq LLM
                  │
                  ▼
       Source-grounded answer
```

## Why FAISS?

FAISS is used for vector similarity search.

The app normalizes embeddings and uses:

```python
faiss.IndexFlatIP(...)
```

Inner Product then behaves like cosine similarity for normalized vectors.

This keeps the implementation simple and easy to understand.

## Google Drive PDF

The application uses this Google Drive file ID:

```text
1zE6ll1wwOX1l6qudvgwp_P3afwDurdkn
```

The Drive file must be accessible to the app. If downloading fails, change the Google Drive sharing setting so that the file can be accessed through its link.

The app uses `gdown`, so you do not need to manually download the PDF into the GitHub repository.

## Groq API

The application uses:

```text
openai/gpt-oss-120b
```

This model ID is selected because current Groq documentation recommends `openai/gpt-oss-120b` as a replacement for the deprecated `llama-3.3-70b-versatile` model.

### Get a Groq API key

Create a Groq API key from the Groq developer console.

Do **not** put the API key directly inside `app.py`.

## Google Colab setup

Open a new Google Colab notebook.

Upload:

```text
app.py
requirements.txt
readme.md
```

Then run:

```python
!pip install -r requirements.txt
```

Set the API key:

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Run:

```python
!streamlit run app.py &>/content/streamlit.log &
```

For a public temporary Colab URL, you can use a tunnel such as:

```python
!npm install -g localtunnel
!lt --port 8501
```

The PDF and embedding model are downloaded when the application initializes.

## Streamlit Community Cloud deployment

1. Create a GitHub repository.
2. Add only:
   - `app.py`
   - `requirements.txt`
   - `readme.md`
3. Create a Streamlit Community Cloud app from the repository.
4. Set the main file to:

```text
app.py
```

5. Open the app's **Secrets** settings.
6. Add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

7. Deploy.

Do not commit a real API key to GitHub.

## Resource considerations

The embedding model is downloaded on first startup and cached for the application process.

The FAISS index is also cached.

On Streamlit Community Cloud, the first startup can take longer because the embedding model and PDF must be downloaded.

The project intentionally does not use a paid vector database.

## Legal-safety design

CyberLawGPT is designed to answer questions such as:

- What does a particular cyber-law section mean?
- What type of conduct is covered by a provision?
- What are the legal risks of unauthorized access?
- What should an organization consider after a cyber incident?
- What is the difference between lawful security testing and unauthorized access?
- How can a student understand a section for an exam?
- What compliance considerations are visible in the supplied law?

For requests that would facilitate cyber abuse, the application is instructed to avoid giving operational instructions and instead provide the relevant legal-risk or compliance perspective supported by the retrieved source.

## Accuracy safeguards

The prompt instructs the model to:

- use the retrieved PDF as the primary legal source;
- avoid inventing section numbers;
- avoid inventing punishments;
- avoid inventing court cases;
- distinguish source statements from inference;
- mention when the PDF does not contain enough information;
- provide retrieved PDF page numbers;
- avoid presenting the output as a final legal opinion.

## Recommended improvement for production

For a production legal-information system, use a maintained official legal corpus rather than a single manually supplied PDF.

The corpus should be versioned and updated when laws, amendments, rules, notifications, or authoritative interpretations change.

The Pakistan Code website currently lists the Prevention of Electronic Crimes Act, 2016 and indicates that it is under review. Always verify important legal questions against the current official text and Gazette notifications.

## Disclaimer

CyberLawGPT is an educational/source-grounded AI application.

It is not a law firm, lawyer, court, government authority, or legal representative. Its responses can contain errors and should not be treated as definitive legal advice.

For an actual legal dispute, investigation, notice, prosecution, complaint, employment matter, or other high-stakes situation, consult a qualified Pakistani legal professional and verify the current law.

## License / usage

Use this project for lawful education, research, compliance, and authorized cybersecurity work.

Do not use the application to facilitate unauthorized access, malware deployment, credential theft, fraud, harassment, privacy violations, disruption, or evasion of law-enforcement/security controls.
