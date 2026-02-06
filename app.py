import os
import logging
import time
import tempfile
import threading
import sqlite3
import smtplib
import json
from email.message import EmailMessage
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
from groq import Groq
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
try:
    # LangChain >=0.1.0 splitters moved to langchain_text_splitters
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - fallback for older installs
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & LOGGING
# -----------------------------------------------------------------------------
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Storage paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "storage" / "documents"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "legalmind.db"

# Environment Variables Validation
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables.")
    raise RuntimeError("Missing GROQ_API_KEY. Please set it in your .env file.")

# Global Resource Initialization
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    # Load embeddings once globally (heavy resource)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Global AI resources initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize AI resources: {e}")
    raise

# Thread-safe Vector DB Storage
vector_db_lock = threading.Lock()
vector_db: Optional[FAISS] = None
GROQ_MODEL = "llama-3.1-8b-instant"

# Email settings (Gmail SMTP)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM") or SMTP_USER

# Background reminder thread
reminder_thread_stop = threading.Event()

# -----------------------------------------------------------------------------
# 2. UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

def tiered_pdf_extract(file_path: str) -> str:
    """Extracts text with a tiered fallback: PyMuPDF -> PyPDF2."""
    text = ""
    try:
        # Tier 1: PyMuPDF (High Accuracy)
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF failed, falling back to PyPDF2: {e}")
        try:
            # Tier 2: PyPDF2 Fallback
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e_inner:
            logger.error(f"All PDF extraction methods failed: {e_inner}")
            return ""
    return text

def get_groq_completion(prompt: str, system_prompt: str = "You are a helpful legal assistant.") -> str:
    """Wrapper for Groq API calls with logging and error handling."""
    start_time = time.time()
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODEL,
            temperature=0.2, # Lower temperature for legal accuracy
            max_tokens=2048
        )
        latency = time.time() - start_time
        logger.info(f"Groq API call completed in {latency:.2f}s")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise

def safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {}

def cosine_similarity(vec_a, vec_b) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))

def extract_structured_case(query: str) -> dict:
    schema_prompt = f"""
    Extract structured fields from this legal query. Respond ONLY as JSON with keys:
    issue, jurisdiction, statute, relief_sought, key_facts.

    Rules:
    - Keep values concise.
    - If unknown, use null.
    - key_facts should be a short bullet-like string or list of facts in one string.

    Query: {query}
    """
    raw = get_groq_completion(schema_prompt, "You extract structured legal fields.")
    data = safe_json_loads(raw)
    return {
        "issue": data.get("issue"),
        "jurisdiction": data.get("jurisdiction"),
        "statute": data.get("statute"),
        "relief_sought": data.get("relief_sought"),
        "key_facts": data.get("key_facts"),
    }

def compute_precedent_similarity(query: str, docs: list) -> float:
    try:
        query_vec = embeddings.embed_query(query)
        doc_texts = [doc.page_content[:2000] for doc in docs]
        doc_vecs = embeddings.embed_documents(doc_texts)
        sims = [cosine_similarity(query_vec, dv) for dv in doc_vecs]
        return round(sum(sims) / max(len(sims), 1), 3)
    except Exception as e:
        logger.warning(f"Similarity calc failed: {e}")
        return 0.0

def score_outcome(structured: dict, precedent_similarity: float) -> tuple:
    score = 0.5
    rationale = []

    if structured.get("issue"):
        score += 0.08
        rationale.append("Issue identified")
    if structured.get("jurisdiction"):
        score += 0.06
        rationale.append("Jurisdiction specified")
    if structured.get("statute"):
        score += 0.08
        rationale.append("Statute identified")
    if structured.get("relief_sought"):
        score += 0.06
        rationale.append("Relief sought defined")
    if structured.get("key_facts"):
        score += 0.10
        rationale.append("Key facts provided")

    if precedent_similarity >= 0.75:
        score += 0.12
        rationale.append("Strong precedent similarity")
    elif precedent_similarity >= 0.55:
        score += 0.06
        rationale.append("Moderate precedent similarity")
    else:
        rationale.append("Weak precedent similarity")

    score = max(0.0, min(1.0, score))
    if score >= 0.72:
        band = "High"
    elif score >= 0.52:
        band = "Medium"
    else:
        band = "Low"

    return round(score, 3), band, rationale

def get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with get_db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                content_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                case_title TEXT NOT NULL,
                court_date TEXT NOT NULL,
                note TEXT,
                reminder_date TEXT NOT NULL,
                sent_at TEXT
            )
        """)

def send_email(to_email: str, subject: str, body: str) -> None:
    if not SMTP_USER or not SMTP_PASS:
        raise RuntimeError("SMTP_USER/SMTP_PASS not set in environment.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

def reminder_worker(poll_seconds: int = 60) -> None:
    while not reminder_thread_stop.is_set():
        now = datetime.now()
        window_end = now + timedelta(seconds=poll_seconds)
        try:
            with get_db_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM reminders
                    WHERE sent_at IS NULL
                      AND reminder_date >= ?
                      AND reminder_date < ?
                    """,
                    (now.isoformat(), window_end.isoformat()),
                ).fetchall()

                for row in rows:
                    subject = f"Court Case Reminder: {row['case_title']}"
                    body = (
                        f"Reminder: Court case scheduled today.\n\n"
                        f"Case: {row['case_title']}\n"
                        f"Court Date: {row['court_date']}\n"
                        f"Note: {row['note'] or '-'}\n\n"
                        f"LegalMind AI Reminder Service"
                    )
                    send_email(row["email"], subject, body)
                    conn.execute(
                        "UPDATE reminders SET sent_at = ? WHERE id = ?",
                        (datetime.now().isoformat(), row["id"]),
                    )
        except Exception as e:
            logger.error(f"Reminder worker error: {e}", exc_info=True)

        reminder_thread_stop.wait(poll_seconds)

# -----------------------------------------------------------------------------
# 3. ERROR HANDLERS
# -----------------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler for structured JSON responses."""
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description, "code": e.code}), e.code
    logger.error(f"Unhandled Exception: {str(e)}", exc_info=True)
    return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500

# -----------------------------------------------------------------------------
# 4. API ROUTES
# -----------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('home.html', page_title="Home")

@app.route('/fee-finder')
def fee_finder_page():
    return render_template('fee_finder.html', page_title="Fee Finder")

@app.route('/law-advisor')
def law_advisor_page():
    return render_template('law_advisor.html', page_title="Law Advisor")

@app.route('/case-predictor')
def case_predictor_page():
    return render_template('case_predictor.html', page_title="Case Predictor")

@app.route('/summarizer')
def summarizer_page():
    return render_template('summarizer.html', page_title="Summarizer")

@app.route('/document-locker')
def document_locker_page():
    return render_template('document_locker.html', page_title="Document Locker")

@app.route('/case-reminders')
def case_reminders_page():
    return render_template('case_reminders.html', page_title="Case Reminders")

@app.route('/api/feefinder', methods=['POST'])
def fee_finder():
    """Estimates lawyer fees based on Indian legal context."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    prompt = f"""
    Based on the following legal scenario in India, estimate the potential fee structure:
    Scenario: {data['query']}
    
    Provide a breakdown including:
    1. Common fee types (Hourly, Flat, Retainer).
    2. Estimated price ranges.
    3. Factors that might increase the cost.
    """
    response = get_groq_completion(prompt, "You are a legal fee estimation expert.")
    return jsonify({"response": response})

@app.route('/api/lawadvisor', methods=['POST'])
def law_advisor():
    """Provides Constitutional advice using Chain-of-Thought reasoning."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    # Step 1: Article identification
    id_prompt = f"Identify the specific Article number of the Indian Constitution relevant to: {data['query']}. Respond ONLY with the Article number."
    article_id = get_groq_completion(id_prompt).strip()

    # Step 2: Explanation
    exp_prompt = f"Explain {article_id} of the Indian Constitution as it applies to: {data['query']}."
    explanation = get_groq_completion(exp_prompt)

    return jsonify({"article": article_id, "explanation": explanation})

@app.route('/api/summarize', methods=['POST'])
def summarize_pdf():
    """Extracts text and provides a concise legal summary."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        text = tiered_pdf_extract(temp_path)
        if not text:
            return jsonify({"error": "Could not extract text from PDF"}), 422

        # Token safety: Limit input to approximately 6000 chars
        summary_prompt = f"Summarize the following legal document accurately: {text[:6000]}"
        summary = get_groq_completion(summary_prompt, "You are a precise legal document summarizer.")
        
        return jsonify({"summary": summary})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/index_case', methods=['POST'])
def index_case():
    """Chunks and embeds PDF case files into the FAISS vector store."""
    global vector_db
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        text = tiered_pdf_extract(temp_path)
        if not text:
            return jsonify({"error": "Failed to extract text for indexing"}), 422

        # RAG Optimization: Overlapping chunks for context preservation
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        with vector_db_lock:
            vector_db = FAISS.from_texts(chunks, embeddings)
        
        logger.info(f"Indexed case with {len(chunks)} chunks.")
        return jsonify({"message": "Case indexed successfully", "chunks": len(chunks)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/predict', methods=['POST'])
def predict_case():
    """Predicts case outcome using RAG and a specialized legal analyst prompt."""
    if vector_db is None:
        return jsonify({"error": "No case document has been indexed yet. Please upload a PDF first."}), 412
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query'"}), 400

    # Retrieve context from FAISS
    with vector_db_lock:
        docs = vector_db.similarity_search(data['query'], k=4)
    
    context = "\n\n".join([doc.page_content for doc in docs])

    # Structured extraction + scoring
    structured = extract_structured_case(data["query"])
    precedent_similarity = compute_precedent_similarity(data["query"], docs)
    score, band, rationale = score_outcome(structured, precedent_similarity)

    # HIGH-QUALITY LEGAL PREDICTOR PROMPT
    system_prompt = "You are a Senior Legal Analyst specialized in Indian Case Law."
    prediction_prompt = f"""
    SYSTEM CONTEXT: You are analyzing a legal dispute based on retrieved case documents.
    USER QUERY: {data['query']}
    
    RETRIEVED DOCUMENT CONTEXT:
    ---
    {context}
    ---

    STRUCTURED FACTS:
    - Issue: {structured.get("issue")}
    - Jurisdiction: {structured.get("jurisdiction")}
    - Statute: {structured.get("statute")}
    - Relief sought: {structured.get("relief_sought")}
    - Key facts: {structured.get("key_facts")}
    - Precedent similarity (0-1): {precedent_similarity}
    - Likelihood band (rule-based): {band} ({score})
    
    INSTRUCTIONS:
    1. Analyze the USER QUERY against the RETRIEVED DOCUMENT CONTEXT.
    2. Identify the core legal issues and relevant statutes/Acts.
    3. Cite specific observations from the document context if applicable.
    4. Provide a "Likely Outcome" prediction with a reasoning section.
    5. Mention potential risks or counter-arguments.
    6. DISCLAIMER: Always end with: 'This is an AI simulation and does not constitute formal legal advice.'
    
    Format the output with clear headers and bullet points.
    """
    
    prediction = get_groq_completion(prediction_prompt, system_prompt)

    structured_lines = ["**Structured Outcome Model**"]
    field_map = [
        ("Issue", structured.get("issue")),
        ("Jurisdiction", structured.get("jurisdiction")),
        ("Statute", structured.get("statute")),
        ("Relief sought", structured.get("relief_sought")),
        ("Key facts", structured.get("key_facts")),
    ]
    for label, value in field_map:
        if value:
            structured_lines.append(f"- {label}: {value}")

    structured_lines.append(f"- Precedent similarity: {precedent_similarity}")
    structured_lines.append(f"- Likelihood band: **{band}** (score: {score})")
    structured_lines.append(f"- Scoring notes: {', '.join(rationale)}")

    structured_md = "\n".join(structured_lines)

    combined = f"{structured_md}\n\n{prediction}"
    return jsonify({
        "prediction": combined,
        "structured": structured,
        "score": score,
        "band": band,
        "precedent_similarity": precedent_similarity,
        "rationale": rationale,
    })

# -----------------------------------------------------------------------------
# 4B. DOCUMENT LOCKER
# -----------------------------------------------------------------------------

@app.route('/api/documents', methods=['GET'])
def list_documents():
    with get_db_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
        documents = [dict(row) for row in rows]
    return jsonify({"documents": documents})

@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    original_name = file.filename
    safe_name = secure_filename(original_name)
    if not safe_name:
        return jsonify({"error": "Invalid filename"}), 400

    doc_id = str(uuid4())
    filename = f"{doc_id}__{safe_name}"
    file_path = DOCS_DIR / filename

    file.save(file_path)
    size_bytes = file_path.stat().st_size
    content_type = file.content_type or "application/octet-stream"

    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, original_name, content_type, size_bytes, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (doc_id, filename, original_name, content_type, size_bytes, datetime.now().isoformat()),
        )

    return jsonify({"message": "Document uploaded", "id": doc_id})

@app.route('/api/documents/<doc_id>', methods=['GET'])
def download_document(doc_id: str):
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()

    if not row:
        return jsonify({"error": "Document not found"}), 404

    file_path = DOCS_DIR / row["filename"]
    if not file_path.exists():
        return jsonify({"error": "File missing on server"}), 410

    return send_from_directory(
        DOCS_DIR,
        row["filename"],
        as_attachment=True,
        download_name=row["original_name"],
    )

@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id: str):
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
        if not row:
            return jsonify({"error": "Document not found"}), 404

        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    file_path = DOCS_DIR / row["filename"]
    if file_path.exists():
        file_path.unlink()

    return jsonify({"message": "Document deleted"})

# -----------------------------------------------------------------------------
# 4C. CASE REMINDERS (EMAIL)
# -----------------------------------------------------------------------------

@app.route('/api/reminders', methods=['GET'])
def list_reminders():
    with get_db_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reminders ORDER BY reminder_date DESC"
        ).fetchall()
        reminders = [dict(row) for row in rows]
    return jsonify({"reminders": reminders})

@app.route('/api/reminders', methods=['POST'])
def create_reminder():
    data = request.get_json() or {}
    required = ["email", "case_title", "court_date", "reminder_time"]
    if not all(k in data and data[k] for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        court_date = datetime.strptime(data["court_date"], "%Y-%m-%d")
        reminder_time = datetime.strptime(data["reminder_time"], "%H:%M").time()
        reminder_date = datetime.combine(court_date.date(), reminder_time)
    except ValueError:
        return jsonify({"error": "Invalid date/time format"}), 400

    reminder_id = str(uuid4())
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO reminders (id, email, case_title, court_date, note, reminder_date, sent_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reminder_id,
                data["email"],
                data["case_title"],
                court_date.date().isoformat(),
                data.get("note"),
                reminder_date.isoformat(),
                None,
            ),
        )

    return jsonify({"message": "Reminder scheduled", "id": reminder_id})

@app.route('/api/reminders/<reminder_id>', methods=['DELETE'])
def delete_reminder(reminder_id: str):
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT id FROM reminders WHERE id = ?",
            (reminder_id,),
        ).fetchone()
        if not row:
            return jsonify({"error": "Reminder not found"}), 404

        conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))

    return jsonify({"message": "Reminder deleted"})

# -----------------------------------------------------------------------------
# 5. SCALABILITY & THREADING NOTES
# -----------------------------------------------------------------------------
"""
Scalability Limitations:
1. In-Memory Vector Store: The 'vector_db' is stored in memory. For a real production app 
   with multiple concurrent users, you should migrate to a persistent database like 
   Qdrant, Pinecone, or a PGVector-enabled Postgres.
2. Global Index: Currently, all users share the same 'vector_db'. In production, use 
   session IDs or user IDs to isolate vector indexes.
3. Synchronous Requests: Flask's default server is not for production. Use Gunicorn 
   with Gevent or Uvicorn to handle high concurrency.
"""

init_db()

if __name__ == '__main__':
    reminder_thread = threading.Thread(target=reminder_worker, daemon=True)
    reminder_thread.start()
    # Use 0.0.0.0 for Docker compatibility
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=False)
