
#!/usr/bin/env python3
"""
RAG System Web Application
Flask backend for the RAG system with web interface
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import traceback
import time
import datetime
import re
import json
import csv
import threading
from collections import deque
from functools import lru_cache
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
try:
    import chromadb
except Exception:
    chromadb = None
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fitz
import pdfplumber

# Base directory (project root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Configure template folder to look into sibling frontend directory
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'frontend'))
CORS(app, supports_credentials=True)

# Configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Global variables for RAG system
model = None
client = None
collection = None
collection_name = "documents"

# Query history and analytics
query_history = deque(maxlen=1000)  # Store last 1000 queries
chat_sessions = {}  # Store chat sessions

# Background PDF processing state
pdf_processing_lock = threading.Lock()
pdf_processing_cancel_event = threading.Event()
pdf_processing_thread = None
pdf_processing_state = {
    "running": False,
    "cancel_requested": False,
    "cancelled": False,
    "done": False,
    "started_at": None,
    "finished_at": None,
    "limit": None,
    "total": 0,
    "successful": 0,
    "failed": 0,
    "current_file": None,
    "failed_files": [],
    "error": None,
}

def _make_history_id():
    return str(time.time_ns())


class InMemoryCollection:
    def __init__(self, name: str):
        self.name = name
        self._items = []

    def add(self, documents, embeddings, ids, metadatas):
        for doc, emb, _id, meta in zip(documents, embeddings, ids, metadatas):
            self._items.append({
                "id": _id,
                "document": doc,
                "embedding": np.array(emb, dtype=np.float32),
                "metadata": meta or {},
            })

    def query(self, query_embeddings, n_results=5, include=None):
        if not self._items:
            return {"documents": [[]], "embeddings": [[]], "metadatas": [[]]}

        q = np.array(query_embeddings[0], dtype=np.float32)
        embs = np.stack([it["embedding"] for it in self._items], axis=0)

        # cosine similarity
        q_norm = np.linalg.norm(q) + 1e-12
        e_norm = np.linalg.norm(embs, axis=1) + 1e-12
        sims = (embs @ q) / (e_norm * q_norm)

        top_idx = np.argsort(-sims)[: max(0, int(n_results))]
        docs = [self._items[i]["document"] for i in top_idx]
        embeddings = [self._items[i]["embedding"].tolist() for i in top_idx]
        metas = [self._items[i]["metadata"] for i in top_idx]

        return {"documents": [docs], "embeddings": [embeddings], "metadatas": [metas]}

    def get(self, where=None):
        items = self._items
        if where:
            def match(meta):
                for k, v in where.items():
                    if meta.get(k) != v:
                        return False
                return True

            items = [it for it in items if match(it["metadata"])]

        return {
            "ids": [it["id"] for it in items],
            "metadatas": [it["metadata"] for it in items],
            "documents": [it["document"] for it in items],
        }

    def delete(self, where=None):
        if not where:
            self._items = []
            return

        def match(meta):
            for k, v in where.items():
                if meta.get(k) != v:
                    return False
            return True

        self._items = [it for it in self._items if not match(it["metadata"])]

class RAGSystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.EphemeralClient() if chromadb is not None else None
        self.collection_name = "documents"
        self.pdf_dir = DATASET_DIR
        self.log_file = os.path.join(BASE_DIR, "time_report_ingestion_updated.csv")
        
        # Initialize collection
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup ChromaDB collection"""
        if self.client is None:
            self.collection = InMemoryCollection(self.collection_name)
            print(f"[OK] Using in-memory collection '{self.collection_name}' (ChromaDB unavailable).")
            return

        existing_collections = [col.name for col in self.client.list_collections()]
        if self.collection_name in existing_collections:
            self.client.delete_collection(self.collection_name)
        
        self.collection = self.client.create_collection(name=self.collection_name)
        print(f"[OK] Created collection '{self.collection_name}'.")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file with fallback methods"""
        text_content = []
        
        # Method 1: Try PyMuPDF (fitz) first
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                if pdf_processing_cancel_event.is_set():
                    doc.close()
                    raise RuntimeError("PDF processing cancelled")
                try:
                    page = doc[page_num]
                    page_text = page.get_text("text")
                    if page_text and page_text.strip():
                        text_content.append(page_text)
                except Exception as e:
                    print(f"  Warning: Could not extract text from page {page_num + 1}: {e}")
                    continue
            doc.close()
            
            if text_content:
                return " ".join(text_content)
        except Exception as e:
            print(f"  Warning: PyMuPDF extraction failed: {e}")
        
        # Method 2: Fallback to pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    if pdf_processing_cancel_event.is_set():
                        raise RuntimeError("PDF processing cancelled")
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_content.append(page_text)
                    except Exception as e:
                        print(f"  Warning: Could not extract text from page: {e}")
                        continue
        except Exception as e:
            print(f"  Warning: pdfplumber extraction failed: {e}")
        
        if not text_content:
            raise ValueError(f"No text could be extracted from {pdf_path}")
        
        return " ".join(text_content)

    def clean_text(self, text):
        """Clean extracted text - improved to preserve more content"""
        if not text or not text.strip():
            return ""
        
        # Remove null bytes and control characters (except newlines and tabs)
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace but preserve sentence structure
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines to double newline
        
        # Remove excessive special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\'\"\/\@\#\$\%\&\*\+\=\<\>]', ' ', text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text, chunk_size=1024, overlap=200):
        """Split text into chunks with validation"""
        if not text or not text.strip():
            return []
        
        # Ensure minimum chunk size
        if chunk_size < 100:
            chunk_size = 100
        if overlap >= chunk_size:
            overlap = chunk_size // 4
        
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            print(f"  Warning: Tokenization failed, using simple split: {e}")
            # Fallback to simple word split
            tokens = text.split()
        
        if not tokens:
            return []
        
        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(tokens), step):
            chunk = " ".join(tokens[i:i + chunk_size])
            if chunk and chunk.strip() and len(chunk.strip()) > 10:  # Minimum chunk length
                chunks.append(chunk)
        
        return chunks if chunks else [text[:chunk_size]]  # Fallback to single chunk

    def analyze_pdf(self, file_path):
        """Analyze PDF file with fallback methods"""
        num_pages = 0
        total_words = 0
        
        # Method 1: Try pdfplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                for page in pdf.pages:
                    if pdf_processing_cancel_event.is_set():
                        return 0, 0
                    try:
                        text = page.extract_text() or ""
                        total_words += len(text.split())
                    except:
                        continue
                if num_pages > 0:
                    return num_pages, total_words
        except Exception as e:
            print(f"  Warning: pdfplumber analysis failed: {e}")
        
        # Method 2: Fallback to PyMuPDF
        try:
            doc = fitz.open(file_path)
            num_pages = len(doc)
            for page_num in range(num_pages):
                if pdf_processing_cancel_event.is_set():
                    doc.close()
                    return 0, 0
                try:
                    page = doc[page_num]
                    text = page.get_text("text") or ""
                    total_words += len(text.split())
                except:
                    continue
            doc.close()
        except Exception as e:
            print(f"  Warning: PyMuPDF analysis failed: {e}")
        
        return num_pages, total_words

    def process_pdf(self, file_path, filename):
        """Process a single PDF file with improved error handling"""
        start_total = time.perf_counter()
        print(f"\n[FILE {datetime.datetime.now().strftime('%H:%M:%S')}] Starting: {filename}")

        try:
            # Validate file exists and is readable
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            if pdf_processing_cancel_event.is_set():
                raise RuntimeError("PDF processing cancelled")
            
            if os.path.getsize(file_path) == 0:
                raise ValueError(f"File is empty: {filename}")
            
            # Extract text
            try:
                raw_text = self.extract_text_from_pdf(file_path)
            except Exception as e:
                raise ValueError(f"Text extraction failed: {str(e)}")

            if pdf_processing_cancel_event.is_set():
                raise RuntimeError("PDF processing cancelled")
            
            if not raw_text or not raw_text.strip():
                raise ValueError(f"No text content found in PDF: {filename}")
            
            # Clean text
            full_text = self.clean_text(raw_text)
            
            if not full_text or len(full_text.strip()) < 10:
                raise ValueError(f"Cleaned text is too short or empty: {filename}")
            
            # Analyze PDF
            num_pages, total_words = self.analyze_pdf(file_path)
            
            # Create chunks
            chunks = self.chunk_text(full_text)

            if pdf_processing_cancel_event.is_set():
                raise RuntimeError("PDF processing cancelled")
            
            if not chunks or len(chunks) == 0:
                raise ValueError(f"No valid chunks created from: {filename}")
            
            print(f"  Extracted {len(chunks)} chunks from {num_pages} pages ({total_words} words)")
            
            # Create embeddings in batches to handle memory
            try:
                embeddings = self.model.encode(chunks, show_progress_bar=False, batch_size=32)
            except Exception as e:
                raise ValueError(f"Embedding creation failed: {str(e)}")

            if pdf_processing_cancel_event.is_set():
                raise RuntimeError("PDF processing cancelled")
            
            if len(embeddings) != len(chunks):
                raise ValueError(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}")
            
            # Sanitize filename for use in doc_id (remove special characters)
            safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
            
            # Store to ChromaDB in batches
            batch_size = 100
            for batch_start in range(0, len(chunks), batch_size):
                if pdf_processing_cancel_event.is_set():
                    raise RuntimeError("PDF processing cancelled")
                batch_end = min(batch_start + batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]
                batch_embeddings = embeddings[batch_start:batch_end]
                
                batch_ids = []
                batch_documents = []
                batch_embeddings_list = []
                batch_metadatas = []
                
                for idx, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    global_idx = batch_start + idx
                    doc_id = f"{safe_filename}_{global_idx}"
                    
                    # Validate chunk
                    if not chunk or not chunk.strip() or len(chunk.strip()) < 5:
                        continue
                    
                    batch_ids.append(doc_id)
                    batch_documents.append(chunk)
                    batch_embeddings_list.append(embedding.tolist())
                    batch_metadatas.append({
                        "id": doc_id,
                        "type": "pdf_chunk",
                        "source_pdf": filename,
                        "chunk_index": global_idx,
                    })
                
                if batch_ids:
                    try:
                        self.collection.add(
                            documents=batch_documents,
                            embeddings=batch_embeddings_list,
                            ids=batch_ids,
                            metadatas=batch_metadatas
                        )
                    except Exception as e:
                        print(f"  Warning: Failed to add batch {batch_start}-{batch_end}: {e}")
                        # Try adding one by one
                        for i, (doc_id, doc, emb, meta) in enumerate(zip(batch_ids, batch_documents, batch_embeddings_list, batch_metadatas)):
                            if pdf_processing_cancel_event.is_set():
                                raise RuntimeError("PDF processing cancelled")
                            try:
                                self.collection.add(
                                    documents=[doc],
                                    embeddings=[emb],
                                    ids=[doc_id],
                                    metadatas=[meta]
                                )
                            except Exception as e2:
                                print(f"  Warning: Failed to add chunk {doc_id}: {e2}")

            total_time = time.perf_counter() - start_total
            print(f"[DONE] Finished {filename} | {len(chunks)} chunks | {total_time:.2f}s")
            return True

        except FileNotFoundError as e:
            print(f"[ERROR] File not found: {filename} - {e}")
            return False
        except RuntimeError as e:
            if "cancelled" in str(e).lower():
                print(f"[CANCEL] Cancelled processing: {filename}")
                return False
            print(f"[ERROR] Runtime error for {filename}: {e}")
            return False
        except ValueError as e:
            print(f"[ERROR] Processing error for {filename}: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error with {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_all_pdfs(self, limit=None):
        """Process all PDF files in the dataset directory with better error tracking"""
        if not os.path.exists(self.pdf_dir):
            print(f"[ERROR] Dataset directory not found: {self.pdf_dir}")
            return 0, 0, 0
        
        pdf_files = []
        for filename in os.listdir(self.pdf_dir):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(self.pdf_dir, filename)
                # Skip if file is too large (>100MB) or empty
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size == 0:
                        print(f"[SKIP] Empty file: {filename}")
                        continue
                    if file_size > 100 * 1024 * 1024:  # 100MB
                        print(f"[SKIP] File too large ({file_size / 1024 / 1024:.1f}MB): {filename}")
                        continue
                    pdf_files.append((file_path, filename))
                except Exception as e:
                    print(f"[SKIP] Cannot access file {filename}: {e}")
                    continue
        
        if not pdf_files:
            print(f"[WARNING] No valid PDF files found in {self.pdf_dir}")
            return 0, 0, 0
        
        if limit:
            pdf_files = pdf_files[:limit]
        
        print(f"[INFO] Processing {len(pdf_files)} PDF file(s)...")
        
        successful = 0
        failed = 0
        failed_files = []
        
        for file_path, filename in pdf_files:
            try:
                if self.process_pdf(file_path, filename):
                    successful += 1
                else:
                    failed += 1
                    failed_files.append(filename)
            except Exception as e:
                print(f"[ERROR] Unexpected error processing {filename}: {e}")
                failed += 1
                failed_files.append(filename)
        
        if failed_files:
            print(f"\n[INFO] Failed to process {len(failed_files)} file(s):")
            for f in failed_files:
                print(f"  - {f}")
        
        return successful, failed, len(pdf_files)

    def generate_answer(self, question, chunks, target_words=125):
        """Generate a concise answer (100-150 words) from retrieved chunks"""
        if not chunks:
            return "I couldn't find relevant information to answer your question. Please try rephrasing your query or adjusting the similarity threshold."
        
        # Combine top chunks
        combined_text = "\n\n".join([chunk['chunk_text'] for chunk in chunks[:3]])
        
        # Extract key sentences
        sentences = sent_tokenize(combined_text)
        
        # Simple answer generation: select most relevant sentences
        answer_sentences = []
        word_count = 0
        target_min = target_words - 25
        target_max = target_words + 25
        
        # Prioritize sentences that contain question keywords
        question_words = set(question.lower().split())
        
        scored_sentences = []
        for sent in sentences:
            sent_words = set(sent.lower().split())
            # Score based on keyword overlap and length
            overlap = len(question_words.intersection(sent_words))
            score = overlap + (1 if 20 < len(sent.split()) < 40 else 0)
            scored_sentences.append((score, sent))
        
        # Sort by relevance
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        # Build answer
        for score, sent in scored_sentences:
            sent_words = len(sent.split())
            if word_count + sent_words <= target_max:
                answer_sentences.append(sent)
                word_count += sent_words
                if word_count >= target_min:
                    break
        
        # If we don't have enough, add more sentences
        if word_count < target_min:
            for score, sent in scored_sentences:
                if sent not in answer_sentences:
                    sent_words = len(sent.split())
                    if word_count + sent_words <= target_max:
                        answer_sentences.append(sent)
                        word_count += sent_words
                        if word_count >= target_min:
                            break
        
        answer = " ".join(answer_sentences)
        
        # Ensure answer is within target range
        words = answer.split()
        if len(words) > target_max:
            answer = " ".join(words[:target_max]) + "..."
        elif len(words) < target_min and len(sentences) > len(answer_sentences):
            # Add one more sentence if possible
            for score, sent in scored_sentences:
                if sent not in answer_sentences:
                    answer += " " + sent
                    break
        
        return answer.strip()

    def query(self, question, top_k=5, similarity_threshold=0.3, generate_answer_flag=True):
        """Query the RAG system"""
        try:
            # Convert query to embedding
            query_embedding = self.model.encode(question)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "embeddings", "metadatas"]
            )

            if results.get("embeddings") is None:
                return [], None

            similar_chunks = []
            
            for i, doc in enumerate(results["documents"][0]):
                chunk_text = doc
                document_id = results["metadatas"][0][i].get("id")
                document_type = results["metadatas"][0][i].get("type")
                chunk_embedding = results["embeddings"][0][i]
                
                similarity_score = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                
                if similarity_score >= similarity_threshold and document_type == "pdf_chunk":
                    similar_chunks.append({
                        "document_id": document_id,
                        "document_type": document_type,
                        "source_pdf": results["metadatas"][0][i].get("source_pdf"),
                        "chunk_text": chunk_text,
                        "similarity_score": similarity_score
                    })

            # Sort by similarity score
            similar_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Generate answer if requested
            answer = None
            if generate_answer_flag and similar_chunks:
                answer = self.generate_answer(question, similar_chunks)
            
            return similar_chunks, answer

        except Exception as e:
            print(f"Error in query: {str(e)}")
            return [], None

    def get_stats(self):
        """Get system statistics"""
        try:
            pdf_chunk_entries = self.collection.get(where={"type": "pdf_chunk"})
            
            # Get unique PDFs
            unique_pdfs = set()
            if pdf_chunk_entries.get('metadatas'):
                for meta in pdf_chunk_entries['metadatas']:
                    if meta.get('source_pdf'):
                        unique_pdfs.add(meta['source_pdf'])
            
            return {
                "total_chunks": len(pdf_chunk_entries['ids']),
                "total_documents": len(unique_pdfs),
                "collection_name": self.collection_name,
                "model_name": "all-MiniLM-L6-v2",
                "query_count": len(query_history)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_document_list(self):
        """Get list of all processed documents"""
        try:
            pdf_chunk_entries = self.collection.get(where={"type": "pdf_chunk"})
            documents = {}
            
            if pdf_chunk_entries.get('metadatas'):
                for meta in pdf_chunk_entries['metadatas']:
                    source_pdf = meta.get('source_pdf')
                    if source_pdf:
                        if source_pdf not in documents:
                            documents[source_pdf] = {
                                "name": source_pdf,
                                "chunk_count": 0
                            }
                        documents[source_pdf]["chunk_count"] += 1
            
            return list(documents.values())
        except Exception as e:
            return []

# Initialize RAG system
rag_system = None

def init_rag_system():
    """Initialize the RAG system"""
    global rag_system
    try:
        rag_system = RAGSystem()
        return True
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"})
    
    stats = rag_system.get_stats()
    return jsonify(stats)

@app.route('/api/process_pdfs', methods=['POST'])
def process_pdfs():
    """Start background processing of PDF files"""
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"})
    
    try:
        data = request.get_json()
        limit = data.get('limit', 3)  # Default to 3 PDFs

        global pdf_processing_thread
        with pdf_processing_lock:
            if pdf_processing_state.get("running"):
                return jsonify({
                    "success": True,
                    "started": False,
                    "message": "PDF processing already running"
                })

            pdf_processing_cancel_event.clear()
            pdf_processing_state.update({
                "running": True,
                "cancel_requested": False,
                "cancelled": False,
                "done": False,
                "started_at": datetime.datetime.now().isoformat(),
                "finished_at": None,
                "limit": limit,
                "total": 0,
                "successful": 0,
                "failed": 0,
                "current_file": None,
                "failed_files": [],
                "error": None,
            })

            def _runner():
                try:
                    pdf_dir = rag_system.pdf_dir
                    pdf_files = []
                    for filename in os.listdir(pdf_dir):
                        if not filename.lower().endswith('.pdf'):
                            continue
                        file_path = os.path.join(pdf_dir, filename)
                        try:
                            file_size = os.path.getsize(file_path)
                            if file_size == 0:
                                continue
                            if file_size > 100 * 1024 * 1024:
                                continue
                            pdf_files.append((file_path, filename))
                        except Exception:
                            continue

                    if limit:
                        pdf_files = pdf_files[:limit]

                    with pdf_processing_lock:
                        pdf_processing_state["total"] = len(pdf_files)

                    for file_path, filename in pdf_files:
                        if pdf_processing_cancel_event.is_set():
                            with pdf_processing_lock:
                                pdf_processing_state["cancelled"] = True
                            break

                        with pdf_processing_lock:
                            pdf_processing_state["current_file"] = filename

                        ok = False
                        try:
                            ok = rag_system.process_pdf(file_path, filename)
                        except Exception:
                            ok = False

                        with pdf_processing_lock:
                            if ok:
                                pdf_processing_state["successful"] += 1
                            else:
                                pdf_processing_state["failed"] += 1
                                pdf_processing_state["failed_files"].append(filename)

                except Exception as e:
                    with pdf_processing_lock:
                        pdf_processing_state["error"] = str(e)
                finally:
                    with pdf_processing_lock:
                        pdf_processing_state["running"] = False
                        pdf_processing_state["done"] = True
                        pdf_processing_state["current_file"] = None
                        pdf_processing_state["finished_at"] = datetime.datetime.now().isoformat()

            pdf_processing_thread = threading.Thread(target=_runner, daemon=True)
            pdf_processing_thread.start()

        return jsonify({
            "success": True,
            "started": True,
            "message": "PDF processing started"
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/process_pdfs/status', methods=['GET'])
def process_pdfs_status():
    with pdf_processing_lock:
        return jsonify({
            "success": True,
            **pdf_processing_state
        })


@app.route('/api/process_pdfs/stop', methods=['POST'])
def process_pdfs_stop():
    with pdf_processing_lock:
        if not pdf_processing_state.get("running"):
            return jsonify({
                "success": True,
                "stopped": False,
                "message": "No PDF processing is running"
            })
        pdf_processing_state["cancel_requested"] = True

    pdf_processing_cancel_event.set()
    return jsonify({
        "success": True,
        "stopped": True,
        "message": "Stop requested"
    })

@app.route('/api/query', methods=['POST'])
def query():
    """Query the RAG system"""
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"})
    
    try:
        data = request.get_json()
        question = data.get('question', '')
        top_k = data.get('top_k', 5)
        similarity_threshold = data.get('similarity_threshold', 0.3)
        session_id = data.get('session_id', 'default')
        generate_answer = data.get('generate_answer', True)
        
        if not question:
            return jsonify({"error": "Question is required"})
        
        start_time = time.time()
        results, answer = rag_system.query(question, top_k, similarity_threshold, generate_answer)
        query_time = time.time() - start_time
        
        # Store in query history
        query_entry = {
            "id": _make_history_id(),
            "question": question,
            "result_count": len(results),
            "timestamp": datetime.datetime.now().isoformat(),
            "query_time": round(query_time, 3),
            "session_id": session_id,
        }
        query_history.append(query_entry)
        
        return jsonify({
            "success": True,
            "results": results,
            "answer": answer,
            "count": len(results),
            "query_time": round(query_time, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "rag_system": rag_system is not None})

# PDF serving endpoints
@app.route('/api/pdf/<path:filename>')
def serve_pdf(filename):
    """Serve PDF file from dataset"""
    try:
        # Security: ensure filename doesn't contain path traversal
        safe_filename = os.path.basename(filename)
        pdf_dir = rag_system.pdf_dir if rag_system is not None else DATASET_DIR
        pdf_path = os.path.join(pdf_dir, safe_filename)
        
        if not os.path.exists(pdf_path):
            return jsonify({"error": "PDF not found"}), 404
        
        if not safe_filename.lower().endswith('.pdf'):
            return jsonify({"error": "Invalid file type"}), 400
        
        return send_file(pdf_path, mimetype='application/pdf')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pdf/list')
def list_pdfs():
    """List all available PDFs"""
    try:
        pdf_dir = rag_system.pdf_dir if rag_system is not None else DATASET_DIR
        if not os.path.exists(pdf_dir):
            return jsonify({"error": "Dataset directory not found"}), 404
        
        pdf_files = []
        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(pdf_dir, filename)
                try:
                    file_size = os.path.getsize(file_path)
                    pdf_files.append({
                        "filename": filename,
                        "size": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2)
                    })
                except:
                    continue
        
        return jsonify({
            "success": True,
            "pdfs": sorted(pdf_files, key=lambda x: x['filename']),
            "count": len(pdf_files)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pdf/remove', methods=['POST'])
def remove_pdf():
    data = request.get_json() or {}
    filename = data.get('filename', '')
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    safe_filename = os.path.basename(filename)
    if not safe_filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type"}), 400

    pdf_dir = rag_system.pdf_dir if rag_system is not None else DATASET_DIR
    pdf_path = os.path.join(pdf_dir, safe_filename)
    if not os.path.exists(pdf_path):
        return jsonify({"error": "PDF not found"}), 404

    try:
        if rag_system is not None:
            try:
                rag_system.collection.delete(where={"source_pdf": safe_filename})
            except Exception:
                pass

        os.remove(pdf_path)

        return jsonify({
            "success": True,
            "filename": safe_filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pdf/remove_all', methods=['POST'])
def remove_all_pdfs():
    pdf_dir = rag_system.pdf_dir if rag_system is not None else DATASET_DIR
    if not os.path.exists(pdf_dir):
        return jsonify({"error": "Dataset directory not found"}), 404

    removed_files = []
    failed_files = []

    try:
        if rag_system is not None:
            try:
                rag_system.collection.delete(where={"type": "pdf_chunk"})
            except Exception:
                pass

        for filename in os.listdir(pdf_dir):
            if not filename.lower().endswith('.pdf'):
                continue
            file_path = os.path.join(pdf_dir, filename)
            try:
                os.remove(file_path)
                removed_files.append(filename)
            except Exception:
                failed_files.append(filename)

        return jsonify({
            "success": True,
            "removed": removed_files,
            "failed": failed_files,
            "removed_count": len(removed_files),
            "failed_count": len(failed_files)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process a PDF file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400
    
    try:
        filename = secure_filename(file.filename)
        target_dir = rag_system.pdf_dir if rag_system is not None else DATASET_DIR
        os.makedirs(target_dir, exist_ok=True)
        filepath = os.path.join(target_dir, filename)
        file.save(filepath)
        
        # Check file size
        if os.path.getsize(filepath) > MAX_FILE_SIZE:
            os.remove(filepath)
            return jsonify({"error": "File size exceeds 50MB limit"}), 400
        
        if rag_system is None:
            return jsonify({
                "success": True,
                "message": f"File '{filename}' uploaded successfully (RAG not initialized, indexing skipped)",
                "filename": filename,
                "indexed": False
            })

        # Process the PDF
        success = rag_system.process_pdf(filepath, filename)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"File '{filename}' processed successfully",
                "filename": filename,
                "indexed": True
            })
        else:
            return jsonify({"error": f"Failed to process file '{filename}'"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/query_history', methods=['GET'])
def get_query_history():
    """Get query history"""
    limit = request.args.get('limit', 50, type=int)
    normalized = []
    for q in list(query_history):
        if isinstance(q, dict) and "id" not in q:
            q = {**q, "id": _make_history_id()}
        normalized.append(q)
    return jsonify({
        "success": True,
        "queries": normalized[-limit:],
        "total": len(normalized)
    })

@app.route('/api/query_history', methods=['DELETE'])
def clear_query_history():
    query_history.clear()
    return jsonify({
        "success": True,
        "total": 0
    })

@app.route('/api/query_history/<entry_id>', methods=['DELETE'])
def delete_query_history_entry(entry_id):
    if not entry_id:
        return jsonify({"error": "Entry id is required"}), 400

    removed = False
    for idx in range(len(query_history) - 1, -1, -1):
        q = query_history[idx]
        if isinstance(q, dict) and q.get("id") == entry_id:
            del query_history[idx]
            removed = True
            break

    if not removed:
        return jsonify({"error": "Entry not found"}), 404

    return jsonify({
        "success": True,
        "total": len(query_history)
    })

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get query analytics"""
    if not query_history:
        return jsonify({
            "success": True,
            "total_queries": 0,
            "average_results": 0,
            "top_queries": [],
            "query_timeline": []
        })
    
    # Calculate analytics
    total_queries = len(query_history)
    total_results = sum(q.get('result_count', 0) for q in query_history)
    average_results = total_results / total_queries if total_queries > 0 else 0
    
    # Get top queries by result count
    top_queries = sorted(
        query_history,
        key=lambda x: x.get('result_count', 0),
        reverse=True
    )[:10]
    
    # Query timeline (last 24 hours)
    now = datetime.datetime.now()
    timeline = []
    for i in range(24):
        hour_start = now - datetime.timedelta(hours=i+1)
        hour_end = now - datetime.timedelta(hours=i)
        count = 0
        for q in query_history:
            try:
                q_time = datetime.datetime.fromisoformat(q.get('timestamp', ''))
                if hour_start <= q_time <= hour_end:
                    count += 1
            except (ValueError, TypeError):
                continue
        timeline.append({
            "hour": hour_start.strftime('%H:00'),
            "count": count
        })
    timeline.reverse()
    
    return jsonify({
        "success": True,
        "total_queries": total_queries,
        "average_results": round(average_results, 2),
        "top_queries": [
            {
                "query": q.get('question', ''),
                "result_count": q.get('result_count', 0),
                "timestamp": q.get('timestamp', '')
            }
            for q in top_queries
        ],
        "query_timeline": timeline
    })

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of all processed documents"""
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"}), 400
    
    documents = rag_system.get_document_list()
    return jsonify({
        "success": True,
        "documents": documents,
        "count": len(documents)
    })

@app.route('/api/export', methods=['POST'])
def export_results():
    """Export query results to JSON or CSV"""
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"}), 400
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        results = data.get('results', [])
        format_type = data.get('format', 'json')  # 'json' or 'csv'
        
        if not results:
            return jsonify({"error": "No results to export"}), 400
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type == 'json':
            filename = f'export_{timestamp}.json'
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "query": query,
                    "timestamp": timestamp,
                    "result_count": len(results),
                    "results": results
                }, f, indent=2, ensure_ascii=False)
            
            return send_file(filepath, as_attachment=True, download_name=filename)
        
        elif format_type == 'csv':
            filename = f'export_{timestamp}.csv'
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Query', 'Source PDF', 'Similarity Score', 'Chunk Text'])
                for result in results:
                    writer.writerow([
                        query,
                        result.get('source_pdf', ''),
                        f"{result.get('similarity_score', 0):.4f}",
                        result.get('chunk_text', '')[:500]  # Limit text length
                    ])
            
            return send_file(filepath, as_attachment=True, download_name=filename)
        
        else:
            return jsonify({"error": "Invalid format. Use 'json' or 'csv'"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint with conversation context"""
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"}), 400
    
    try:
        data = request.get_json()
        question = data.get('question', '')
        session_id = data.get('session_id', 'default')
        top_k = data.get('top_k', 5)
        similarity_threshold = data.get('similarity_threshold', 0.3)
        context = data.get('context', [])  # Previous messages
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Get results (query returns tuple: similar_chunks, answer)
        results, answer = rag_system.query(question, top_k, similarity_threshold)
        
        # Store in query history
        query_entry = {
            "id": _make_history_id(),
            "question": question,
            "result_count": len(results),
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": session_id
        }
        query_history.append(query_entry)
        
        # Update chat session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_sessions[session_id].append({
            "role": "user",
            "content": question,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Generate response summary (simple concatenation of top results)
        response_text = ""
        if results:
            response_text = "\n\n".join([
                f"[Source: {r['source_pdf']}, Score: {r['similarity_score']:.2f}]\n{r['chunk_text'][:500]}"
                for r in results[:3]
            ])
        
        chat_sessions[session_id].append({
            "role": "assistant",
            "content": response_text,
            "results": results,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        return jsonify({
            "success": True,
            "results": results,
            "count": len(results),
            "response": response_text,
            "session_id": session_id
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Get chat history for a session"""
    if session_id in chat_sessions:
        return jsonify({
            "success": True,
            "messages": chat_sessions[session_id]
        })
    else:
        return jsonify({
            "success": True,
            "messages": []
        })

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    """Get query suggestions based on history"""
    if not query_history:
        # Default suggestions
        return jsonify({
            "success": True,
            "suggestions": [
                "What is machine learning?",
                "Explain neural networks",
                "What is supervised learning?",
                "How does deep learning work?",
                "What are clustering algorithms?"
            ]
        })
    
    # Get unique queries from history
    unique_queries = {}
    for q in query_history:
        query_text = q.get('question', '')
        if query_text:
            if query_text not in unique_queries:
                unique_queries[query_text] = 0
            unique_queries[query_text] += 1
    
    # Sort by frequency and return top 5
    top_queries = sorted(unique_queries.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return jsonify({
        "success": True,
        "suggestions": [q[0] for q in top_queries]
    })

if __name__ == '__main__':
    print("[INFO] Starting RAG System Web Application...")
    
    # Initialize RAG system
    if init_rag_system():
        print("[OK] RAG System initialized successfully!")
    else:
        print("[ERROR] Failed to initialize RAG system")
        if chromadb is None:
            print("[ERROR] ChromaDB failed to import. This is commonly caused by using an unsupported Python version.")
        print("[INFO] Starting web server anyway (UI will load, but RAG APIs will return initialization errors).")

    print("[INFO] Starting web server...")
    print("[INFO] Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
