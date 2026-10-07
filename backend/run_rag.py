#!/usr/bin/env python3
"""
RAG System Runner
This script runs the RAG system with the available PDF files in the dataset directory.
"""

import os
import sys
import traceback

def main():
    try:
        # Import required modules
        print("[INFO] Importing required modules...")
        import requests
        import fitz
        import re
        import json
        import argparse
        import csv
        import time
        import datetime
        from nltk.tokenize import word_tokenize
        from sentence_transformers import SentenceTransformer
        import chromadb
        import ollama
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        import textwrap
        import pdfplumber
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import defaultdict
        from typing import List, Dict, Any
        print("[OK] All modules imported successfully!")
        
        # Initialize the system
        print("\n[INFO] Initializing RAG system...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.EphemeralClient()
        collection_name = "documents"
        
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        pdf_dir = os.path.join(backend_dir, "dataset")
        log_file = os.path.join(project_root, "time_report_ingestion_updated.csv")
        # CLI configuration
        parser = argparse.ArgumentParser(description="Run RAG pipeline and test retrieval.")
        parser.add_argument("--similarity-threshold", type=float, default=0.30, help="Similarity threshold for filtering retrieved chunks.")
        parser.add_argument("--top-k", type=int, default=3, help="Number of top results to consider per query.")
        args = parser.parse_args()
        print(f"[INFO] Retrieval config -> top_k={args.top_k}, similarity_threshold={args.similarity_threshold}")
        
        # Clean up existing collection
        existing_collections = [col.name for col in client.list_collections()]
        if collection_name in existing_collections:
            client.delete_collection(collection_name)
            print(f"[INFO] Deleted existing collection '{collection_name}'.")
        
        # Create new collection
        collection = client.create_collection(name=collection_name)
        print(f"[INFO] Created collection '{collection_name}'.")
        
        # PDF processing functions
        def extract_text_from_pdf(pdf_path):
            doc = fitz.open(pdf_path)
            return " ".join(page.get_text("text") for page in doc)

        def clean_text(text):
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        def chunk_text(text, chunk_size=1024, overlap=200):
            tokens = word_tokenize(text)
            chunks = []
            for i in range(0, len(tokens), chunk_size - overlap):
                chunks.append(" ".join(tokens[i:i + chunk_size]))
            return chunks

        def analyze_pdf(file_path):
            try:
                with pdfplumber.open(file_path) as pdf:
                    num_pages = len(pdf.pages)
                    total_words = 0
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        total_words += len(text.split())
                    return num_pages, total_words
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                return 0, 0

        def process_pdf(file_path, filename):
            insight_time = embedding_time = store_time = 0.0
            start_total = time.perf_counter()
            print(f"\n[FILE {datetime.datetime.now().strftime('%H:%M:%S')}] Starting: {filename}")

            try:
                full_text = clean_text(extract_text_from_pdf(file_path))
                num_pages, total_words = analyze_pdf(file_path)

                # Step 1: Embedding
                chunks = chunk_text(full_text)
                start_embed = time.perf_counter()
                embeddings = model.encode(chunks)
                end_embed = time.perf_counter()
                embedding_time = end_embed - start_embed

                # Step 2: Store to Chroma
                start_store = time.perf_counter()
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    doc_id = f"{filename}_{idx}"
                    metadata = {
                        "id": doc_id,
                        "type": "pdf_chunk",
                        "source_pdf": filename,
                        "chunk_index": idx,
                    }

                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding.tolist()],
                        ids=[doc_id],
                        metadatas=[metadata]
                    )
                end_store = time.perf_counter()
                store_time = end_store - start_store

                total_time = time.perf_counter() - start_total

                print(f"[DONE] Finished {filename} | Embed: {embedding_time:.2f}s | Store: {store_time:.2f}s | Total: {total_time:.2f}s")

                with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        filename,
                        num_pages,
                        total_words,
                        f"{embedding_time:.2f}",
                        f"{store_time:.2f}",
                        f"{total_time:.2f}",
                        "success",
                        ""
                    ])

            except Exception as e:
                print(f"[ERROR] Error with {filename}: {e}")
                with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        filename,
                        0,
                        0,
                        f"{embedding_time:.2f}",
                        f"{store_time:.2f}",
                        f"{time.perf_counter() - start_total:.2f}",
                        "failed",
                        str(e)
                    ])
                return filename

            return None

        # Process all PDF files
        print(f"\n[INFO] Processing PDF files from {pdf_dir}...")
        failed_pdfs = []
        pdf_files = [
            (os.path.join(pdf_dir, filename), filename)
            for filename in os.listdir(pdf_dir)
            if filename.lower().endswith(".pdf")
        ]

        print(f"Found {len(pdf_files)} PDF files to process...")
        
        # Initialize log file
        with open(log_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "num_pages", "total_words", 
                "embedding_time", "store_time", "total_time", 
                "status", "error_message"
            ])

        # Process each PDF (limit to first 3 for demo)
        for file_path, filename in pdf_files[:3]:
            result = process_pdf(file_path, filename)
            if result:
                failed_pdfs.append(result)

        if failed_pdfs:
            csv_path = os.path.join(pdf_dir, "failed_to_map_pdf.csv")
            with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Failed PDF Filenames"])
                for pdf_name in failed_pdfs:
                    writer.writerow([pdf_name])
            print(f"[INFO] {len(failed_pdfs)} failed PDFs saved to {csv_path}")
        else:
            print("[SUCCESS] All PDFs processed successfully!")

        # Test the RAG system
        print(f"\n[INFO] Testing RAG system...")
        
        # Retrieve all pdf_chunk entries
        pdf_chunk_entries = collection.get(where={"type": "pdf_chunk"})
        print(f"PDF chunks stored: {len(pdf_chunk_entries['ids'])}")

        # RAG query functions
        def convert_query_to_embedding(query):
            return model.encode(query)

        def retrieve_similar_chunks(query, top_k=50, similarity_threshold=0.4):
            try:
                query_embedding = convert_query_to_embedding(query)
                collection = client.get_collection("documents")
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "embeddings", "metadatas"]
                )

                if results.get("embeddings") is None:
                    raise ValueError("Embeddings were not returned in query results!")

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

                return similar_chunks

            except Exception as e:
                raise RuntimeError(f"Error retrieving similar chunks: {str(e)}")

        def process_query_and_retrieve(query, top_k=5, similarity_threshold=0.65):
            try:
                similar_chunks = retrieve_similar_chunks(query, top_k, similarity_threshold)
                
                if not similar_chunks:
                    print("No relevant chunks found above the similarity threshold.")
                    return None
                
                print(f"Found {len(similar_chunks)} relevant chunks:")
                for i, chunk in enumerate(similar_chunks, start=1):
                    print(f"{i}. {chunk['document_id']} (Score: {chunk['similarity_score']:.4f})")
                    print(f"Text: {chunk['chunk_text'][:300]}...")
                    print("-" * 50)
                
                return similar_chunks

            except Exception as e:
                print(f"Error: {str(e)}")
                return None

        # Test queries
        test_queries = [
            "What is machine learning?",
            "What are neural networks?",
            "What is supervised learning?",
            "What is deep learning?"
        ]

        for query in test_queries:
            print(f"\n[QUERY] {query}")
            print("=" * 60)
            filtered_chunks = process_query_and_retrieve(query, top_k=args.top_k, similarity_threshold=args.similarity_threshold)
            if filtered_chunks:
                print(f"[SUCCESS] Found {len(filtered_chunks)} relevant chunks for query: {query}")
            else:
                print(f"[WARN] No relevant chunks found for query: {query}")
            print()

        print("[DONE] RAG system test completed successfully!")

    except Exception as e:
        print(f"[ERROR] Error running RAG system: {e}")
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
