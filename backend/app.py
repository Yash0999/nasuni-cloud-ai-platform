# Add these imports at the top of app.py
import os
import re
import sys
import base64
import bcrypt
import requests
from io import BytesIO
from datetime import datetime, timedelta
from collections import Counter

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)

from pymongo import MongoClient
from werkzeug.utils import secure_filename

# File parsing libraries
import PyPDF2
import docx
import mimetypes

# Embedding / semantic search
from sentence_transformers import SentenceTransformer, util
import numpy as np

# AWS S3
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Load .env
from dotenv import load_dotenv
load_dotenv()

# ---------------- Initialize OCR Dependencies Gracefully ----------------
# Initialize global variables first
PYTESSERACT_AVAILABLE = False
PDF2IMAGE_AVAILABLE = False

# Check if tesseract is installed and in PATH
try:
    import pytesseract
    from PIL import Image
    try:
        # Try to get tesseract version to verify it's working
        pytesseract.get_tesseract_version()
        PYTESSERACT_AVAILABLE = True
        print("✅ Tesseract OCR is available")
    except Exception as e:
        print(f"⚠️ Tesseract not available: {e}")
        PYTESSERACT_AVAILABLE = False
except ImportError:
    print("⚠️ pytesseract not installed")
    PYTESSERACT_AVAILABLE = False

# Check if pdf2image is available
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
    print("✅ pdf2image is available")
except ImportError:
    print("⚠️ pdf2image not installed")
    PDF2IMAGE_AVAILABLE = False

# ---------------- Flask App Setup ----------------
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', 'fallback-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=int(os.getenv('JWT_TOKEN_EXPIRE_HOURS', '24')))
jwt = JWTManager(app)

# CORS (restrict in production)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# ---------------- Globals ----------------
files_col = None
users_col = None
s3 = None
bucket_name = None
aws_region = os.getenv("AWS_REGION", "us-east-1")

# ---------------- Enhanced AI Services ----------------
class EnhancedAIServices:
    def __init__(self):
        self.api_token = os.getenv('HF_API_TOKEN')
        self.headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
        print("🤖 Enhanced AI Services Initialized")
        self.sentence_model = None
        self.model_loaded = False
        
        # Advanced model configurations
        self.models = {
            "summarization": "google/pegasus-xsum",  # Best for comprehensive summarization
            "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "keywords": "ml6team/keyphrase-extraction-distilbert-inspec",
            "ner": "dslim/bert-base-NER",
            "qa_primary": "deepset/roberta-base-squad2",
            "qa_fallback": "bert-large-uncased-whole-word-masking-finetuned-squad",
            "toxicity_primary": "unitary/toxic-bert",
            "toxicity_secondary": "facebook/roberta-hate-speech-dynabench-r4-target",
            "image_caption": "nlpconnect/vit-gpt2-image-captioning",
            "image_classification": "google/vit-base-patch16-224"
        }
        
        # Initialize local pipelines for faster processing
        self.local_pipelines = {}
        self._initialize_local_pipelines()

    def _initialize_local_pipelines(self):
        """Initialize local pipelines for faster processing"""
        try:
            # Check if transformers is available
            from transformers import pipeline
            import torch
            
            if torch.cuda.is_available():
                device = 0
            else:
                device = -1
                
            # Initialize QA pipelines
            self.local_pipelines["qa_primary"] = pipeline(
                "question-answering", 
                model=self.models["qa_primary"],
                device=device
            )
            self.local_pipelines["qa_fallback"] = pipeline(
                "question-answering", 
                model=self.models["qa_fallback"],
                device=device
            )
            
            # Initialize summarization pipeline
            self.local_pipelines["summarization"] = pipeline(
                "summarization",
                model=self.models["summarization"],
                device=device
            )
            
            print("✅ Local pipelines initialized")
        except Exception as e:
            print(f"⚠️ Local pipeline initialization failed: {e}")
            self.local_pipelines = {}

    def is_available(self):
        return bool(self.api_token)

    # ---------- Text extraction ----------
    def extract_text_from_file(self, file_content, filename, file_extension):
        """Extract text with improved PDF handling and fallbacks"""
        try:
            text = ""
            ext = file_extension.lower()

            if ext == 'pdf':
                try:
                    reader = PyPDF2.PdfReader(BytesIO(file_content))
                    total_pages = len(reader.pages)
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"{page_text}\n\n"
                    # Clean up extracted text
                    text = re.sub(r'\s+', ' ', text).strip()
                except Exception as e:
                    print(f"PyPDF2 extraction error: {e}")
                    # fallback to OCR if available
                    if PYTESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE:
                        try:
                            images = convert_from_bytes(file_content)
                            for img in images:
                                text += pytesseract.image_to_string(img) + "\n"
                        except Exception as e2:
                            print(f"pdf2image/pytesseract fallback failed: {e2}")
                            text = ""
                # if extracted text too short, hint for OCR
                if len(text.strip()) < 120 and not PYTESSERACT_AVAILABLE:
                    text = f"PDF document: {filename} ({total_pages if 'total_pages' in locals() else 'unknown'} pages). Text extraction limited. Install pytesseract/pdf2image for scanned PDFs."
            elif ext in ['txt', 'md', 'csv']:
                try:
                    text = file_content.decode('utf-8', errors='ignore')
                except Exception:
                    text = str(file_content)
            elif ext in ['docx', 'doc']:
                try:
                    doc = docx.Document(BytesIO(file_content))
                    for p in doc.paragraphs:
                        if p.text and p.text.strip():
                            text += p.text + "\n"
                except Exception as e:
                    print(f"docx parse error: {e}")
                    text = ""
            elif ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'] and PYTESSERACT_AVAILABLE:
                try:
                    img = Image.open(BytesIO(file_content))
                    text = pytesseract.image_to_string(img)
                except Exception as e:
                    print(f"Image OCR error: {e}")
                    text = f"Image file: {filename}"
            else:
                # Unknown binary file: return filename as placeholder
                text = f"File: {filename} (type: {ext})"
            text = re.sub(r'\s+', ' ', text).strip()
            return text if text else f"Content from {filename}"
        except Exception as e:
            print(f"Text extraction error for {filename}: {e}")
            return f"Content from {filename} (extraction failed: {str(e)})"

    # ---------- Improved Summarization ----------
    def split_text_into_chunks(self, text, max_chunk_size=800):
        """Improved text chunking for better summarization"""
        # First try to split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size and we have content, start new chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # If chunks are still too large, split by sentences
        if chunks and any(len(chunk) > max_chunk_size * 1.5 for chunk in chunks):
            refined_chunks = []
            for chunk in chunks:
                if len(chunk) > max_chunk_size * 1.5:
                    sentences = re.split(r'[.!?]+', chunk)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    current_sentences = ""
                    for sentence in sentences:
                        if len(current_sentences) + len(sentence) > max_chunk_size and current_sentences:
                            refined_chunks.append(current_sentences)
                            current_sentences = sentence
                        else:
                            current_sentences += " " + sentence if current_sentences else sentence
                    if current_sentences:
                        refined_chunks.append(current_sentences)
                else:
                    refined_chunks.append(chunk)
            chunks = refined_chunks
        
        return chunks

    def query_hf_api(self, model, inputs, parameters=None):
        if not self.is_available():
            return {"error": "Hugging Face API token not configured"}
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        try:
            payload = {"inputs": inputs}
            if parameters:
                payload["parameters"] = parameters
            response = requests.post(api_url, headers=self.headers, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                return {"error": "Model is loading", "status": "loading"}
            else:
                return {"error": f"API error: {response.status_code}", "detail": response.text}
        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except Exception as e:
            return {"error": f"API request failed: {str(e)}"}

    # ===== ENHANCED QA SYSTEM =====
    def document_qa_advanced(self, question, context):
        """Advanced QA with multiple models and context expansion"""
        if not question or not context:
            return {"error": "Question and context required"}
        
        # Step 1: Try primary model
        result = self._qa_with_model(question, context, "qa_primary")
        
        # Step 2: If answer is not satisfactory, try fallback model
        if not self._is_good_answer(result.get("answer", "")):
            fallback_result = self._qa_with_model(question, context, "qa_fallback")
            
            # Compare answers and choose the best one
            if self._is_better_answer(fallback_result.get("answer", ""), result.get("answer", "")):
                result = fallback_result
        
        # Step 3: If still not good, expand context and retry
        if not self._is_good_answer(result.get("answer", "")) and len(context) > 1000:
            expanded_context = self._expand_context(question, context)
            expanded_result = self._qa_with_model(question, expanded_context, "qa_primary")
            
            if self._is_better_answer(expanded_result.get("answer", ""), result.get("answer", "")):
                result = expanded_result
        
        return {"qa_result": result}
    
    def _qa_with_model(self, question, context, model_key):
        """QA with a specific model"""
        try:
            if model_key in self.local_pipelines:
                # Use local pipeline for speed
                result = self.local_pipelines[model_key]({
                    "question": question,
                    "context": context
                })
                return {
                    "answer": result["answer"],
                    "score": result["score"],
                    "model": self.models[model_key],
                    "context_used": len(context)
                }
            else:
                # Fallback to API
                result = self.query_hf_api(self.models[model_key], {
                    "question": question,
                    "context": context
                })
                
                if isinstance(result, dict) and "answer" in result:
                    return {
                        "answer": result["answer"],
                        "score": result.get("score", 0),
                        "model": self.models[model_key],
                        "context_used": len(context)
                    }
        except Exception as e:
            print(f"QA with {model_key} failed: {e}")
        
        return {"answer": "I couldn't find an answer to your question.", "score": 0}
    
    def _is_good_answer(self, answer):
        """Check if the answer is satisfactory"""
        if not answer or len(answer) < 10:
            return False
        
        # Check for generic answers
        generic_phrases = [
            "i don't know", "i cannot answer", "no information",
            "not mentioned", "not provided", "unable to answer"
        ]
        
        answer_lower = answer.lower()
        for phrase in generic_phrases:
            if phrase in answer_lower:
                return False
        
        return True
    
    def _is_better_answer(self, new_answer, old_answer):
        """Compare two answers and determine which is better"""
        if not self._is_good_answer(old_answer):
            return self._is_good_answer(new_answer)
        
        # Prefer longer, more detailed answers
        if len(new_answer) > len(old_answer) * 1.5:
            return True
        
        return False
    
    def _expand_context(self, question, context):
        """Expand context to find relevant information"""
        # Split context into sentences
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Use TF-IDF to find most relevant sentences
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences + [question])
            
            # Calculate similarity between question and sentences
            question_vec = tfidf_matrix[-1]
            sentence_vecs = tfidf_matrix[:-1]
            similarities = cosine_similarity(question_vec, sentence_vecs).flatten()
            
            # Get top 5 most relevant sentences
            top_indices = similarities.argsort()[-5:][::-1]
            expanded_context = '. '.join([sentences[i] for i in top_indices])
            
            return expanded_context
        except Exception as e:
            print(f"Context expansion failed: {e}")
            return context

    # ===== ENHANCED TOXICITY DETECTION =====
    def toxicity_check_advanced(self, text):
        """Advanced toxicity check with multiple models"""
        if not text or len(text.strip()) < 5:
            return {"error": "Text too short for toxicity check", "toxicity": []}
        
        results = {}
        
        # Check with primary model
        primary_result = self._toxicity_with_model(text, "toxicity_primary")
        if primary_result:
            results.update(primary_result)
        
        # Check with secondary model
        secondary_result = self._toxicity_with_model(text, "toxicity_secondary")
        if secondary_result:
            # Map hate speech labels to toxicity categories
            for label, score in secondary_result.items():
                if "hate" in label.lower():
                    results["hate_speech"] = max(results.get("hate_speech", 0), score)
                elif "offensive" in label.lower():
                    results["offensive_language"] = max(results.get("offensive_language", 0), score)
        
        # Calculate overall toxicity score
        if results:
            max_toxicity = max(results.values())
            results["overall_toxicity"] = max_toxicity
            
            # Add severity classification
            if max_toxicity > 0.8:
                results["severity"] = "high"
            elif max_toxicity > 0.5:
                results["severity"] = "medium"
            elif max_toxicity > 0.2:
                results["severity"] = "low"
            else:
                results["severity"] = "minimal"
        
        return {"toxicity_scores": results}
    
    def _toxicity_with_model(self, text, model_key):
        """Toxicity check with a specific model"""
        try:
            # Use API for toxicity models
            result = self.query_hf_api(self.models[model_key], text[:1000])
            
            if isinstance(result, list) and len(result) > 0:
                scores = {}
                for item in result:
                    if isinstance(item, dict) and 'label' in item:
                        scores[item['label']] = item['score']
                return scores
        except Exception as e:
            print(f"Toxicity check with {model_key} failed: {e}")
        
        return None

    # ===== ENHANCED SUMMARIZATION =====
    def summarize_large_text_advanced(self, text, max_chunk_size=1000):
        """Advanced summarization with comprehensive coverage"""
        if len(text) < 200:
            return {"summary": text, "model": "text_too_short", "chunks_processed": 1}
        
        try:
            # Step 1: Clean and preprocess text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Step 2: Extract key sentences (extractive summarization)
            key_sentences = self._extractive_summarization(text, max_sentences=10)
            
            # Step 3: If text is short, summarize directly
            if len(text) <= 1500:
                return self._summarize_with_pegasus(text)
            
            # Step 4: Split into meaningful chunks
            chunks = self._split_into_meaningful_chunks(text, max_chunk_size)
            
            # Step 5: Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 100:
                    chunk_summary = self._summarize_with_pegasus(chunk)
                    if chunk_summary and len(chunk_summary) > 30:
                        chunk_summaries.append(chunk_summary)
                        print(f"✅ Chunk {i+1} summarized ({len(chunk_summary)} chars)")
            
            # Step 6: Combine summaries
            combined_summary = ' '.join(chunk_summaries)
            
            # Step 7: Final summarization if needed
            if len(combined_summary) > 800:
                final_summary = self._summarize_with_pegasus(combined_summary)
            else:
                final_summary = combined_summary
            
            # Step 8: Ensure comprehensive coverage
            final_summary = self._ensure_comprehensive_coverage(final_summary, key_sentences)
            
            return {
                "summary": final_summary, 
                "model": "advanced-pegasus-xsum", 
                "chunks_processed": len(chunk_summaries), 
                "original_length": len(text),
                "key_points_covered": len(key_sentences)
            }
            
        except Exception as e:
            print(f"Advanced summarization failed: {e}")
            return self.summarize_large_text(text, max_chunk_size)
    
    def _extractive_summarization(self, text, max_sentences=10):
        """Extract key sentences using TF-IDF"""
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) <= max_sentences:
                return sentences
            
            # Create TF-IDF matrix
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).ravel()
            
            # Get top sentences
            top_indices = sentence_scores.argsort()[-max_sentences:][::-1]
            key_sentences = [sentences[i] for i in sorted(top_indices)]
            
            return key_sentences
        except Exception as e:
            print(f"Extractive summarization failed: {e}")
            return []
    
    def _split_into_meaningful_chunks(self, text, max_chunk_size):
        """Split text into meaningful chunks preserving context"""
        # First try to split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size and we have content, start new chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # If chunks are still too large, split by sentences
        if chunks and any(len(chunk) > max_chunk_size * 1.5 for chunk in chunks):
            refined_chunks = []
            for chunk in chunks:
                if len(chunk) > max_chunk_size * 1.5:
                    sentences = re.split(r'[.!?]+', chunk)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    current_sentences = ""
                    for sentence in sentences:
                        if len(current_sentences) + len(sentence) > max_chunk_size and current_sentences:
                            refined_chunks.append(current_sentences)
                            current_sentences = sentence
                        else:
                            current_sentences += " " + sentence if current_sentences else sentence
                    if current_sentences:
                        refined_chunks.append(current_sentences)
                else:
                    refined_chunks.append(chunk)
            chunks = refined_chunks
        
        return chunks
    
    def _summarize_with_pegasus(self, text):
        """Summarize text using Pegasus model"""
        try:
            if len(text) < 100:
                return text
                
            # Use local pipeline if available
            if "summarization" in self.local_pipelines:
                result = self.local_pipelines["summarization"](
                    text,
                    max_length=min(300, len(text) // 2),
                    min_length=max(50, len(text) // 10),
                    do_sample=False,
                    truncation=True
                )
                return result[0]['summary_text']
            else:
                # Use API
                result = self.query_hf_api(self.models["summarization"], text, {
                    "max_length": min(300, len(text) // 2),
                    "min_length": max(50, len(text) // 10),
                    "do_sample": False,
                    "truncation": True
                })
                
                if "error" not in result and isinstance(result, list) and len(result) > 0:
                    return result[0].get('summary_text', text[:200] + '...')
        except Exception as e:
            print(f"Pegasus summarization failed: {e}")
        
        # Fallback to simple extract
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 20]
        if meaningful_sentences:
            return '. '.join(meaningful_sentences[:3]) + '.'
        return text[:200] + '...'
    
    def _ensure_comprehensive_coverage(self, summary, key_sentences):
        """Ensure summary covers all key points"""
        if not key_sentences:
            return summary
            
        # Check coverage of key points
        summary_lower = summary.lower()
        missing_points = []
        
        for sentence in key_sentences:
            # Extract key phrases (simple approach)
            words = sentence.lower().split()
            key_phrases = [words[i:i+3] for i in range(len(words)-2)]
            
            # Check if any key phrase is in summary
            covered = False
            for phrase in key_phrases:
                phrase_text = ' '.join(phrase)
                if phrase_text in summary_lower:
                    covered = True
                    break
            
            if not covered:
                missing_points.append(sentence)
        
        # Add missing points to summary
        if missing_points:
            summary += "\n\nKey points:\n- " + "\n- ".join(missing_points[:3])
        
        return summary

    # Update existing methods to use advanced versions
    def document_qa(self, question, context):
        """Use advanced QA system"""
        return self.document_qa_advanced(question, context)
    
    def toxicity_check(self, text):
        """Use advanced toxicity check"""
        return self.toxicity_check_advanced(text)
    
    def summarize_large_text(self, text, max_chunk_size=800):
        """Use advanced summarization"""
        return self.summarize_large_text_advanced(text, max_chunk_size)
    
    def summarize_text(self, text):
        """Summarize shorter texts using advanced model"""
        if len(text) < 100:
            return {"summary": text[:150], "model": "text_too_short"}
        
        try:
            # Use Pegasus for short texts too
            result = self._summarize_with_pegasus(text)
            return {"summary": result, "model": self.models["summarization"]}
        except Exception as e:
            print(f"Pegasus summarization failed: {e}")
            return self.simple_summarize(text)

    def summarize_chunk(self, chunk_text):
        """Summarize individual chunk with better error handling"""
        if len(chunk_text) < 50:
            return chunk_text
            
        try:
            result = self._summarize_with_pegasus(chunk_text)
            if result and len(result) > 20:
                return result
        except Exception as e:
            print(f"Chunk summarization failed: {e}")
        
        # Fallback: use first few sentences
        sentences = re.split(r'[.!?]+', chunk_text)
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 20]
        if meaningful_sentences:
            return '. '.join(meaningful_sentences[:2]) + '.'
        return chunk_text[:150] + '...'

    def summarize_combined_chunks(self, combined_text, summary_length):
        """Summarize combined chunks using Pegasus"""
        try:
            result = self._summarize_with_pegasus(combined_text)
            return result
        except Exception as e:
            print(f"Combined chunks summarization failed: {e}")
            return combined_text[:summary_length] + '...'

    def simple_summarize(self, text):
        """Simple fallback summarization"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 3:
            summary = '. '.join(sentences[:3]) + '.'
        else:
            summary = text[:250] + "..." if len(text) > 250 else text
        return {"summary": summary, "model": "simple"}

    # ---------- Image classification ----------
    def classify_image(self, image_data, filename):
        try:
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            result = self.query_hf_api(self.models["image_classification"], image_b64)
            if "error" not in result and isinstance(result, list):
                top_predictions = result[:3]
                return {
                    "predictions": top_predictions, 
                    "primary_label": top_predictions[0]['label'], 
                    "confidence": top_predictions[0]['score'], 
                    "model": self.models["image_classification"]
                }
        except Exception as e:
            print(f"Image classification error: {e}")
        return {
            "predictions": [{"label": "image", "score": 1.0}], 
            "primary_label": "image", 
            "confidence": 1.0, 
            "model": "fallback"
        }

    # ---------- Semantic search ----------
    def initialize_semantic_model(self):
        if not self.sentence_model and not self.model_loaded:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model_loaded = True
                print("✅ Semantic search model loaded")
            except Exception as e:
                print(f"❌ Failed to load semantic model: {e}")
                self.model_loaded = False

    def get_text_embedding(self, text):
        self.initialize_semantic_model()
        if self.sentence_model:
            try:
                embedding = self.sentence_model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
            except Exception as e:
                print(f"Embedding generation error: {e}")
        return None

    def semantic_similarity(self, query, documents):
        self.initialize_semantic_model()
        if not self.sentence_model or not documents:
            return []
        try:
            query_embedding = self.sentence_model.encode(query, convert_to_tensor=True)
            doc_embeddings = self.sentence_model.encode(documents, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
            results = []
            for i, score in enumerate(similarities):
                results.append({
                    "document": documents[i], 
                    "similarity": round(score.item(), 4)
                })
            return sorted(results, key=lambda x: x["similarity"], reverse=True)
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            return []

    # ---------- Keywords ----------
    def extract_keywords_hf(self, text):
        """Use advanced keyphrase extraction model"""
        try:
            result = self.query_hf_api(self.models["keywords"], text[:1000])
            if "error" not in result:
                # Extract keyphrases
                if isinstance(result, list) and len(result) > 0:
                    keywords = []
                    for item in result:
                        if isinstance(item, dict) and 'word' in item:
                            # Only keep high-scoring keywords
                            if item.get('score', 0) > 0.3:
                                keywords.append(item['word'])
                    return keywords[:8]  # Return up to 8 keywords
        except Exception as e:
            print(f"HF keyword extraction error: {e}")
        return None

    def extract_keywords_tfidf(self, text, max_keywords=8):
        text2 = text.lower()
        text2 = re.sub(r'[^\w\s]', ' ', text2)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their'}
        words = re.findall(r'\b[a-z]{3,15}\b', text2)
        words = [w for w in words if w not in stop_words]
        word_freq = Counter(words)
        keywords = [word.capitalize() for word, freq in word_freq.most_common(max_keywords)]
        return {"keywords": keywords, "method": "tfidf"}

    def extract_keywords_simple(self, text):
        words = re.findall(r'\b[A-Za-z]{3,}\b', text)
        unique_words = list(set(words))[:6]
        return {"keywords": [w.capitalize() for w in unique_words], "method": "simple"}

    def extract_keywords_advanced(self, text):
        if len(text) < 50:
            return self.extract_keywords_simple(text)
        hf_keywords = self.extract_keywords_hf(text)
        if hf_keywords and len(hf_keywords) > 0:
            return {"keywords": hf_keywords[:8], "method": "zero-shot"}
        return self.extract_keywords_tfidf(text)

    # ---------- Sentiment ----------
    def analyze_sentiment(self, text):
        """Use advanced RoBERTa model for sentiment analysis"""
        if len(text) < 10:
            return {"sentiment": "neutral", "confidence": 1.0}
        try:
            result = self.query_hf_api(self.models["sentiment"], text[:512])
            if "error" not in result and isinstance(result, list) and len(result) > 0:
                sentiment_data = result[0]
                if isinstance(sentiment_data, list):
                    # Find the highest scoring sentiment
                    best = max(sentiment_data, key=lambda x: x['score'])
                    return {
                        "sentiment": best['label'], 
                        "confidence": best['score'], 
                        "scores": sentiment_data
                    }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
        return {"sentiment": "neutral", "confidence": 1.0}

    # ---------- Improved NER ----------
    def named_entity_recognition(self, text):
        """Use advanced BERT model for named entity recognition"""
        if not text or len(text.strip()) < 10:
            return {"error": "Text too short for NER", "entities": []}
        if not self.is_available():
            return {"error": "Hugging Face token not configured", "entities": []}
        try:
            result = self.query_hf_api(self.models["ner"], text[:1000])
            if isinstance(result, list):
                # Format entities for better display
                entities = []
                for entity in result:
                    if isinstance(entity, dict) and 'entity_group' in entity:
                        entities.append({
                            "text": entity.get('word', ''),
                            "type": entity.get('entity_group', ''),
                            "score": round(entity.get('score', 0), 3)
                        })
                return {"entities": entities, "model": self.models["ner"]}
            else:
                return {"error": "Unexpected NER response", "raw": result}
        except Exception as e:
            print(f"NER error: {e}")
            return {"error": str(e), "entities": []}

    # ---------- OCR/Layout ----------
    def ocr_layout(self, file_content, filename, file_extension):
        # Declare global variables at the very beginning of the method
        global PYTESSERACT_AVAILABLE, PDF2IMAGE_AVAILABLE
        
        try:
            ext = file_extension.lower()
            if PYTESSERACT_AVAILABLE:
                try:
                    if ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']:
                        img = Image.open(BytesIO(file_content))
                        text = pytesseract.image_to_string(img)
                        return {"ocr_text": text[:1000] + "..." if len(text) > 1000 else text, "method": "pytesseract"}
                    elif ext == 'pdf' and PDF2IMAGE_AVAILABLE:
                        try:
                            images = convert_from_bytes(file_content)
                            full_text = ""
                            for img in images:
                                full_text += pytesseract.image_to_string(img) + "\n"
                            return {"ocr_text": full_text[:1000] + "..." if len(full_text) > 1000 else full_text, "method": "pytesseract-pdf"}
                        except Exception as e:
                            print(f"pdf2image OCR error: {e}")
                            # fallback to text extraction
                            text = self.extract_text_from_file(file_content, filename, 'pdf')
                            return {"ocr_text": text[:1000] + "..." if len(text) > 1000 else text, "method": "pypdf2-fallback", "note": str(e)}
                    else:
                        return {"error": "Unsupported file type for OCR", "method": "pytesseract"}
                except Exception as e:
                    print(f"Pytesseract runtime error: {e}")
                    # Check if it's a PATH issue
                    if "tesseract" in str(e).lower() or "is not installed" in str(e).lower():
                        PYTESSERACT_AVAILABLE = False
                        print("⚠️ Tesseract not found in PATH, disabling OCR")
            # Fallback if tesseract is not available
            if self.is_available():
                if ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']:
                    b64 = base64.b64encode(file_content).decode('utf-8')
                    result = self.query_hf_api("microsoft/layoutlmv3-base", b64)
                    return {"layout_response": result, "method": "hf-layoutlmv3"}
                elif ext == 'pdf':
                    text = self.extract_text_from_file(file_content, filename, ext)
                    return {"ocr_text": text[:1000] + "..." if len(text) > 1000 else text, "method": "pypdf2-fallback", "note": "Install pytesseract/pdf2image for better OCR"}
                else:
                    return {"error": "No OCR available for this file type", "note": "Install pytesseract or provide HF token"}
            else:
                text = self.extract_text_from_file(file_content, filename, ext)
                return {"ocr_text": text[:1000] + "..." if len(text) > 1000 else text, "method": "pypdf2-fallback", "note": "No local OCR and HF token not configured"}
        except Exception as e:
            print(f"OCR/Layout error: {e}")
            return {"error": str(e)}

    # ---------- Image Captioning ----------
    def image_captioning(self, image_data):
        if not self.is_available():
            return {"error": "Hugging Face token not configured"}
        try:
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            result = self.query_hf_api(self.models["image_caption"], image_b64)
            # Extract caption from response
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get('generated_text', 'No caption generated')
                return {"caption": caption}
            return {"caption": "No caption generated"}
        except Exception as e:
            print(f"Image caption error: {e}")
            return {"error": str(e)}

    # ---------- High-level process_file ----------
    def process_file(self, file_content, filename, file_extension):
        result = {
            "text_extracted": False,
            "extracted_text": "",
            "summary": "",
            "keywords": [],
            "sentiment": "neutral",
            "sentiment_confidence": 0.0,
            "char_count": 0,
            "ai_processed": False,
            "file_category": "unknown",
            "ai_models_used": [],
            "embedding": None,
            "error": None,
            "image_analysis": {},
            "document_stats": {}
        }
        try:
            file_category = self.get_file_category(file_extension)
            result["file_category"] = file_category

            if file_category == "text":
                extracted_text = self.extract_text_from_file(file_content, filename, file_extension)
                result["extracted_text"] = extracted_text
                result["char_count"] = len(extracted_text)
                result["text_extracted"] = True

                embedding = self.get_text_embedding(extracted_text[:1000])
                if embedding:
                    result["embedding"] = embedding

                ai_result = self.process_text_content(extracted_text)
                result.update(ai_result)

            elif file_category == "image":
                image_result = self.process_image_content(file_content, filename)
                result.update(image_result)
                result["char_count"] = len(filename)
            else:
                result["summary"] = f"File processed: {filename}"
                result["keywords"] = [file_extension, "file", "document"]
                result["ai_models_used"] = ["file_processing"]
                result["char_count"] = len(filename)

            result["ai_processed"] = True
            print(f"✅ Enhanced AI processing completed for {filename}")

        except Exception as e:
            print(f"AI processing error for {filename}: {e}")
            result["error"] = str(e)
            result["summary"] = "AI processing encountered an error"
            result["keywords"] = ["processing-error"]

        return result

    def process_text_content(self, text):
        result = {
            "summary": "",
            "keywords": [],
            "sentiment": "neutral",
            "sentiment_confidence": 0.0,
            "ai_models_used": [],
            "document_stats": {}
        }

        result["document_stats"] = {
            "total_characters": len(text),
            "total_words": len(text.split()),
            "estimated_pages": max(1, len(text) // 1500)
        }

        if len(text) > 2000:
            summary_result = self.summarize_large_text(text)
            result["summary"] = summary_result["summary"]
            result["ai_models_used"].append(summary_result.get("model", "chunked"))
            if "chunks_processed" in summary_result:
                result["document_stats"]["chunks_processed"] = summary_result["chunks_processed"]
            if "key_points_covered" in summary_result:
                result["document_stats"]["key_points_covered"] = summary_result["key_points_covered"]
        else:
            summary_result = self.summarize_text(text)
            result["summary"] = summary_result["summary"]
            result["ai_models_used"].append(summary_result.get("model", "pegasus-xsum"))

        keyword_result = self.extract_keywords_advanced(text)
        if isinstance(keyword_result, dict):
            result["keywords"] = keyword_result.get("keywords", [])
            result["ai_models_used"].append(keyword_result.get("method", "keyword_extraction"))
        else:
            result["keywords"] = keyword_result
            result["ai_models_used"].append("keyword_extraction")

        sentiment_result = self.analyze_sentiment(text)
        result["sentiment"] = sentiment_result.get("sentiment", "neutral")
        result["sentiment_confidence"] = sentiment_result.get("confidence", 0.0)
        result["ai_models_used"].append("sentiment-analysis")

        return result

    def process_image_content(self, image_data, filename):
        result = {
            "summary": "",
            "keywords": [],
            "image_analysis": {},
            "ai_models_used": []
        }
        classification_result = self.classify_image(image_data, filename)
        result["image_analysis"] = classification_result
        primary_label = classification_result.get("primary_label", "image")
        confidence = classification_result.get("confidence", 1.0)
        result["summary"] = f"Image classified as: {primary_label} (confidence: {confidence:.2f})"
        result["keywords"] = [primary_label, "image", "visual", "classification"]
        result["ai_models_used"].append(classification_result.get("model", "image-classifier"))
        return result

    def get_file_category(self, file_extension):
        text_types = ['pdf', 'txt', 'doc', 'docx', 'rtf', 'md', 'csv']
        image_types = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff']
        if file_extension in text_types:
            return "text"
        elif file_extension in image_types:
            return "image"
        else:
            return "other"

# Initialize AI services
ai_services = EnhancedAIServices()

# ---------------- MongoDB Connection ----------------
try:
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    db = client["nasuni"]
    files_col = db["files"]
    users_col = db["users"]
    client.admin.command('ping')
    print("✅ MongoDB connected successfully")

    # create default admin if none
    if users_col.count_documents({}) == 0:
        admin_password = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt())
        users_col.insert_one({
            "email": "admin@nasuni.com",
            "password": admin_password.decode('utf-8'),
            "name": "System Administrator",
            "role": "admin",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        print("✅ Default admin user created: admin@nasuni.com / admin123")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")

# ---------------- AWS S3 Configuration ----------------
try:
    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("AWS_SECRET_KEY")
    bucket_name = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
    if aws_access_key and aws_secret_key and bucket_name:
        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            try:
                s3.head_bucket(Bucket=bucket_name)
                print(f"✅ AWS S3 bucket '{bucket_name}' accessible (Region: {aws_region})")
            except ClientError as e:
                err_code = e.response.get("Error", {}).get("Code", "")
                print(f"⚠️ S3 head_bucket check failed for '{bucket_name}': {err_code} - {str(e)}")
        except NoCredentialsError:
            s3 = None
            print("⚠️ AWS credentials invalid - using local file storage")
        except Exception as e:
            s3 = None
            print(f"⚠️ AWS S3 init failed: {e} - falling back to local storage")
            os.makedirs("local_storage", exist_ok=True)
    else:
        s3 = None
        os.makedirs("local_storage", exist_ok=True)
        print("⚠️ AWS credentials or bucket not configured - using local file storage")
except Exception as e:
    print(f"⚠️ AWS S3 configuration failed: {e}")
    s3 = None
    os.makedirs("local_storage", exist_ok=True)

# ---------------- Authentication Routes ----------------
@app.route("/register", methods=["POST", "OPTIONS"])
def register():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        role = data.get('role', 'user')

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        if users_col.find_one({"email": email}):
            return jsonify({"error": "User already exists"}), 400

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = {
            "email": email,
            "password": hashed_password.decode('utf-8'),
            "name": name,
            "role": role,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        users_col.insert_one(user)
        access_token = create_access_token(identity=email)
        return jsonify({
            "message": "User created successfully",
            "access_token": access_token,
            "user": {"email": email, "name": name, "role": role}
        }), 201
    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        user = users_col.find_one({"email": email})
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401
        if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({"error": "Invalid credentials"}), 401
        access_token = create_access_token(identity=email)
        return jsonify({
            "message": "Login successful",
            "access_token": access_token,
            "user": {"email": user['email'], "name": user['name'], "role": user['role']}
        })
    except Exception as e:
        return jsonify({"error": f"Login failed: {str(e)}"}), 500

@app.route("/profile", methods=["GET", "OPTIONS"])
@jwt_required()
def profile():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        current_user = get_jwt_identity()
        user = users_col.find_one({"email": current_user}, {"password": 0})
        if not user:
            return jsonify({"error": "User not found"}), 404
        return jsonify({
            "user": {
                "email": user['email'],
                "name": user['name'],
                "role": user['role'],
                "created_at": user.get('created_at', '')
            }
        })
    except Exception as e:
        return jsonify({"error": f"Profile fetch failed: {str(e)}"}), 500

# ---------------- File Upload & Processing ----------------
@app.route("/upload", methods=["POST", "OPTIONS"])
@jwt_required()
def upload_file():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file:
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'bin'
            file_content = file.read()
            current_user = get_jwt_identity()

            # Process file with AI
            ai_result = ai_services.process_file(file_content, filename, file_extension)
            if ai_result.get("error"):
                print(f"AI processing error: {ai_result['error']}")

            # Store file in S3 or locally
            file_id = None
            if s3 and bucket_name:
                try:
                    file_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                    s3.put_object(Bucket=bucket_name, Key=file_id, Body=file_content, ContentType=file.content_type)
                    file_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{file_id}"
                except Exception as e:
                    print(f"S3 upload failed: {e}")
                    file_id = None
            if not file_id:
                file_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                local_path = os.path.join("local_storage", file_id)
                with open(local_path, 'wb') as f:
                    f.write(file_content)
                file_url = f"/local_storage/{file_id}"

            # Save metadata to MongoDB
            file_doc = {
                "file_id": file_id,
                "filename": filename,
                "file_extension": file_extension,
                "file_url": file_url,
                "uploaded_by": current_user,
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ai_processed": ai_result.get("ai_processed", False),
                "text_extracted": ai_result.get("text_extracted", False),
                "char_count": ai_result.get("char_count", 0),
                "file_category": ai_result.get("file_category", "unknown"),
                "summary": ai_result.get("summary", ""),
                "keywords": ai_result.get("keywords", []),
                "sentiment": ai_result.get("sentiment", "neutral"),
                "sentiment_confidence": ai_result.get("sentiment_confidence", 0.0),
                "ai_models_used": ai_result.get("ai_models_used", []),
                "embedding": ai_result.get("embedding"),
                "image_analysis": ai_result.get("image_analysis", {}),
                "document_stats": ai_result.get("document_stats", {})
            }
            files_col.insert_one(file_doc)

            # Clean response - remove embedding for security
            response_doc = file_doc.copy()
            response_doc.pop('embedding', None)
            response_doc.pop('_id', None)

            return jsonify({
                "message": "File uploaded and processed successfully",
                "file": response_doc
            }), 201
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/files", methods=["GET", "OPTIONS"])
@jwt_required()
def list_files():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        current_user = get_jwt_identity()
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        skip = (page - 1) * per_page
        files = list(files_col.find({"uploaded_by": current_user}, {
            "file_id": 1, "filename": 1, "file_extension": 1, "uploaded_at": 1,
            "char_count": 1, "summary": 1, "keywords": 1, "sentiment": 1,
            "file_category": 1, "ai_processed": 1, "image_analysis": 1
        }).sort("uploaded_at", -1).skip(skip).limit(per_page))
        for f in files:
            f['_id'] = str(f['_id'])
        total = files_col.count_documents({"uploaded_by": current_user})
        return jsonify({
            "files": files,
            "pagination": {"page": page, "per_page": per_page, "total": total}
        })
    except Exception as e:
        return jsonify({"error": f"File list failed: {str(e)}"}), 500

@app.route("/files/<file_id>", methods=["GET", "OPTIONS"])
@jwt_required()
def get_file(file_id):
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        current_user = get_jwt_identity()
        file_doc = files_col.find_one({"file_id": file_id, "uploaded_by": current_user})
        if not file_doc:
            return jsonify({"error": "File not found"}), 404
        file_doc['_id'] = str(file_doc['_id'])
        # Remove embedding from response
        file_doc.pop('embedding', None)
        return jsonify({"file": file_doc})
    except Exception as e:
        return jsonify({"error": f"File fetch failed: {str(e)}"}), 500

@app.route("/files/<file_id>", methods=["DELETE", "OPTIONS"])
@jwt_required()
def delete_file(file_id):
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        current_user = get_jwt_identity()
        file_doc = files_col.find_one({"file_id": file_id, "uploaded_by": current_user})
        if not file_doc:
            return jsonify({"error": "File not found"}), 404
        # Delete from storage
        if s3 and bucket_name:
            try:
                s3.delete_object(Bucket=bucket_name, Key=file_id)
            except Exception as e:
                print(f"S3 delete failed: {e}")
        else:
            local_path = os.path.join("local_storage", file_id)
            if os.path.exists(local_path):
                os.remove(local_path)
        # Delete from MongoDB
        files_col.delete_one({"file_id": file_id})
        return jsonify({"message": "File deleted successfully"})
    except Exception as e:
        return jsonify({"error": f"File deletion failed: {str(e)}"}), 500

@app.route("/files/<file_id>/download", methods=["GET", "OPTIONS"])
@jwt_required()
def download_file(file_id):
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        current_user = get_jwt_identity()
        file_doc = files_col.find_one({"file_id": file_id, "uploaded_by": current_user})
        if not file_doc:
            return jsonify({"error": "File not found"}), 404

        # Download from S3 if configured
        if s3 and bucket_name:
            try:
                s3_obj = s3.get_object(Bucket=bucket_name, Key=file_id)
                # Use send_file to stream the file to the frontend for download
                return send_file(
                    BytesIO(s3_obj['Body'].read()),
                    download_name=file_doc['filename'],
                    as_attachment=True
                )
            except Exception as e:
                print(f"S3 download failed: {e}")
                return jsonify({"error": "S3 download failed."}), 500

        # Or download from local storage
        else:
            local_path = os.path.join("local_storage", file_id)
            if os.path.exists(local_path):
                return send_file(
                    local_path,
                    download_name=file_doc['filename'],
                    as_attachment=True
                )
            else:
                return jsonify({"error": "File not found"}), 404

    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

# ---------------- File Analysis Endpoint ----------------
@app.route("/api/analyze/file/<file_id>/<analysis_type>", methods=["GET", "OPTIONS"])
@jwt_required()
def analyze_file(file_id, analysis_type):
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    
    try:
        current_user = get_jwt_identity()
        file_doc = files_col.find_one({"file_id": file_id, "uploaded_by": current_user})
        
        if not file_doc:
            return jsonify({"error": "File not found"}), 404
            
        # Get file content
        file_content = None
        if s3 and bucket_name:
            try:
                s3_obj = s3.get_object(Bucket=bucket_name, Key=file_id)
                file_content = s3_obj['Body'].read()
            except Exception as e:
                print(f"S3 download failed: {e}")
        
        if not file_content:
            local_path = os.path.join("local_storage", file_id)
            if os.path.exists(local_path):
                with open(local_path, 'rb') as f:
                    file_content = f.read()
        
        if not file_content:
            return jsonify({"error": "Could not retrieve file content"}), 500
        
        # Extract text
        extracted_text = ai_services.extract_text_from_file(
            file_content, 
            file_doc['filename'], 
            file_doc['file_extension']
        )
        
        # Perform analysis based on type
        if analysis_type == "summary":
            result = ai_services.summarize_large_text(extracted_text)
        elif analysis_type == "keywords":
            result = ai_services.extract_keywords_advanced(extracted_text)
        elif analysis_type == "sentiment":
            result = ai_services.analyze_sentiment(extracted_text)
        elif analysis_type == "ner":
            result = ai_services.named_entity_recognition(extracted_text)
        elif analysis_type == "qa":
            # For QA, need additional question parameter
            question = request.args.get('question', '')
            if not question:
                return jsonify({"error": "Question parameter required for QA analysis"}), 400
            result = ai_services.document_qa(question, extracted_text)
        elif analysis_type == "semantic":
            # For semantic search, need query parameter
            query = request.args.get('query', '')
            if not query:
                return jsonify({"error": "Query parameter required for semantic analysis"}), 400
            # Get other documents for comparison
            other_files = list(files_col.find(
                {"uploaded_by": current_user, "file_id": {"$ne": file_id}}, 
                {"summary": 1}
            ).limit(10))
            documents = [f.get('summary', '') for f in other_files if f.get('summary')]
            if documents:
                result = ai_services.semantic_similarity(query, documents)
            else:
                result = {"error": "No other documents found for comparison"}
        elif analysis_type == "toxicity":
            result = ai_services.toxicity_check(extracted_text)
        else:
            return jsonify({"error": "Invalid analysis type"}), 400
            
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# ---------------- AI Analysis Endpoints ----------------
@app.route("/api/analyze/summary", methods=["POST", "OPTIONS"])
@jwt_required()
def analyze_summary():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "Text required"}), 400
        result = ai_services.summarize_large_text(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Summary analysis failed: {str(e)}"}), 500

@app.route("/api/analyze/keywords", methods=["POST", "OPTIONS"])
@jwt_required()
def analyze_keywords():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "Text required"}), 400
        result = ai_services.extract_keywords_advanced(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Keyword extraction failed: {str(e)}"}), 500

@app.route("/api/analyze/sentiment", methods=["POST", "OPTIONS"])
@jwt_required()
def analyze_sentiment():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "Text required"}), 400
        result = ai_services.analyze_sentiment(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Sentiment analysis failed: {str(e)}"}), 500

@app.route("/api/analyze/ner", methods=["POST", "OPTIONS"])
@jwt_required()
def analyze_ner():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "Text required"}), 400
        result = ai_services.named_entity_recognition(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"NER failed: {str(e)}"}), 500

@app.route("/api/analyze/qa", methods=["POST", "OPTIONS"])
@jwt_required()
def analyze_qa():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        data = request.get_json()
        question = data.get('question', '')
        context = data.get('context', '')
        if not question or not context:
            return jsonify({"error": "Question and context required"}), 400
        
        result = ai_services.document_qa(question, context)
        
        # Clean response - extract only the answer
        if "qa_result" in result:
            clean_result = {
                "answer": result["qa_result"].get("answer", ""),
                "score": result["qa_result"].get("score", 0),
                "context_used": result["qa_result"].get("context_used", 0)
            }
            return jsonify({"qa_result": clean_result})
        else:
            return jsonify(result)
            
    except Exception as e:
        return jsonify({"error": f"QA failed: {str(e)}"}), 500

@app.route("/api/analyze/semantic", methods=["POST", "OPTIONS"])
@jwt_required()
def analyze_semantic():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        data = request.get_json()
        query = data.get('query', '')
        documents = data.get('documents', [])
        if not query or not documents:
            return jsonify({"error": "Query and documents required"}), 400
        
        result = ai_services.semantic_similarity(query, documents)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Semantic analysis failed: {str(e)}"}), 500

@app.route("/api/analyze/toxicity", methods=["POST", "OPTIONS"])
@jwt_required()
def analyze_toxicity():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "Text required"}), 400
        result = ai_services.toxicity_check(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Toxicity analysis failed: {str(e)}"}), 500

@app.route("/api/analyze/ocr", methods=["POST", "OPTIONS"])
@jwt_required()
def analyze_ocr():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        file_content = file.read()
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'bin'
        
        result = ai_services.ocr_layout(file_content, filename, file_extension)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"OCR analysis failed: {str(e)}"}), 500

# ---------------- Health Check ----------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "mongodb": "connected" if files_col else "disconnected",
            "s3": "connected" if s3 else "disconnected",
            "ai": "available" if ai_services.is_available() else "unavailable",
            "ocr": "available" if PYTESSERACT_AVAILABLE else "unavailable"
        }
    })

# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)