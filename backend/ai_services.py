from transformers import pipeline
from sentence_transformers import SentenceTransformer
import PyPDF2
import textract
import re
import os
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIServices:
    def __init__(self):
        logger.info("Initializing AI Services...")
        
        try:
            # Initialize summarization pipeline with a smaller model for faster loading
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn",
                min_length=30,
                max_length=150
            )
            logger.info("✓ Summarization model loaded")
        except Exception as e:
            logger.error(f"Summarization model failed: {e}")
            self.summarizer = None
        
        try:
            # Initialize embedding model for future semantic search
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Sentence transformer loaded")
        except Exception as e:
            logger.error(f"Sentence transformer failed: {e}")
            self.embedder = None
        
        try:
            # Keyword extraction using a simpler approach (NER was too heavy)
            self.keyword_pipeline = pipeline(
                "text2text-generation",
                model="mrm8488/t5-base-finetuned-common-gen",
                max_length=20
            )
            logger.info("✓ Keyword extraction model loaded")
        except Exception as e:
            logger.error(f"Keyword model failed: {e}")
            self.keyword_pipeline = None
    
    def extract_text_from_file(self, file_content: bytes, file_extension: str, filename: str) -> str:
        """Extract text from various file types"""
        try:
            temp_path = None
            try:
                # Create temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name
                
                if file_extension == 'pdf':
                    with open(temp_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        return text.strip()
                
                elif file_extension in ['txt']:
                    with open(temp_path, 'r', encoding='utf-8', errors='ignore') as file:
                        return file.read()
                
                elif file_extension in ['doc', 'docx']:
                    try:
                        text = textract.process(temp_path).decode('utf-8', errors='ignore')
                        return text
                    except:
                        return f"Document file: {filename}"
                
                else:
                    return f"File: {filename} (type: {file_extension})"
                    
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Text extraction error for {filename}: {e}")
            return f"Error extracting text: {str(e)}"
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize text using BART model"""
        if len(text.strip()) < 100:
            return "Text too short for meaningful summarization."
        
        if not self.summarizer:
            return "Summarization service temporarily unavailable."
        
        try:
            # Clean and prepare text
            text = text.strip()
            
            # Truncate very long texts to avoid token limits
            if len(text) > 1024:
                text = text[:1000] + "... [text truncated]"
            
            summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return "Summarization failed due to technical issues."
    
    def extract_keywords_simple(self, text: str, max_keywords: int = 8) -> List[str]:
        """Extract keywords using simple frequency analysis and patterns"""
        try:
            if len(text.strip()) < 50:
                return ["short-text"]
            
            # Convert to lowercase and remove special characters
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must'}
            
            # Extract words
            words = re.findall(r'\b[a-z]{3,15}\b', text)
            words = [word for word in words if word not in stop_words]
            
            # Get frequency
            from collections import Counter
            word_freq = Counter(words)
            
            # Get most common words
            common_words = word_freq.most_common(max_keywords)
            
            # Capitalize first letter of each keyword
            keywords = [word.capitalize() for word, count in common_words]
            
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return ["keywords-unavailable"]
    
    def extract_keywords_advanced(self, text: str) -> List[str]:
        """Advanced keyword extraction using ML model"""
        if not self.keyword_pipeline or len(text.strip()) < 50:
            return self.extract_keywords_simple(text)
        
        try:
            # Use model to generate keywords
            prompt = f"extract keywords from: {text[:500]}"
            result = self.keyword_pipeline(prompt, max_length=30, num_return_sequences=1)
            
            if result and len(result) > 0:
                keywords_text = result[0]['generated_text']
                keywords = [k.strip().capitalize() for k in keywords_text.split(',')]
                return keywords[:5]
            else:
                return self.extract_keywords_simple(text)
                
        except Exception as e:
            logger.error(f"Advanced keyword extraction failed: {e}")
            return self.extract_keywords_simple(text)
    
    def process_file(self, file_content: bytes, filename: str, file_extension: str) -> Dict:
        """Main method to process file and extract AI insights"""
        result = {
            "text_extracted": False,
            "summary": "",
            "keywords": [],
            "char_count": 0,
            "ai_processed": False,
            "error": None
        }
        
        try:
            # Extract text
            extracted_text = self.extract_text_from_file(file_content, file_extension, filename)
            result["char_count"] = len(extracted_text)
            
            if len(extracted_text.strip()) > 50:
                result["text_extracted"] = True
                result["ai_processed"] = True
                
                # Generate summary
                result["summary"] = self.summarize_text(extracted_text)
                
                # Extract keywords
                result["keywords"] = self.extract_keywords_advanced(extracted_text)
                
                logger.info(f"✓ AI processing completed for {filename}")
            else:
                result["summary"] = "Insufficient text content for AI analysis."
                result["keywords"] = ["minimal-text"]
                
        except Exception as e:
            logger.error(f"AI processing error for {filename}: {e}")
            result["error"] = str(e)
            result["summary"] = "AI processing encountered an error."
            result["keywords"] = ["processing-error"]
        
        return result

# Global AI service instance
ai_services = AIServices()