import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import re
import sys
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

class HallucinationDetector:
    def __init__(self, model_name: str = "llama3.2:1b"):
        """
        Initialize the hallucination detector with specified LLM model.
        Using llama3.2:1b (1.3GB) - the smallest available model
        """
        self.model_name = model_name
        self.question_counter = 1
        
        # Initialize sentence transformer
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading sentence transformer: {e}")
            sys.exit(1)
        
        # Try to load spaCy model for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("⚠️  spaCy English model not found. Using simple sentence splitting.")
            self.nlp = None
    
    def check_ollama_connection(self) -> bool:
        """
        Check if Ollama is running using subprocess.
        """
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            
            if self.model_name in result.stdout:
                print(f"✅ Ollama is running and '{self.model_name}' is available")
                print(f"📊 Using the smallest model (1.3GB) for faster processing")
                return True
            else:
                print(f"❌ Model '{self.model_name}' not found in available models.")
                print("Available models:")
                print(result.stdout)
                
                # Suggest using llama3.2:3b if 1b is not available
                if "llama3.2:3b" in result.stdout:
                    print("\n⚠️  Falling back to llama3.2:3b (2.0GB)")
                    self.model_name = "llama3.2:3b"
                    return True
                elif "llama3:latest" in result.stdout:
                    print("\n⚠️  Falling back to llama3:latest (4.7GB) - this is larger")
                    self.model_name = "llama3:latest"
                    return True
                else:
                    return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running ollama list: {e}")
            print("Please ensure Ollama is installed and running.")
            return False
        except FileNotFoundError:
            print("❌ Ollama command not found. Please install Ollama from https://ollama.ai/")
            return False
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using Ollama via subprocess.
        Using simpler prompt for faster generation with small model.
        """
        try:
            # Simplify the prompt for the small model
            simple_prompt = f"Please answer concisely: {prompt}"
            
            # Use ollama run command which is more reliable
            result = subprocess.run(
                ['ollama', 'run', self.model_name, simple_prompt],
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout for small model
            )
            
            if result.returncode == 0 and result.stdout:
                response = result.stdout.strip()
                # Clean up response - remove any command prompts or extra spaces
                response = re.sub(r'^>>>\s*', '', response)  # Remove >>> prefix if present
                return response
            else:
                print(f"⚠️  Generation failed: {result.stderr}")
                return ""
                
        except subprocess.TimeoutExpired:
            print("❌ Generation timed out")
            return ""
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return ""
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy or simple regex fallback.
        """
        if self.nlp and len(text.strip()) > 0:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        else:
            # Simple regex-based sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def calculate_sentence_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Calculate semantic similarity between two sentences.
        """
        if not sentence1 or not sentence2:
            return 0.0
            
        try:
            embeddings = self.sentence_model.encode([sentence1, sentence2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"⚠️  Error calculating similarity: {e}")
            return 0.0
    
    def extract_key_entities(self, sentence: str) -> Set[str]:
        """
        Extract key entities and concepts from a sentence.
        """
        entities = set()
        
        try:
            # Extract capitalized phrases (potential proper nouns)
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
            entities.update(proper_nouns)
            
            # Extract numbers and dates
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', sentence)
            entities.update(numbers)
            
            # Extract quoted phrases
            quoted = re.findall(r'\"([^\"]*)\"', sentence)
            entities.update(quoted)
            
        except Exception as e:
            print(f"⚠️  Error extracting entities: {e}")
        
        return entities
    
    def detect_hallucinations(self, prompt: str, num_generations: int = 3, 
                            similarity_threshold: float = 0.3) -> Dict:
        """
        Detect hallucinations by comparing multiple generations.
        Using fewer generations for faster processing with small model.
        """
        print(f"\n📝 Generating {num_generations} responses using {self.model_name}...")
        print(f"Question: '{prompt}'")
        print("⏳ This may take a moment with the small model...")
        
        # Generate multiple responses
        responses = []
        for i in range(num_generations):
            print(f"  Generating response {i+1}/{num_generations}...")
            response = self.generate_response(prompt)
            if response and len(response.strip()) > 0:
                responses.append(response)
                print(f"    ✅ Response {i+1} generated ({len(response)} chars)")
            else:
                print(f"    ⚠️  Empty response {i+1}")
        
        if not responses:
            print("❌ No valid responses generated.")
            return {}
        
        # Split each response into sentences
        all_sentences = []
        response_sentences = []
        
        for i, response in enumerate(responses):
            sentences = self.split_into_sentences(response)
            response_sentences.append(sentences)
            for sentence in sentences:
                all_sentences.append({
                    'sentence': sentence,
                    'response_id': i,
                    'response_text': response
                })
        
        print(f"  Extracted {len(all_sentences)} sentences total")
        
        if len(all_sentences) < 2:
            print("⚠️  Not enough sentences to compare")
            return {}
        
        # Create similarity matrix and detect hallucinations
        hallucination_results = []
        
        for i, sent_dict in enumerate(all_sentences):
            current_sentence = sent_dict['sentence']
            current_response_id = sent_dict['response_id']
            
            # Compare with all other sentences from different responses
            similarities = []
            for j, other_sent_dict in enumerate(all_sentences):
                if i != j and other_sent_dict['response_id'] != current_response_id:
                    similarity = self.calculate_sentence_similarity(
                        current_sentence, other_sent_dict['sentence']
                    )
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                max_similarity = np.max(similarities)
                
                # Determine if hallucinated based on similarity thresholds
                is_hallucinated = avg_similarity < similarity_threshold and max_similarity < 0.6
                
                hallucination_results.append({
                    'sentence': current_sentence,
                    'response_id': current_response_id,
                    'avg_similarity': round(avg_similarity, 3),
                    'max_similarity': round(max_similarity, 3),
                    'is_hallucinated': is_hallucinated,
                    'key_entities': list(self.extract_key_entities(current_sentence)),
                    'response_text_preview': sent_dict['response_text'][:100] + "..." if len(sent_dict['response_text']) > 100 else sent_dict['response_text']
                })
        
        # Create hallucination table
        hallucination_table = pd.DataFrame(hallucination_results)
        
        # Summary statistics
        total_sentences = len(hallucination_table)
        hallucinated_count = hallucination_table['is_hallucinated'].sum() if not hallucination_table.empty else 0
        hallucination_rate = hallucinated_count / total_sentences if total_sentences > 0 else 0
        
        summary = {
            'prompt': prompt,
            'num_responses': len(responses),
            'total_sentences': total_sentences,
            'hallucinated_sentences': hallucinated_count,
            'hallucination_rate': round(hallucination_rate, 3),
            'avg_similarity_across_responses': round(hallucination_table['avg_similarity'].mean(), 3) if not hallucination_table.empty else 0,
            'responses': responses
        }
        
        return {
            'summary': summary,
            'hallucination_table': hallucination_table,
            'detailed_results': hallucination_results
        }

def display_hallucination_table(hallucination_table):
    """
    Display a clean table showing all hallucinated sentences.
    """
    if hallucination_table.empty:
        print("✅ No hallucinated sentences found.")
        return
    
    # Filter only hallucinated sentences
    hallucinated_sentences = hallucination_table[hallucination_table['is_hallucinated']]
    
    if hallucinated_sentences.empty:
        print("✅ No hallucinated sentences found.")
        return
    
    print(f"\n📊 HALLUCINATION DETECTION RESULTS TABLE")
    print("=" * 120)
    
    # Create a clean display table
    display_data = []
    for idx, row in hallucinated_sentences.iterrows():
        display_data.append({
            'Response #': row['response_id'] + 1,
            'Sentence': row['sentence'],
            'Avg Similarity': row['avg_similarity'],
            'Max Similarity': row['max_similarity'],
            'Key Entities': ', '.join(row['key_entities']) if row['key_entities'] else 'None',
            'Confidence': 'HIGH' if row['avg_similarity'] < 0.2 else 'MEDIUM' if row['avg_similarity'] < 0.3 else 'LOW'
        })
    
    # Convert to DataFrame for nice formatting
    display_df = pd.DataFrame(display_data)
    
    # Display the table
    print(display_df.to_string(index=False, max_colwidth=80))
    print("=" * 120)
    print("📈 CONFIDENCE LEVELS:")
    print("   HIGH: Avg Similarity < 0.2 (Very likely hallucinated)")
    print("   MEDIUM: Avg Similarity 0.2-0.3 (Likely hallucinated)") 
    print("   LOW: Avg Similarity 0.3-0.6 (Possibly hallucinated)")

def save_all_responses_and_hallucinations(results, user_input, question_number):
    """
    Save all 3 responses with question number, and hallucinations at the bottom if any.
    """
    import datetime
    
    # Create timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hallucination_analysis_{timestamp}.csv"
    
    # Prepare data for all responses
    all_data = []
    
    # 1. Add all responses first
    summary = results['summary']
    for i, response in enumerate(summary['responses']):
        all_data.append({
            'Type': 'RESPONSE',
            'Question_Number': question_number,
            'Question': user_input,
            'Response_Number': i + 1,
            'Content': response,
            'Consistency_Score': '',
            'Confidence_Level': '',
            'Notes': 'Original generated response'
        })
    
    # 2. Add hallucinations at the bottom if any exist
    hallucination_table = results['hallucination_table']
    hallucinated_sentences = hallucination_table[hallucination_table['is_hallucinated']]
    
    if not hallucinated_sentences.empty:
        for idx, row in hallucinated_sentences.iterrows():
            all_data.append({
                'Type': 'HALLUCINATION',
                'Question_Number': question_number,
                'Question': user_input,
                'Response_Number': row['response_id'] + 1,
                'Content': row['sentence'],
                'Consistency_Score': row['avg_similarity'],
                'Confidence_Level': 'HIGH' if row['avg_similarity'] < 0.2 else 'MEDIUM' if row['avg_similarity'] < 0.3 else 'LOW',
                'Notes': f"Potential hallucination detected (score: {row['avg_similarity']})"
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    return filename

def interactive_mode():
    """
    Interactive mode where users can input their own questions.
    """
    print("🎯 INTERACTIVE HALLUCINATION DETECTION")
    print("=" * 50)
    print("Using llama3.2:1b (1.3GB) - Small & Fast Model")
    print("Ask any question and I'll analyze it for potential hallucinations!")
    print("Type 'quit' or 'exit' to stop the program.")
    print("Type 'examples' to see some example questions.")
    print("=" * 50)
    
    # Initialize detector with smallest model
    detector = HallucinationDetector(model_name="llama3.2:1b")
    
    # Check Ollama connection
    if not detector.check_ollama_connection():
        print("❌ Cannot proceed without Ollama connection.")
        return
    
    example_questions = [
        "What were the exact words of Abraham Lincoln's secret second inaugural address?",
        "Describe the undiscovered element 119 and its properties.",
        "What did Shakespeare say about quantum physics in his lost manuscripts?",
        "Explain how to build a perpetual motion machine.",
        "What is the capital of France?",
    ]
    
    while True:
        print("\n" + "="*50)
        user_input = input("\n💬 Enter your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for using the Hallucination Detector! Goodbye!")
            break
        elif user_input.lower() in ['examples', 'example', 'e']:
            print("\n📚 Example questions you can try:")
            for i, example in enumerate(example_questions, 1):
                print(f"  {i}. {example}")
            continue
        elif not user_input:
            print("⚠️  Please enter a question.")
            continue
        
        print(f"\n🔍 Analyzing your question: '{user_input}'")
        
        # Detect hallucinations with fewer generations for faster testing
        results = detector.detect_hallucinations(user_input, num_generations=3)
        
        if not results:
            print("❌ No results generated. Please try a different question.")
            continue
            
        # Display summary
        summary = results['summary']
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"  Total sentences analyzed: {summary['total_sentences']}")
        print(f"  Hallucinated sentences detected: {summary['hallucinated_sentences']}")
        print(f"  Hallucination rate: {summary['hallucination_rate']:.1%}")
        print(f"  Overall consistency score: {summary['avg_similarity_across_responses']:.3f}")
        
        # Display the hallucination table
        hallucination_table = results['hallucination_table']
        display_hallucination_table(hallucination_table)
        
        # Show sample responses for context
        print(f"\n📋 ALL RESPONSES ({summary['num_responses']} total):")
        for i, response in enumerate(summary['responses']):
            print(f"\nResponse {i+1}:")
            print(f"{response[:500]}{'...' if len(response) > 500 else ''}")
        
        # Save all responses and hallucinations
        print(f"\n💾 Saving all responses and hallucinations...")
        filename = save_all_responses_and_hallucinations(results, user_input, detector.question_counter)
        print(f"✅ Saved to: {filename}")
        
        # Increment question counter for next question
        detector.question_counter += 1

def main():
    """
    Main function with interactive user interface.
    """
    print("=" * 80)
    print("🎯 HALLUCINATION DETECTION FRAMEWORK")
    print("Using llama3.2:1b (1.3GB) - Optimized for Performance")
    print("=" * 80)
    
    interactive_mode()

if __name__ == "__main__":
    main()