import ollama
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import re
import os
import sys
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

class HallucinationDetector:
    def __init__(self, model_name: str = "llama3:latest"):
        """
        Initialize the hallucination detector with specified LLM model.
        
        Args:
            model_name: Name of the Ollama model to use for generation
        """
        self.model_name = model_name
        
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
        Check if Ollama is running and the specified model is available.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Check if Ollama is running
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.model_name not in available_models:
                print(f"❌ Model '{self.model_name}' not found in available models.")
                print(f"Available models: {available_models}")
                return False
            
            print(f"✅ Model '{self.model_name}' is available")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print("Please ensure Ollama is installed and running.")
            print("Install from: https://ollama.ai/")
            return False
    
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate a response using the specified Ollama model.
        
        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature for generation
            
        Returns:
            Generated response text
        """
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': temperature}
            )
            return response['response']
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return ""
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy or simple regex fallback.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
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
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            
        Returns:
            Similarity score between 0 and 1
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
        
        Args:
            sentence: Input sentence
            
        Returns:
            Set of key entities/concepts
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
    
    def detect_hallucinations(self, prompt: str, num_generations: int = 5, 
                            similarity_threshold: float = 0.3) -> Dict:
        """
        Detect hallucinations by comparing multiple generations.
        
        Args:
            prompt: Input prompt for generation
            num_generations: Number of times to generate response
            similarity_threshold: Threshold for considering sentences dissimilar
            
        Returns:
            Dictionary containing detection results and hallucination table
        """
        print(f"\n📝 Generating {num_generations} responses for your question...")
        print(f"Question: '{prompt}'")
        
        # Generate multiple responses
        responses = []
        for i in range(num_generations):
            print(f"  Generating response {i+1}/{num_generations}...")
            response = self.generate_response(prompt, temperature=0.7 + (i * 0.1))
            if response:
                responses.append(response)
            else:
                print(f"  ⚠️  Failed to generate response {i+1}")
        
        if not responses:
            print("❌ No responses generated. Please check Ollama connection.")
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

def interactive_mode():
    """
    Interactive mode where users can input their own questions.
    """
    print("🎯 INTERACTIVE HALLUCINATION DETECTION")
    print("=" * 50)
    print("Ask any question and I'll analyze it for potential hallucinations!")
    print("Type 'quit' or 'exit' to stop the program.")
    print("Type 'examples' to see some example questions.")
    print("=" * 50)
    
    # Initialize detector
    detector = HallucinationDetector(model_name="llama3:latest")
    
    # Check Ollama connection
    if not detector.check_ollama_connection():
        print("❌ Cannot proceed without Ollama connection.")
        return
    
    example_questions = [
        "Explain the causes of World War I and its main consequences.",
        "What are the health benefits of intermittent fasting?",
        "Describe the latest advancements in quantum computing.",
        "How do black holes form and what happens inside them?",
        "What is the capital of France and its main attractions?",
        "Explain the theory of relativity in simple terms.",
        "What are the main features of Python programming language?"
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
        
        # Detect hallucinations
        results = detector.detect_hallucinations(user_input, num_generations=5)
        
        if not results:
            print("❌ No results generated. Please try a different question.")
            continue
            
        # Display summary
        summary = results['summary']
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"  Total sentences analyzed: {summary['total_sentences']}")
        print(f"  Potentially hallucinated sentences: {summary['hallucinated_sentences']}")
        print(f"  Hallucination confidence: {summary['hallucination_rate']:.1%}")
        print(f"  Overall consistency score: {summary['avg_similarity_across_responses']:.3f}")
        
        # Display hallucination table
        hallucination_table = results['hallucination_table']
        if not hallucination_table.empty:
            hallucinated_sentences = hallucination_table[hallucination_table['is_hallucinated']]
            
            if not hallucinated_sentences.empty:
                print(f"\n🚨 POTENTIAL HALLUCINATIONS DETECTED:")
                print("=" * 80)
                for idx, row in hallucinated_sentences.iterrows():
                    print(f"\nResponse {row['response_id'] + 1}:")
                    print(f"  📝 Sentence: {row['sentence']}")
                    print(f"  📊 Consistency score: {row['avg_similarity']} (lower = more likely hallucinated)")
                    print(f"  🔑 Key entities: {row['key_entities'] if row['key_entities'] else 'None detected'}")
                    print("-" * 80)
            else:
                print("\n✅ No potential hallucinations detected. The responses are consistent!")
        
        # Show sample responses for context
        print(f"\n📋 SAMPLE RESPONSES (showing 2 of {summary['num_responses']}):")
        for i, response in enumerate(summary['responses'][:2]):
            print(f"\nResponse {i+1}:")
            print(f"{response[:300]}{'...' if len(response) > 300 else ''}")
        
        # Ask if user wants to see more details
        see_details = input("\n🔍 Would you like to see the full detailed analysis table? (y/n): ").strip().lower()
        if see_details in ['y', 'yes']:
            print(f"\n📈 FULL ANALYSIS TABLE:")
            print(hallucination_table[['sentence', 'response_id', 'avg_similarity', 'is_hallucinated']].to_string(index=False))
        
        # Ask if user wants to save results
        save_results = input("\n💾 Would you like to save these results to a CSV file? (y/n): ").strip().lower()
        if save_results in ['y', 'yes']:
            filename = f"hallucination_analysis_{user_input[:20].replace(' ', '_')}.csv"
            detailed_results = []
            for sent_result in results['detailed_results']:
                detailed_results.append({
                    'question': user_input,
                    'response_id': sent_result['response_id'],
                    'sentence': sent_result['sentence'],
                    'avg_similarity': sent_result['avg_similarity'],
                    'max_similarity': sent_result['max_similarity'],
                    'is_hallucinated': sent_result['is_hallucinated'],
                    'key_entities': sent_result['key_entities']
                })
            
            detailed_df = pd.DataFrame(detailed_results)
            detailed_df.to_csv(filename, index=False)
            print(f"✅ Results saved to '{filename}'")

def main():
    """
    Main function with interactive user interface.
    """
    print("=" * 80)
    print("🎯 HALLUCINATION DETECTION FRAMEWORK")
    print("Interactive Mode - Ask Any Question!")
    print("=" * 80)
    
    interactive_mode()

if __name__ == "__main__":
    main()