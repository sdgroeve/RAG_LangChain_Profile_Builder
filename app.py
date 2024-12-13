# app.py
import os
import json
import argparse
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearcherExpertiseRAG:
    def __init__(self, 
                 expertise_data: dict,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 response_model: str = 'llama3'):
        """
        Initialize the RAG system with researcher expertise data and Ollama integration.
        """
        self.expertise_data = expertise_data
        self.model = SentenceTransformer(embedding_model)
        
        # Ollama configuration
        self.response_model = response_model
        self.ollama_url = "http://localhost:11434/api/chat"

        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client()
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        # Populate database once during initialization
        self._populate_database()

    def _test_ollama_connection(self):
        """
        Test connection to Ollama server
        """
        try:
            response = requests.get("http://localhost:11434/api/version")
            response.raise_for_status()
            logger.info("Successfully connected to Ollama server")
        except requests.RequestException as e:
            logger.error(f"Failed to connect to Ollama server: {str(e)}")
            logger.error("Please ensure Ollama is running on localhost:11434")
            raise ConnectionError("Could not connect to Ollama server. Please ensure it is running.")

    def _evaluate_query(self, query: str) -> tuple[bool, str]:
        """
        Evaluate if the query is related to researcher expertise search.
        Returns a tuple of (is_valid, message).
        """
        # Construct prompt for query evaluation
        evaluation_prompt = f"""Evaluate if the following query is related to finding researchers based on expertise, scientific input, or academic collaboration. 
Query: "{query}"

Consider if the query:
1. Asks about specific research expertise or academic topics
2. Seeks to find researchers with particular knowledge
3. Is related to academic or scientific collaboration
4. Is relevant to academic research or scientific domains

Respond with either:
- If valid: "VALID: <reason>"
- If invalid: "INVALID: <explanation of what kind of queries are expected>"
"""

        # Ollama API request payload for evaluation
        payload = {
            "model": self.response_model,
            "messages": [
                {
                    "role": "user",
                    "content": evaluation_prompt
                }
            ],
            "stream": False
        }

        try:
            # Send request to Ollama
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            
            # Extract response
            evaluation = response.json()['message']['content']
            
            # Parse the evaluation
            is_valid = evaluation.startswith("VALID:")
            message = evaluation[7:].strip()  # Remove "VALID: " or "INVALID: " prefix
            
            return is_valid, message

        except requests.RequestException as e:
            logger.error(f"Error evaluating query: {str(e)}")
            return False, f"Error evaluating query: {str(e)}"

    def _populate_database(self):
        """
        Embed and store researcher expertise chunks in ChromaDB.
        """
        try:
            # Delete collection if it exists
            try:
                self.chroma_client.delete_collection("researcher_expertises")
            except:
                pass

            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name="researcher_expertises", 
                metadata={"hnsw:space": "cosine"}
            )

            document_ids = []
            embeddings = []
            metadatas = []
            
            for researcher, expertise in self.expertise_data.items():
                chunks = self._chunk_expertise(expertise)
                
                for i, chunk in enumerate(chunks):
                    # Create unique ID
                    doc_id = f"{researcher}_{i}"
                    document_ids.append(doc_id)
                    
                    # Embed chunk
                    embedding = self.model.encode(chunk).tolist()
                    embeddings.append(embedding)
                    
                    # Store metadata
                    metadatas.append({
                        "researcher": researcher,
                        "original_text": chunk
                    })

            # Add to collection
            self.collection.add(
                ids=document_ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Successfully populated database with {len(document_ids)} chunks")
        except Exception as e:
            logger.error(f"Error populating database: {str(e)}")
            raise

    def _chunk_expertise(self, expertise_text: str) -> list:
        """
        Split researcher expertise into meaningful chunks.
        """
        chunks = [
            chunk.strip() for chunk in expertise_text.split('\n') 
            if chunk.strip() and len(chunk.strip()) > 30
        ]
        return chunks

    def search_expertise(self, query: str, top_k: int = 5) -> dict:
        """
        Search for researchers with similar expertise and generate a comprehensive response.
        First evaluates if the query is appropriate for expertise search.
        """
        # Evaluate query first
        is_valid_query, evaluation_message = self._evaluate_query(query)
        
        if not is_valid_query:
            return {
                'query': query,
                'search_results': [],
                'generated_response': f"Invalid Query: {evaluation_message}",
                'is_valid': False
            }

        try:
            # Proceed with search if query is valid
            # Embed query
            query_embedding = self.model.encode(query).tolist()
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            # Group results by researcher
            researcher_expertise = defaultdict(list)
            researcher_scores = defaultdict(list)
            
            for i in range(len(results['ids'][0])):
                researcher = results['metadatas'][0][i]['researcher']
                expertise = results['metadatas'][0][i]['original_text']
                score = results['distances'][0][i]
                
                researcher_expertise[researcher].append(expertise)
                researcher_scores[researcher].append(score)

            # Create consolidated results
            formatted_results = []
            for researcher in researcher_expertise:
                # Calculate average similarity score
                avg_score = sum(researcher_scores[researcher]) / len(researcher_scores[researcher])
                # Combine all expertise chunks
                combined_expertise = ' '.join(researcher_expertise[researcher])
                
                formatted_results.append({
                    'researcher': researcher,
                    'expertise_chunk': combined_expertise,
                    'similarity_score': avg_score
                })

            # Sort by average similarity score
            formatted_results.sort(key=lambda x: x['similarity_score'])

            # Generate response using Ollama
            response = self._generate_ollama_response(query, formatted_results)

            return {
                'query': query,
                'search_results': formatted_results,
                'generated_response': response,
                'is_valid': True
            }
        except Exception as e:
            logger.error(f"Error during RAG process: {str(e)}")
            return {
                'query': query,
                'search_results': [],
                'generated_response': f"Error during RAG process: {str(e)}",
                'is_valid': False
            }

    def _generate_ollama_response(self, query: str, search_results: list) -> str:
        """
        Generate a comprehensive response using Ollama/Llama3.
        """
        # Construct prompt with context and researcher data
        prompt = f"""Based on the following query and researcher expertise information, generate a response that provides one comprehensive summary for each researcher. Each researcher should be mentioned exactly once.

Query: {query}

Context:
"""
        for result in search_results:
            prompt += f"Researcher: {result['researcher']}\n"
            prompt += f"Expertise: {result['expertise_chunk']}\n\n"
        
        prompt += """Generate a response that:
1. Creates ONE detailed paragraph per researcher
2. Starts each paragraph with the researcher's name in bold using HTML tags (<strong>Name</strong>)
3. Provides a comprehensive summary of their expertise related to the query
4. Highlights key insights and potential collaborations
5. Ensures each researcher is mentioned exactly once
6. Maintains a clear and professional tone

Important: Do not repeat researchers or split their expertise across multiple paragraphs."""

        # Ollama API request payload
        payload = {
            "model": self.response_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }

        try:
            # Send request to Ollama
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            
            # Extract response
            generated_text = response.json()['message']['content']
            return generated_text

        except requests.RequestException as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

def run_cli_mode(expertise_data_path: str, query: str):
    """
    Run the RAG system in CLI mode with a single query.
    """
    try:
        # Load expertise data
        with open(expertise_data_path, 'r') as f:
            expertise_data = json.load(f)
        
        # Initialize RAG system
        rag_system = ResearcherExpertiseRAG(expertise_data=expertise_data)
        
        # Process query
        result = rag_system.search_expertise(query)
        
        # Print results
        if result['is_valid']:
            print("\nQuery Results:")
            print("-" * 80)
            print(f"Query: {result['query']}")
            print("-" * 80)
            print("\nGenerated Response:")
            print(result['generated_response'])
            print("\nDetailed Search Results:")
            print("-" * 80)
            for r in result['search_results']:
                print(f"\nResearcher: {r['researcher']}")
                print(f"Similarity Score: {r['similarity_score']:.4f}")
                print(f"Expertise: {r['expertise_chunk'][:200]}...")
        else:
            print("\nInvalid Query:")
            print(result['generated_response'])
    except Exception as e:
        logger.error(f"Error in CLI mode: {str(e)}")
        print(f"Error: {str(e)}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load JSON data for web mode
json_data = {}
expertise_file = os.path.join('output', 'publications_data_expertise_summary.json')
try:
    if os.path.exists(expertise_file):
        with open(expertise_file, 'r') as f:
            json_data = json.load(f)
            logger.info(f"Successfully loaded expertise data from {expertise_file}")
    else:
        logger.warning(f"Expertise data file not found at {expertise_file}")
except Exception as e:
    logger.error(f"Error loading expertise data: {str(e)}")

# Initialize RAG system for web mode
try:
    rag_system = ResearcherExpertiseRAG(expertise_data=json_data)
    logger.info("Successfully initialized RAG system")
except Exception as e:
    logger.error(f"Error initializing RAG system: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '')
    
    try:
        # Perform RAG search and generate response
        result = rag_system.search_expertise(query)
        
        # Prepare response with search results and generated text
        response = {
            'message': result['generated_response'],
            'search_results': result['search_results'] if result.get('is_valid', False) else []
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({
            'message': f"An error occurred: {str(e)}",
            'search_results': []
        }), 500

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Researcher Expertise RAG System')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode')
    parser.add_argument('--data', type=str, help='Path to expertise data JSON file')
    parser.add_argument('--query', type=str, help='Query string for CLI mode')
    parser.add_argument('--port', type=int, default=5001, help='Port for web server (default: 5001)')
    
    args = parser.parse_args()
    
    if args.cli:
        if not args.data or not args.query:
            print("Error: Both --data and --query are required for CLI mode")
            parser.print_help()
            exit(1)
        run_cli_mode(args.data, args.query)
    else:
        app.run(debug=True, port=args.port)
