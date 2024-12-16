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
from difflib import get_close_matches

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
        self.researcher_names = list(expertise_data.keys())
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

    def get_researcher_suggestions(self, query: str, max_suggestions: int = 5) -> list:
        """
        Get researcher name suggestions based on a partial query.
        Uses fuzzy matching to find similar names.
        """
        if not query:
            return []
            
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # First try direct startswith matching
        direct_matches = [
            name for name in self.researcher_names 
            if name.lower().startswith(query_lower)
        ]
        
        # If we have enough direct matches, return them
        if len(direct_matches) >= max_suggestions:
            return direct_matches[:max_suggestions]
            
        # Otherwise, use fuzzy matching to find similar names
        all_matches = set(direct_matches)
        
        # Get fuzzy matches
        fuzzy_matches = get_close_matches(
            query, 
            [name for name in self.researcher_names if name not in all_matches],
            n=max_suggestions - len(all_matches),
            cutoff=0.6
        )
        
        all_matches.update(fuzzy_matches)
        
        # Also check for substring matches
        substring_matches = [
            name for name in self.researcher_names 
            if query_lower in name.lower() and name not in all_matches
        ]
        
        all_matches.update(substring_matches[:max_suggestions - len(all_matches)])
        
        return sorted(list(all_matches))[:max_suggestions]

    def get_researcher_expertise(self, researcher_name: str) -> dict:
        """
        Get expertise details for a specific researcher.
        Includes fuzzy matching for researcher names.
        """
        # First try exact match
        print(researcher_name)
        print(self.expertise_data.keys())
        if researcher_name in self.expertise_data:
            return {
                'researcher': researcher_name,
                'publications': self.expertise_data[researcher_name]
            }
            
        # Try case-insensitive match
        for name in self.expertise_data:
            if name.lower() == researcher_name.lower():
                return {
                    'researcher': name,
                    'publications': self.expertise_data[name]
                }
                
        # Try fuzzy matching
        matches = get_close_matches(researcher_name, self.researcher_names, n=1, cutoff=0.6)
        if matches:
            matched_name = matches[0]
            return {
                'researcher': matched_name,
                'publications': self.expertise_data[matched_name]
            }
            
        return None

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
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            evaluation = response.json()['message']['content']
            is_valid = evaluation.startswith("VALID:")
            message = evaluation[7:].strip()
            return is_valid, message

        except requests.RequestException as e:
            logger.error(f"Error evaluating query: {str(e)}")
            return False, f"Error evaluating query: {str(e)}"

    def _populate_database(self):
        """
        Embed paper summaries with researcher metadata in ChromaDB.
        """
        try:
            try:
                self.chroma_client.delete_collection("researcher_papers")
            except:
                pass

            self.collection = self.chroma_client.create_collection(
                name="researcher_papers", 
                metadata={"hnsw:space": "cosine"}
            )

            document_ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for researcher, publications in self.expertise_data.items():
                for i, pub in enumerate(publications):
                    if "summary" in pub:
                        # Create unique ID
                        doc_id = f"{researcher}_{i}"
                        document_ids.append(doc_id)
                        
                        # Get the summary
                        summary = pub["summary"]
                        documents.append(summary)
                        
                        # Embed summary
                        embedding = self.model.encode(summary).tolist()
                        embeddings.append(embedding)
                        
                        # Store metadata
                        metadatas.append({
                            "researcher": researcher,
                            "year": pub.get("year", "Unknown"),
                            "url": pub.get("url", ""),
                            "doi": pub.get("doi", ""),
                            "keywords": ", ".join(pub.get("keywords", [])),
                            "abstract": pub.get("abstract", "")
                        })

            if document_ids:
                self.collection.add(
                    ids=document_ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Successfully populated database with {len(document_ids)} paper summaries")
            else:
                logger.warning("No paper summaries found to populate the database")
        except Exception as e:
            logger.error(f"Error populating database: {str(e)}")
            raise

    def search_expertise(self, query: str, top_k: int = 5) -> dict:
        """
        Search for similar paper summaries and generate expertise summary.
        """
        is_valid_query, evaluation_message = self._evaluate_query(query)
        
        if not is_valid_query:
            return {
                'query': query,
                'search_results': [],
                'generated_response': f"Invalid Query: {evaluation_message}",
                'is_valid': False
            }

        try:
            query_embedding = self.model.encode(query).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            # Group results by researcher
            researcher_papers = defaultdict(list)
            researcher_scores = defaultdict(list)
            
            for i in range(len(results['ids'][0])):
                researcher = results['metadatas'][0][i]['researcher']
                paper_summary = results['documents'][0][i]
                score = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                
                researcher_papers[researcher].append({
                    'summary': paper_summary,
                    'metadata': metadata
                })
                researcher_scores[researcher].append(score)

            # Create consolidated results
            formatted_results = []
            for researcher in researcher_papers:
                avg_score = sum(researcher_scores[researcher]) / len(researcher_scores[researcher])
                papers = researcher_papers[researcher]
                
                formatted_results.append({
                    'researcher': researcher,
                    'papers': papers,
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
        Generate expertise summary based on similar paper summaries.
        """
        prompt = f"""Based on the following research paper summaries, generate a comprehensive expertise profile for each researcher. Focus on their core research areas and methodological strengths.

Query: {query}

Paper Summaries:
"""
        for result in search_results:
            prompt += f"\nResearcher: {result['researcher']}\n"
            for paper in result['papers']:
                prompt += f"- {paper['summary']}\n"
                if paper['metadata']['keywords']:
                    prompt += f"  Keywords: {paper['metadata']['keywords']}\n"
        
        prompt += """
Generate a response that:
1. Creates ONE detailed paragraph per researcher
2. Starts each paragraph with the researcher's name in bold using HTML tags (<strong>Name</strong>)
3. Synthesizes their expertise based on the paper summaries
4. Focuses on recurring themes and methodological approaches
5. Ensures each researcher is mentioned exactly once
6. Maintains a clear and professional tone

Important: Focus on the expertise demonstrated in the papers, not individual paper summaries."""

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
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
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
        with open(expertise_data_path, 'r') as f:
            expertise_data = json.load(f)
        
        rag_system = ResearcherExpertiseRAG(expertise_data=expertise_data)
        result = rag_system.search_expertise(query)
        
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
                for paper in r['papers']:
                    print(f"\nPaper Summary: {paper['summary']}")
                    if paper['metadata']['keywords']:
                        print(f"Keywords: {paper['metadata']['keywords']}")
        else:
            print("\nInvalid Query:")
            print(result['generated_response'])
    except Exception as e:
        logger.error(f"Error in CLI mode: {str(e)}")
        print(f"Error: {str(e)}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variable for RAG system
rag_system = None

def initialize_rag_system():
    """Initialize or reinitialize the RAG system with fresh data."""
    global rag_system
    expertise_file = os.path.join('output', 'publications_data_expertise.json')
    researchers_file = os.path.join('researchers', 'hint.researchers.txt')
    
    try:
        # Read researchers list
        with open(researchers_file, 'r') as f:
            researchers = set(line.strip() for line in f)
        logger.info(f"Loaded {len(researchers)} researchers from {researchers_file}")
        
        # Load expertise data
        if os.path.exists(expertise_file):
            with open(expertise_file, 'r') as f:
                json_data = json.load(f)
                logger.info(f"Successfully loaded expertise data from {expertise_file}")
                
                # Calculate statistics
                researchers_with_data = set(json_data.keys())
                researchers_without_data = researchers - researchers_with_data
                
                # Log statistics
                logger.info(f"Total number of researchers: {len(researchers)}")
                logger.info(f"Number of researchers with data: {len(researchers_with_data)}")
                logger.info(f"Number of researchers without data: {len(researchers_without_data)}")
                
                # Log researchers without data
                if researchers_without_data:
                    logger.info("Researchers without data:")
                    for researcher in sorted(researchers_without_data):
                        logger.info(f"  - {researcher}")
                
                rag_system = ResearcherExpertiseRAG(expertise_data=json_data)
                logger.info("Successfully initialized RAG system")
        else:
            logger.error(f"Expertise data file not found at {expertise_file}")
            raise FileNotFoundError(f"Expertise data file not found at {expertise_file}")
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        raise

@app.route('/')
def index():
    # Ensure RAG system is initialized
    global rag_system
    if rag_system is None:
        initialize_rag_system()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global rag_system
    data = request.json
    query = data.get('message', '')
    
    try:
        # Ensure RAG system is initialized
        if rag_system is None:
            initialize_rag_system()
            
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

@app.route('/expertise', methods=['POST'])
def expertise():
    global rag_system
    data = request.json
    researcher = data.get('researcher', '')
    
    try:
        # Ensure RAG system is initialized
        if rag_system is None:
            initialize_rag_system()
            
        # Get researcher expertise details
        result = rag_system.get_researcher_expertise(researcher)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({
                'error': f'No expertise data found for researcher: {researcher}'
            }), 404
    
    except Exception as e:
        logger.error(f"Error processing expertise request: {str(e)}")
        return jsonify({
            'error': f"An error occurred: {str(e)}"
        }), 500

@app.route('/suggest', methods=['POST'])
def suggest():
    global rag_system
    data = request.json
    query = data.get('query', '')
    
    try:
        # Ensure RAG system is initialized
        if rag_system is None:
            initialize_rag_system()
            
        # Get researcher suggestions
        suggestions = rag_system.get_researcher_suggestions(query)
        return jsonify({'suggestions': suggestions})
    
    except Exception as e:
        logger.error(f"Error processing suggestion request: {str(e)}")
        return jsonify({
            'error': f"An error occurred: {str(e)}",
            'suggestions': []
        }), 500

if __name__ == '__main__':
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
        # Initialize RAG system before starting the server
        initialize_rag_system()
        app.run(debug=True, port=args.port)
