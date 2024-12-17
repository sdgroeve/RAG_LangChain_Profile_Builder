# Profile RAG System

A Flask-based web application that uses Retrieval-Augmented Generation (RAG) to intelligently search and match researcher expertise based on natural language queries. The system provides detailed, context-aware responses about researchers' expertise and potential collaborations.

## Features

- 🔍 Natural language query processing for expertise search
- 🧠 Semantic search using sentence transformers
- 💡 Intelligent query validation
- 🤖 AI-powered response generation using Ollama/Llama3
- 🎯 Relevance-based researcher matching
- 💻 Clean, responsive web interface
- 🔄 Real-time chat-like interaction
- 📟 Command-line interface support
- 💾 Persistent embeddings storage
- 🔄 On-demand embedding regeneration

## System Requirements

- Python 3.8+
- Ollama running locally with Llama3 model
- Sufficient RAM for embedding operations
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is running locally with the Llama3 model:
```bash
# Install Ollama if not already installed
curl https://ollama.ai/install.sh | sh
# Pull the Llama3 model
ollama pull llama3
```

4. Prepare your researcher expertise data in JSON format:
```json
{
    "researcher_name": "expertise description",
    ...
}
```

## Data Preparation

The system includes a comprehensive script for preparing researcher expertise data:

### parse_publications.py

This script handles the complete pipeline of extracting publication data and generating expertise descriptions. It combines publication retrieval and expertise generation into a single workflow with detailed logging and statistics.

Features:
- Retrieves publications from UGent's research portal
- Filters for publications where the researcher is first or last author
- Focuses on A1 journal articles from the past 9 years
- Generates expertise descriptions using Ollama/Llama3
- Provides detailed statistics and logging of the entire process

The script extracts and processes:
- Abstract
- Publication type
- DOI
- UGent classification
- Keywords
- Generated expertise summaries

Usage:
```bash
python parse_publications.py researchers/input.txt [--publications output_publications.json] [--expertise output_expertise.json]
```

Parameters:
- `input.txt`: Text file with researcher names (one per line)
- `--publications`: Output file for raw publication data (default: publications_data.json)
- `--expertise`: Output file for publication data with expertise summaries (default: publications_data_expertise.json)

The script provides detailed logging of:
- Total researchers processed
- Total publications found and successfully parsed
- Number of expertise descriptions generated
- Failed operations (URL checks, publication fetches, etc.)
- Researchers with no publications found

Output files:
- Publications data file: Contains the raw publication data
- Expertise data file: Contains the publication data enhanced with expertise summaries

## Usage

The system can be used in two modes: Web Interface or Command Line Interface (CLI).

### Web Interface Mode

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5001
```

3. Enter natural language queries about researcher expertise in the chat interface, for example:
- "Who has expertise in machine learning?"
- "Find researchers working on climate change"
- "Which researchers specialize in quantum computing?"

### Command Line Interface (CLI) Mode

The CLI mode allows you to query researcher expertise directly from the command line:

```bash
python app.py --cli --data path/to/expertise.json --query "your query here"
```

Parameters:
- `--cli`: Activates CLI mode
- `--data`: Path to your expertise data JSON file
- `--query`: Your expertise search query
- `--port`: (Optional) Specify port for web mode (default: 5001)

Example CLI usage:
```bash
# Search for machine learning experts
python app.py --cli --data researchers.json --query "Who has expertise in machine learning?"

# Find climate change researchers
python app.py --cli --data researchers.json --query "Find researchers working on climate change"
```

CLI output includes:
- Query validation result
- Generated response summary
- Detailed search results with similarity scores
- Individual researcher expertise snippets

## Technical Details

### Architecture

The system consists of several key components:

1. **Vector Database (ChromaDB)**
   - Stores embedded researcher expertise data
   - Enables efficient semantic search
   - Maintains metadata for result generation

2. **Embedding Model (Sentence Transformers)**
   - Model: 'all-MiniLM-L6-v2'
   - Converts text to dense vector representations
   - Enables semantic similarity matching

3. **LLM Integration (Ollama/Llama3)**
   - Query validation
   - Response generation
   - Context-aware summaries

4. **Web Interface**
   - Real-time chat interface
   - Responsive design using Tailwind CSS
   - Dynamic message formatting

5. **Embedding Management**
   - Persistent storage in `embeddings/` directory
   - Automatic loading of existing embeddings
   - On-demand regeneration via API endpoint

### Key Classes

#### ResearcherExpertiseRAG

Main class handling the RAG system functionality:

- `__init__`: Initializes the RAG system with expertise data and models
- `search_expertise`: Performs semantic search and generates responses
- `_evaluate_query`: Validates if queries are expertise-related
- `_generate_ollama_response`: Creates detailed responses using LLM
- `_load_embeddings`: Loads persisted embeddings from disk
- `_save_embeddings`: Saves embeddings to disk for future use
- `_populate_database`: Generates and stores embeddings

### API Endpoints

- `GET /`: Serves the main chat interface
- `POST /chat`: Handles expertise queries and returns AI-generated responses
- `POST /regenerate-embeddings`: Forces regeneration of embeddings

### Embedding Management

The system now includes persistent storage of embeddings to improve startup time and resource usage:

1. **Storage Location**
   - Embeddings are stored in the `embeddings/` directory
   - Default file: `embeddings/embeddings.pkl`

2. **Automatic Loading**
   - System checks for existing embeddings on startup
   - If found, loads them instead of regenerating
   - Falls back to generation if loading fails

3. **Manual Regeneration**
   - Endpoint: `POST /regenerate-embeddings`
   - Forces complete regeneration of embeddings
   - Useful after data updates or if embeddings become corrupted

Example of forcing embedding regeneration:
```bash
curl -X POST http://localhost:5001/regenerate-embeddings
```

## Implementation Details

1. **Query Processing**
   - Queries are validated using LLM to ensure relevance
   - Valid queries are embedded using sentence transformers
   - Semantic search is performed against stored expertise vectors

2. **Response Generation**
   - Top-k similar expertise chunks are retrieved
   - Results are grouped by researcher
   - LLM generates comprehensive summaries
   - Responses include expertise details and collaboration potential

3. **Data Storage**
   - Researcher expertise is stored in ChromaDB
   - Embeddings are persisted to disk for efficiency
   - Metadata maintains researcher-expertise relationships

4. **Web Interface**
   - Real-time interaction using JavaScript
   - Dynamic message formatting
   - Loading states and error handling
   - Responsive design for various screen sizes

5. **CLI Interface**
   - Direct command-line querying
   - Formatted console output
   - Support for custom data sources
   - Batch processing capability
