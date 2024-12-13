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

## Data Preparation Scripts

The system includes two scripts for preparing researcher expertise data:

### hint2publications.py

This script extracts publication data from UGent's research portal for specified researchers. It focuses on:
- Retrieving publications where the researcher is either first or last author
- Filtering for A1 journal articles from the past 9 years
- Extracting key information including:
  - Abstract
  - Publication type
  - DOI
  - UGent classification
  - Keywords

Usage:
1. Create a text file with researcher names (one per line) in the `researchers` directory
2. Update the input and output file paths in the script:
```python
input_file = "researchers/test.researchers.txt"
output_json_file = "output/test.publications_data.json"
```
3. Run the script:
```bash
python hint2publications.py
```

### generate_expertise.py

This script processes the publication data extracted by hint2publications.py to generate expertise summaries using Ollama/Llama3. It performs:
- Generation of concise summaries for each publication's abstract
- Creation of comprehensive expertise profiles by analyzing patterns across a researcher's publications
- Output of both individual publication summaries and overall researcher expertise profiles

Usage:
1. Ensure hint2publications.py has been run first to generate the publication data
2. Run the script:
```bash
python generate_expertise.py
```

The script generates two output files:
- `output/test.publications_data_expertise.json`: Contains the original publication data enhanced with individual paper summaries
- `output/test.publications_data_expertise_summary.json`: Contains the final expertise profiles for each researcher

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

### Key Classes

#### ResearcherExpertiseRAG

Main class handling the RAG system functionality:

- `__init__`: Initializes the RAG system with expertise data and models
- `search_expertise`: Performs semantic search and generates responses
- `_evaluate_query`: Validates if queries are expertise-related
- `_generate_ollama_response`: Creates detailed responses using LLM

### API Endpoints

- `GET /`: Serves the main chat interface
- `POST /chat`: Handles expertise queries and returns AI-generated responses

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
   - Embeddings are generated using sentence transformers
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
