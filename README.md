# Profile RAG System

[Previous content remains the same until Data Preparation section]

## Data Preparation

The system includes two scripts for preparing and processing researcher expertise data:

### parse_publications.py

This script handles the complete pipeline of extracting publication data and generating expertise descriptions. It combines publication retrieval and expertise generation into a single workflow with detailed logging and statistics.

Features:
- Retrieves publications from UGent's research portal
- Filters for publications where the researcher is first or last author
- Focuses on A1 journal articles from the past 9 years
- Generates dual expertise descriptions using both Llama3 and Llama-3.3-70B models
- Provides detailed statistics and logging of the entire process

Usage:
```bash
python parse_publications.py researchers/input.txt [--publications output_publications.json] [--expertise output_expertise.json]
```

### generate_embeddings.py

This script handles the generation and storage of embeddings for the expertise data. It processes the expertise summaries and creates a binary file that can be efficiently loaded by the web application.

Features:
- Generates embeddings for both base and llama-70b summaries
- Uses SentenceTransformer for high-quality embeddings
- Stores embeddings in a pickle file for fast loading
- Maintains metadata for each embedding
- Supports both new dual-summary format and legacy format

Usage:
```bash
python generate_embeddings.py [--input output/T.expertise.json] [--model all-MiniLM-L6-v2] [--output-dir embeddings]
```

Parameters:
- `--input`: Input expertise data JSON file (default: output/T.expertise.json)
- `--model`: Sentence transformer model to use (default: all-MiniLM-L6-v2)
- `--output-dir`: Output directory for embeddings (default: embeddings)

## System Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is running locally with both required models:
```bash
# Install Ollama if not already installed
curl https://ollama.ai/install.sh | sh
# Pull the required models
ollama pull llama3
ollama pull llama3:70b
```

4. Generate expertise data:
```bash
# Generate expertise data with summaries
python parse_publications.py researchers/input.txt

# Generate embeddings for the expertise data
python generate_embeddings.py
```

5. Start the web application:
```bash
python app.py
```

## System Architecture

The system is split into three main components:

1. **Data Generation (parse_publications.py)**
   - Retrieves researcher publications
   - Generates expertise summaries using dual LLM models
   - Outputs JSON data with publication details and summaries

2. **Embedding Generation (generate_embeddings.py)**
   - Processes expertise summaries
   - Generates embeddings using SentenceTransformer
   - Creates and stores embedding data for efficient loading
   - Handles both new dual-summary and legacy formats

3. **Web Application (app.py)**
   - Loads pre-generated embeddings
   - Provides web interface for expertise search
   - Handles real-time querying and response generation
   - Maintains ChromaDB for efficient similarity search

This separation of concerns allows for:
- Independent scaling of data processing and web serving
- Efficient resource usage (embeddings are generated once and reused)
- Easy updates to expertise data without web application changes
- Better maintainability and debugging

[Rest of the content remains the same]
