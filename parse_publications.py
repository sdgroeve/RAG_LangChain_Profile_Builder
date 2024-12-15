import requests
import unicodedata
import re
import json
import argparse
import logging
import os
from bs4 import BeautifulSoup
from datetime import datetime
from langchain_ollama import OllamaLLM

# Suppress Ollama HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parse_publications.log'),
        logging.StreamHandler()
    ]
)

def normalize_name(name):
    """Normalizes a name by replacing special characters with their base equivalents."""
    name = name.split("(")[0].strip()
    name = unicodedata.normalize('NFD', name)
    name = ''.join(char for char in name if unicodedata.category(char) != 'Mn')
    name = re.sub(r"[^\w\s]", "", name)
    return name

def construct_possible_urls(name):
    """Constructs potential publication list URLs for a researcher."""
    base_url = "https://research.ugent.be/web/person/"
    normalized_name = normalize_name(name)
    parts = normalized_name.split()
    
    if len(parts) < 2:
        formatted_name = "-".join(parts).lower()
        return [f"{base_url}{formatted_name}-0/publications/en"]

    first_name_last = "-".join(parts).lower()
    last_name_first = f"{parts[-1].lower()}-{'-'.join(parts[:-1]).lower()}"
    
    return [
        f"{base_url}{first_name_last}-0/publications/en",
        f"{base_url}{last_name_first}-0/publications/en"
    ]

def check_url_exists(url):
    """Checks if a URL exists by sending a HEAD request."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException as e:
        logging.error(f"Failed to check URL {url}: {str(e)}")
        return False

def extract_author_position(soup, researcher_surname):
    """Checks if the researcher is the first or last author based on their surname."""
    authors = soup.find_all('meta', attrs={'name': re.compile(r'citation_author|dc.creator')})
    author_names = [author['content'] for author in authors if 'content' in author.attrs]
    if not author_names:
        return False  # No authors listed
    
    researcher_surname = researcher_surname.lower()
    if researcher_surname in author_names[0].lower():
        return True
    if researcher_surname in author_names[-1].lower():
        return True
    return False

def extract_publication_details(publication_url, researcher_surname):
    """Extracts details from a specific publication page if the researcher is first or last author."""
    try:
        response = requests.get(publication_url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if not extract_author_position(soup, researcher_surname):
                return None
            
            abstract = soup.find('dd', itemprop='description')
            abstract_text = abstract.text.strip() if abstract else "Abstract not available"
            
            publication_type = soup.find('dd', string=re.compile(r'Journal Article'))
            publication_type_text = publication_type.text.strip() if publication_type else "Type not specified"
            publication_type_text = re.sub(r"\s+", " ", publication_type_text)
            
            doi_element = soup.find('meta', attrs={'name': 'dc.identifier', 'content': re.compile(r'doi\.org')})
            doi = doi_element['content'] if doi_element else "DOI not available"
            
            classification = soup.find('dt', string="UGent classification")
            classification_text = classification.find_next('dd').text.strip() if classification else "Classification not specified"
            
            keywords = []
            keyword_element = soup.find('dd', itemprop='keywords')
            if keyword_element:
                keyword_links = keyword_element.find_all('a', href=True)
                keywords = [link.text.strip() for link in keyword_links]

            return {
                "abstract": abstract_text,
                "type": publication_type_text,
                "doi": doi,
                "classification": classification_text,
                "keywords": keywords,
            }
    except requests.RequestException as e:
        logging.error(f"Failed to extract details from {publication_url}: {str(e)}")
        return None
    
def extract_publication_urls(publications_url):
    """Extracts publication URLs and years from a researcher's publication page."""
    try:
        response = requests.get(publications_url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            publications = []
            current_year = datetime.now().year
            for publication in soup.find_all('div', class_='bg-blue-hover'):
                link = publication.find('a', href=True)
                year_span = publication.find('div', {'data-type': 'year'})
                if link and year_span:
                    publication_url = link['href']
                    try:
                        publication_year = int(year_span.text.strip())
                    except:
                        publication_year = 1000                    
                    if current_year - publication_year <= 9:
                        publications.append((publication_url, publication_year))
            return publications
        logging.error(f"Failed to fetch publications from {publications_url}: Status code {response.status_code}")
        return []
    except requests.RequestException as e:
        logging.error(f"Failed to fetch publications from {publications_url}: {str(e)}")
        return []

def generate_expertise_description(llm, abstract):
    """Generate expertise description using LLM."""
    try:
        prompt = (
            f"Summarize the key points of this research paper abstract in no more than two sentences, focusing on the task being addressed and the solution proposed, while minimizing emphasis on conclusions or results."
            f"Just write the summary, don't start with Here is the summary or similar"
            f"Abstract: {abstract}\n\n"
        )
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        logging.error(f"Failed to generate expertise description: {str(e)}")
        return "Failed to generate expertise description"

def main():
    parser = argparse.ArgumentParser(description='Parse publications and generate expertise descriptions')
    parser.add_argument('input_file', help='Input file containing researcher names')
    parser.add_argument('--publications', default='publications_data.json',
                      help='Output file for publications data (default: publications_data.json)')
    parser.add_argument('--expertise', default='publications_data_expertise.json',
                      help='Output file for expertise data (default: publications_data_expertise.json)')
    args = parser.parse_args()

    # Initialize statistics
    stats = {
        "total_researchers": 0,
        "total_publications_found": 0,
        "total_publications_parsed": 0,
        "total_embeddings_generated": 0,
        "failed_url_checks": 0,
        "failed_publication_fetches": 0,
        "failed_embedding_generations": 0,
        "researchers_with_no_publications": 0,
        "failed_researcher_urls": []  # New: Track researchers whose URLs all failed
    }

    # Initialize Ollama LLM
    try:
        llm = OllamaLLM(model="llama3", temperature=0)
        logging.info("Successfully initialized LLM model")
    except Exception as e:
        logging.error(f"Failed to initialize LLM model: {str(e)}")
        return
    
    # Step 1: Gather publications data
    data = {}
    try:
        with open(args.input_file, "r", encoding="utf-8") as file:
            names = [line.strip() for line in file if line.strip()]
            stats["total_researchers"] = len(names)
            logging.info(f"Found {len(names)} researchers in input file")
    except FileNotFoundError:
        logging.error(f"Error: The file '{args.input_file}' does not exist.")
        return

    for name in names:
        logging.info(f"Processing researcher: {name}")
        data[name] = []
        researcher_publications_found = 0
        researcher_surname = name.split()[-1]
        
        possible_urls = construct_possible_urls(name)
        url_found = False
        for url in possible_urls:
            if check_url_exists(url):
                url_found = True
                publication_links = extract_publication_urls(url)
                stats["total_publications_found"] += len(publication_links)
                researcher_publications_found = len(publication_links)
                logging.info(f"Found {len(publication_links)} publications for {name}")
                
                successful_parses = 0
                for pub_url, pub_year in publication_links:
                    details = extract_publication_details(pub_url, researcher_surname)
                    if details and details.get("classification") == "A1":
                        data[name].append({
                            "year": pub_year,
                            "url": pub_url,
                            **details
                        })
                        successful_parses += 1
                    else:
                        stats["failed_publication_fetches"] += 1
                
                stats["total_publications_parsed"] += successful_parses
                logging.info(f"Successfully parsed {successful_parses} A1 publications for {name}")
                break
            else:
                stats["failed_url_checks"] += 1
        
        if not url_found:
            logging.warning(f"No valid URL found for researcher: {name}")
            stats["researchers_with_no_publications"] += 1
            # Add all attempted URLs for this researcher to the failed list
            stats["failed_researcher_urls"].extend(possible_urls)

    # Save initial publications data
    with open(args.publications, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    logging.info(f"Publications data saved to '{args.publications}'")

    # Step 2: Generate expertise descriptions
    for author, publications in data.items():
        logging.info(f"Generating expertise descriptions for {author}")
        successful_embeddings = 0
        for pub in publications:
            abstract = pub.get("abstract", "")
            if abstract:
                expertise = generate_expertise_description(llm, abstract)
                if expertise != "Failed to generate expertise description":
                    pub["summary"] = expertise
                    successful_embeddings += 1
                else:
                    stats["failed_embedding_generations"] += 1
        
        stats["total_embeddings_generated"] += successful_embeddings
        logging.info(f"Generated {successful_embeddings} expertise descriptions for {author}")

    # Save updated publications data with expertise
    with open(args.expertise, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    logging.info(f"Publications data with expertise saved to '{args.expertise}'")

    # Log final statistics
    logging.info("\n=== Final Statistics ===")
    logging.info(f"Total researchers processed: {stats['total_researchers']}")
    logging.info(f"Total publications found: {stats['total_publications_found']}")
    logging.info(f"Total A1 publications successfully parsed: {stats['total_publications_parsed']}")
    logging.info(f"Total expertise descriptions generated: {stats['total_embeddings_generated']}")
    logging.info(f"Failed URL checks: {stats['failed_url_checks']}")
    logging.info(f"Failed publication fetches: {stats['failed_publication_fetches']}")
    logging.info(f"Failed expertise generations: {stats['failed_embedding_generations']}")
    logging.info(f"Researchers with no publications found: {stats['researchers_with_no_publications']}")
    
    # Log list of failed URLs
    if stats["failed_researcher_urls"]:
        logging.info("\n=== Failed URLs ===")
        for url in stats["failed_researcher_urls"]:
            logging.info(url)

if __name__ == "__main__":
    main()
