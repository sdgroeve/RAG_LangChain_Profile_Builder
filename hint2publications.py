import requests
import unicodedata
import re
import json
from bs4 import BeautifulSoup
from datetime import datetime

# Define the input and output file paths
input_file = "researchers/test.researchers.txt"  # Text file containing researcher names (one per line)
output_json_file = "output/test.publications_data.json"

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
    except requests.RequestException:
        return False

def extract_author_position(soup, researcher_surname):
    """Checks if the researcher is the first or last author based on their surname."""
    authors = soup.find_all('meta', attrs={'name': re.compile(r'citation_author|dc.creator')})
    author_names = [author['content'] for author in authors if 'content' in author.attrs]
    if not author_names:
        return False  # No authors listed
    
    # Check if the researcher is the first or last author
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
            
            # Check author position
            if not extract_author_position(soup, researcher_surname):
                return None
            
            # Extract abstract
            abstract = soup.find('dd', itemprop='description')
            abstract_text = abstract.text.strip() if abstract else "Abstract not available"
            
            # Extract publication type
            publication_type = soup.find('dd', string=re.compile(r'Journal Article'))
            publication_type_text = publication_type.text.strip() if publication_type else "Type not specified"
            publication_type_text = re.sub(r"\s+", " ", publication_type_text)
            
            # Extract DOI
            doi_element = soup.find('meta', attrs={'name': 'dc.identifier', 'content': re.compile(r'doi\.org')})
            doi = doi_element['content'] if doi_element else "DOI not available"
            
            # Extract UGent classification
            classification = soup.find('dt', string="UGent classification")
            classification_text = classification.find_next('dd').text.strip() if classification else "Classification not specified"
            
            # Extract keywords
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
    except requests.RequestException:
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
                    # Filter by year (past 9 years)
                    if current_year - publication_year <= 9:
                        publications.append((publication_url, publication_year))
            return publications
        return []
    except requests.RequestException:
        return []

def main():
    data = {}

    # Read the names from the input file
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            names = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
        return

    # Process each name
    for name in names:
        print(name)
        data[name] = []
        researcher_surname = name.split()[-1]  # Extract surname
        possible_urls = construct_possible_urls(name)
        for url in possible_urls:
            if check_url_exists(url):
                publication_links = extract_publication_urls(url)
                for pub_url, pub_year in publication_links:
                    details = extract_publication_details(pub_url, researcher_surname)
                    if details and details.get("classification") == "A1":
                        data[name].append({
                            "year": pub_year,
                            "url": pub_url,
                            **details
                        })
                break

    # Write data to JSON
    with open(output_json_file, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"Data has been written to '{output_json_file}'.")

if __name__ == "__main__":
    main()
