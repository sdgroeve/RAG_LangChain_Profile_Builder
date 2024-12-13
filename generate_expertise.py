from langchain.llms import Ollama
import json

# Initialize Ollama LLM
llm = Ollama(model="llama3", temperature=0)

# Define a function to generate expertise descriptions
def generate_expertise_description(abstract):
    prompt = (
        f"Summarize the key points of this research paper abstract in no more than two sentences, focusing on the task being addressed and the solution proposed, while minimizing emphasis on conclusions or results."
        f"Just write the summary, don't start with Here is the summary or similar"
        f"Abstract: {abstract}\n\n"
    )
    response = llm(prompt)
    return response.strip()

# Define a function to summarize expertise by researcher
def summarize_researcher_expertise(researcher, expertise_list):
    combined_expertise = "\n".join(expertise_list)
    prompt = (
        f'Given the following list of abstract summaries from research papers where the researcher is either the first or last author, identify the overlapping areas of expertise demonstrated across multiple papers. Focus on recurring themes, methodologies, and domains of application to generate a concise researcher profile highlighting their core expertises and research focus.'
        f"Do not write an introduction, just the expertises."
        f"\n\nHere are the summaries: {combined_expertise}"
    )
    response = llm(prompt)
    return response.strip()

# Load the JSON data
with open('output/test.publications_data.json', 'r') as file:
    data = json.load(file)

# Process each publication and group expertise by researcher
expertise_by_researcher = {}
for author, publications in data.items():
    print(author)
    expertise_by_researcher[author] = []
    for pub in publications:
        abstract = pub.get("abstract", "")
        if abstract:
            expertise = generate_expertise_description(abstract)
            pub["summary"] = expertise
            # Add the expertise for this paper to the author's group
            expertise_by_researcher[author].append(expertise)

# Generate a detailed expertise description for each researcher
final_expertise_by_researcher = {}
for researcher, expertise_list in expertise_by_researcher.items():
    if expertise_list:
        final_expertise_by_researcher[researcher] = summarize_researcher_expertise(researcher, expertise_list)

# Save the final expertise descriptions to a new JSON file
with open('output/test.publications_data_expertise_summary.json', 'w') as file:
    json.dump(final_expertise_by_researcher, file, indent=4)

# Save the updated publications data with individual expertise descriptions
with open('output/test.publications_data_expertise.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Final expertise descriptions generated and saved successfully.")
