import json
import os

# 1. THE PARSER ENGINE
def parse_government_scheme(file_content):
    """
    Parses a standardized government scheme text into a hierarchical dictionary.
    """
    expected_sections =[
        "Tags", "Details", "Benefits", "Eligibility",
        "Application Process", "Documents Required"
    ]

    # Split the text into lines and remove empty lines
    lines =[line.strip() for line in file_content.split('\n') if line.strip()]

    if len(lines) < 2:
        return None # File is too short/empty

    # Extract the metadata (Line 1 & 2)
    ministry_or_region = lines[0]
    scheme_title = lines[1]

    # Initialize the hierarchical dictionary for this specific document
    doc_data = {
        "Ministry": ministry_or_region,
        "Title": scheme_title
    }

    current_section = "Overview" # Fallback if text starts before a header
    section_content = []

    # Iterate through the rest of the lines
    for line in lines[2:]:
        # Check if the line matches any expected section header
        is_header = False
        matched_header = ""
        for sec in expected_sections:
            if line.lower() == sec.lower():
                is_header = True
                matched_header = sec # Enforce consistent casing
                break

        if is_header:
            # Save the PREVIOUS section before moving to the new one
            if section_content:
                if current_section == "Tags":
                    # Convert comma-separated string into a clean JSON list/array
                    tags_string = ' '.join(section_content)
                    doc_data["Tags"] =[t.strip() for t in tags_string.split(',') if t.strip()]
                else:
                    doc_data[current_section] = '\n'.join(section_content)

            # Reset for the NEW section
            current_section = matched_header
            section_content =[]
        else:
            section_content.append(line)

    # Save the very last section processed in the loop
    if section_content:
        if current_section == "Tags":
            tags_string = ' '.join(section_content)
            doc_data["Tags"] =[t.strip() for t in tags_string.split(',') if t.strip()]
        else:
            doc_data[current_section] = '\n'.join(section_content)

    return doc_data


# 2. FILE PROCESSING MANAGER
def process_all_files(input_directory, output_json_file):
    """
    Reads all .txt files in a directory and compiles them into a hierarchical JSON.
    """
    master_corpus = {}
    doc_counter = 1

    if not os.path.exists(input_directory):
        print(f"Directory '{input_directory}' not found. Please create it and add .txt files.")
        return

    # Use sorted() to ensure doc_001, doc_002 map to files in alphabetical order
    for filename in sorted(os.listdir(input_directory)):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_directory, filename)

            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

            # --- FILTERING OUT INVALID / NOT FOUND PAGES ---
            if "Unknown Scheme Name" in content or "not found" in content.lower() or len(content.strip()) < 50:
                print(f"Skipping {filename} (Page not found, failed scrape, or empty)")
                continue

            # Parse the file into a clean dictionary
            doc_data = parse_government_scheme(content)

            # Add to master dictionary under a parent doc key
            if doc_data:
                doc_id = f"doc_{doc_counter:03d}"
                master_corpus[doc_id] = doc_data
                doc_counter += 1

    # Save to JSON
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(master_corpus, f, indent=4, ensure_ascii=False)

    print(f"\n--- SUCCESS ---")
    print(f"Successfully processed {doc_counter - 1} valid files.")
    print(f"Saved to: {output_json_file}")

INPUT_FOLDER = "Scheme_Data" 
OUTPUT_FILE = "gov_corpus.json"
process_all_files(INPUT_FOLDER, OUTPUT_FILE)
