# UniVerna: Hybrid Retrieval Engine for Indian Languages

**UniVerna** is a Cross-Lingual Information Retrieval (CLIR) system designed to solve the **Vocabulary Mismatch** problem in domain-specific vernacular search. It allows users to ask questions about complex Indian Government schemes in their native languages via a Telegram Bot, and retrieves highly accurate, context-aware answers.

## The Problem
Indian government schemes are often locked inside complex, unstructured PDFs and web portals, predominantly written in English. When citizens query these documents in their native languages, standard search engines fail:
* **Semantic Search** captures the meaning but loses exact keyword precision (e.g., losing the exactness of "Form 16" or "Section 144").
* **Sparse/Keyword Search** captures exact entities but completely fails across different languages and scripts.

## Our Solution
UniVerna utilizes a multi-stage, multi-vector architecture to combine the best of both worlds. By fusing Dense (semantic) and Sparse (lexical) retrieval, applying deep Cross-Encoder reranking, and ensembling the results.

## System Architecture
<img width="4054" height="1650" alt="image" src="https://github.com/user-attachments/assets/99b5a7ed-4bfd-4eab-830d-93b4e069c18e" />

## Team Members
* **Maram Ruthvi:** Dataset building, Web Scraping, & Corpus Curation.
* **Abhishek Rana:** Evaluation Pipeline, Retrieval Algorithms, & Ensembling Logic.
* **Vaibhav Helambe:** Application Layer, LLM Integration, & Telegram Bot Service.

## Configuration, Installation and Operating instructions
### Data


### Evaluation
1. Go to Lightning AI website (https://lightning.ai/)
2. Create a new studio in a workspace
3. Choose L4 environment
4. Upload the jupyter notebook present in the Evaluation folder
5. Run all the cells of the jupyter notebook to get all the evaluation results

### Application


## A file manifest (a list of files in the directory or archive)


## Copyright and Licensing Information
This project is licensed under the MIT License.
You are free to use, modify, and distribute this project with proper attribution.
See the LICENSE file for the full license text.

## Contact information for the distributor or author
Natural Language Processing: Team 8 <br>
Name: Abhishek Rana, GitHub Username: abskrana <br>
Name: Vaibhav Helambe, GitHub Username: Helambe-vaibhav <br>
Name: Maram Ruthvi, GitHub Username: Ruthvi5

## Credits and acknowledgments
Natural Language Processing: Team 8 <br>
Name: Abhishek Rana, Email Address: abhishekrana21092003@gmail.com <br>
Name: Vaibhav Helambe, Email Address: helambevaibhav2001@gmail.com <br>
Name: Maram Ruthvi, Email Address: 142301021@smail.iitpkd.ac.in


## Link to datasets and other relevant repositories
