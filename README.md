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
1. Clone the Repository
2. Install Required Libraries
3. Run Web Scraper file using "python myscheme_scraper.py" - Folder Scheme_Data/ will be created which Contains scraped scheme documents
4. Convert raw text files into structured dataset by running "python txt_parser.py" - final dataset gov_corpus.json is created.

### Evaluation
1. Go to Lightning AI website (https://lightning.ai/)
2. Create a new studio in a workspace
3. Choose L4 environment
4. Upload the jupyter notebook present in the Evaluation folder as well as the gov_corpus.json present in the Data folder
5. Run all the cells of the jupyter notebook to get all the evaluation results

### Application

The application is a **Telegram RAG Bot** consisting of three FastAPI services (Bot · LLM · RAG) running on `localhost:8000–8002`.

> 📖 **For full details** (architecture, config variables, database schema, deployment notes), see [`Application/README.md`](Application/README.md).

**Prerequisites**
- Python 3.11, `uv` package manager, `bash` (Linux / WSL / Lightning AI CloudSpace)
- A **Telegram Bot Token** ([@BotFather](https://t.me/botfather))
- A **Google Gemini API Key** (for the default `gemini` LLM backend)

**Quick Configuration**
Set the two mandatory environment variables before starting:
```bash
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export GEMINI_API_KEY="your_gemini_api_key"
```
All other settings (ports, timeouts, rate limits, etc.) have sensible defaults in `config.py`.

**Installation & Startup**
```bash
cd Application
bash start.sh          # creates .venv, installs deps, starts all 3 services
```
`start.sh` automatically registers the Telegram webhook when `WEBHOOK_HOST` or `LIGHTNING_CLOUDSPACE_HOST` is set. If not auto-registered, run:
```bash
curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
     -d "url=https://<your-public-host>/webhook"
```

**Shutdown**
```bash
bash stop.sh           # gracefully stops all 3 services
```


## A file manifest (a list of files in the directory or archive)

### Root
| File | Description |
|---|---|
| `README.md` | This file — project overview, setup instructions, and manifest |
| `LICENSE` | MIT License text |

### Application/
| File | Description |
|---|---|
| `README.md` | Full technical documentation for the Telegram RAG Bot (architecture, config, DB schema, deployment) |
| `main.py` | FastAPI webhook server (port 8000) — handles Telegram updates, onboarding, rate limiting, and the RAG pipeline |
| `llm_server.py` | LLM inference server (port 8001) — supports Google Gemini and HuggingFace backends |
| `rag_server.py` | RAG retrieval server (port 8002) — loads corpus, encodes with LaBSE + BGE-M3, serves 7-method MRR ensemble search |
| `database.py` | Async SQLite layer (aiosqlite) — manages `users`, `user_profiles`, `messages`, and `query_cache` tables |
| `config.py` | Central configuration — all settings read from environment variables with defaults |
| `gov_corpus.json` | Hierarchical knowledge base of Indian government schemes used by the RAG server |
| `requirements.txt` | Pinned Python dependencies for all three services |
| `start.sh` | Bash startup script — creates virtual environment, installs dependencies, starts all services, registers Telegram webhook |
| `stop.sh` | Bash shutdown script — gracefully stops all services and cleans up PID file |

### Data/
| File | Description |
|---|---|
| `gov_corpus.json` | Raw government scheme corpus (source data before processing into Application format) |
| `myscheme_scraper.py` | Web scraper that collects government scheme data from myscheme.gov.in |
| `txt_parser.py` | Parser that converts raw scraped text files into structured JSON corpus format |
| `telugu_english_queries.xlsx` | Bilingual (Telugu–English) query dataset used for evaluation |

### Evaluation/
| File | Description |
|---|---|
| `NLP_Project_Evaluation.ipynb` | Jupyter notebook — runs the full evaluation pipeline and generates retrieval metrics (MRR, NDCG, precision, recall) on Lightning AI L4 |


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
MIRACL Multilingual Dataset - https://huggingface.co/datasets/miracl/miracl <br>
MyScheme (Scraped website) - https://www.myscheme.gov.in/
