# SARAL RAG-based Chatbot

This is the code used for the SARAL x RAG Chatbot streamlit application. 

---

## Installation
```
#Make sure you have anaconda installed.

git clone https://github.com/botmahn/saral_chatbot.git
cd saral_chatbot
conda env create -f environment.yml
```
---

## Running the application
```streamlit run app.py```

---

## Features

- PDF and LaTeX support.
- Automated chunking and embedding into ChromaDB.
- Ollama backend.
- MMR retrieval.
- LangChain Orchestration.
- LangSmith Logging.
- Provenance of chunks and section associations.
- Chat History Retention.
- Evaluation based on coverage, number of chunks, number of sentences, etc.

---

## Workflow

1. Run the app.
2. Upload the main.tex file of a paper (I used the DeepSeek-OCR paper) in the sidebar.
3. Set the DB paths.
4. The backend is ollama. Set the ollama model in the sidebar.
5. Enter the query (more details --> better results).
6. Choose the speaker script duration in the buttons above the chat input placeholder.
7. Run.
8. Once the generation starts, we can observe the retrieved chunks and their parent sections.
9. After the initial generation is over, we can chat with the model and choose to refine the slides.
10. Observe the coverage and evaluation in the "evaluation" tab. 
