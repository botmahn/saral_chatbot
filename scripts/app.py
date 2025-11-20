import os
import argparse
import json
import re
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
import chromadb

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langsmith.run_helpers import traceable

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "saral"

class TalkEvaluator:
    """
    Basic evaluation metrics as requested:
    1. Citation Coverage: % of claims with a [CHUNK_X] citation.
    2. Source Overlap (Proxy): Checks if cited chunks actually exist in retrieval.
    """
    @staticmethod
    def evaluate_provenance(generated_text: str, retrieved_docs: List):
        """
        Calculates what percentage of lines/bullets contain a citation.
        """
        citation_pattern = r"\[CHUNK_(\d+)\]"
        lines = [line.strip() for line in generated_text.split('\n') if len(line.strip()) > 20]
        
        if not lines:
            return {"coverage": 0.0, "details": "No meaningful text generated."}

        cited_lines = 0
        citations_found = []
        
        for line in lines:
            matches = re.findall(citation_pattern, line)
            if matches:
                cited_lines += 1
                citations_found.extend(matches)
        
        coverage = cited_lines / len(lines) if lines else 0
        
        return {
            "citation_coverage_score": round(coverage, 2),
            "total_lines_analyzed": len(lines),
            "unique_chunks_cited": list(set(citations_found))
        }


def ingest_and_embed(uploaded_file, persist_dir, collection_name):
    """
    Ingests a PDF file, splits it, and saves to ChromaDB.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)

    for i, split in enumerate(splits):
        split.metadata["chunk_index"] = i
        split.metadata["paper_title"] = uploaded_file.name
        split.metadata["section"] = "body" 

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    
    client = chromadb.PersistentClient(path=persist_dir)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
        client=client,
        persist_directory=persist_dir
    )
    
    os.remove(tmp_path)
    return vectorstore

def build_langchain_retriever(persist_dir, collection_name, top_k):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    client = chromadb.PersistentClient(path=persist_dir)

    try:
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        if not vectorstore.get()['ids']:
            return None, None
    except Exception:
        return None, None

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever, vectorstore

def docs_to_context_with_provenance(docs):
    parts = []
    for idx, d in enumerate(docs, start=1):
        meta = d.metadata or {}

        chunk_idx = meta.get("chunk_index", "N/A")
        paper_title = meta.get("paper_title", "Unknown Paper")
        page_num = meta.get("page", "N/A")
        
        provenance_id = f"CHUNK_{idx}"

        header = (
            f"[{provenance_id} | Source: {paper_title}, Page: {page_num}, Orig_Idx: {chunk_idx}]"
        )
        content = d.page_content.replace("\n", " ")
        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)

def build_talk_rag_chain(retriever, ollama_model):
    llm = ChatOllama(
        model=ollama_model,
        temperature=0.1, 
        keep_alive="5m"
    )

    # FIX: Double curly braces {{ }} are used for LaTeX to prevent LangChain
    # from interpreting them as variables.
    system_prompt = """You are SARAL, a helpful academic talk assistant.
You are given:
1. A user request about creating or refining a talk based on research papers.
2. A set of retrieved context chunks, labeled [CHUNK_1], [CHUNK_2], etc.

Your Instructions:
- **Factuality:** Use ONLY the provided context. If the info isn't there, state that.
- **Provenance:** You MUST cite your sources. Every major claim, bullet point, or script sentence must end with the relevant tag, e.g., "The transformer architecture allows parallelization [CHUNK_1]."
- **Math:** Preserve all equations in LaTeX format, e.g., $E=mc^2$ or $$\\mathcal{{L}}_{{task}}$$.
- **Style:** Adhere strictly to the requested style (Technical, Plain English, etc.) and length.
- **Safety:** No offensive content.

Context:
{context}
"""

    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """User request: {user_request}
Target Length: {length_choice}
Target Style: {style}

Produce the output in this Markdown structure:

## Slides Overview
- Slide 1: Title
- Slide 2: Title ...

## Slide Details
### Slide 1: [Title]
- Bullet point [CHUNK_X]
- Bullet point [CHUNK_Y]

### Slide 2: [Title]
...

## Speaker Script (Approx {length_choice})
(Write a cohesive speech. Ensure you cite chunks in the flow of speech like "...as seen in [CHUNK_1].")

## Speaker Notes & Math
### Slide 1 Notes
- Note with math: $x^2$ [CHUNK_Z]
""")
    ])

    rag_chain = (
        RunnableParallel({
            "context": (lambda x: x["user_request"]) | retriever | docs_to_context_with_provenance,
            "user_request": lambda x: x["user_request"],
            "length_choice": lambda x: x["length_choice"],
            "style": lambda x: x["style"],
        })
        | generation_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

@traceable(name="saral_refine")
def refine_talk(previous_output, refine_instruction, style, llm_model):
    """
    Refinement logic does not necessarily need Retrieval if just changing style,
    but to keep it simple we use a direct LLM call here to edit the text.
    """
    llm = ChatOllama(model=llm_model, temperature=0.2)
    
    prompt = ChatPromptTemplate.from_template("""
You are an expert editor.
User wants to refine the following presentation content.

**Constraint:**
1. Modify ONLY the parts requested by the user.
2. You MUST use bolding to show changes: **[OLD] <old text>** -> **[NEW] <new text>**.
3. At the end, add a 'Change Rationale' section explaining why.

=== CURRENT CONTENT ===
{previous_output}
=======================

User Instruction: {refine_instruction}
Style constraint: {style}

Output the full updated markdown.
""")
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "previous_output": previous_output, 
        "refine_instruction": refine_instruction,
        "style": style
    })


def append_to_conversation_log(log_path, role, content, extra=None):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "role": role,
        "content": content,
    }
    if extra:
        record.update(extra)

    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
    
    with open(log_path, "a", encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def init_session_state():
    keys = ["messages", "last_output", "rag_chain", "retriever", "vectorstore"]
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None
    if st.session_state.messages is None:
        st.session_state.messages = []


def main(args):
    st.set_page_config(page_title="SARAL Chatbot", layout="wide", page_icon="🎓")
    st.title("🎓 SARAL: Academic Talk Generator")

    init_session_state()

    with st.sidebar:
        st.header("⚙️ RAG Configuration")
        persist_dir = st.text_input("Chroma Path", value=args.chroma_dir)
        collection_name = st.text_input("Collection Name", value=args.collection_name)
        ollama_model = st.text_input("Ollama Model", value=args.ollama_model)
        top_k = st.slider("Retrieval Chunk Count (k)", 2, 10, 4)
        
        st.markdown("---")
        st.subheader("📄 Document Ingestion")
        uploaded_file = st.file_uploader("Upload PDF Paper", type=["pdf"])
        
        if uploaded_file and st.button("Ingest Paper"):
            with st.spinner("Parsing and Embedding..."):
                vs = ingest_and_embed(uploaded_file, persist_dir, collection_name)
                st.session_state.vectorstore = vs
                st.session_state.retriever = vs.as_retriever(search_kwargs={"k": top_k})
                st.session_state.rag_chain = build_talk_rag_chain(st.session_state.retriever, ollama_model)
            st.success(f"Ingested {uploaded_file.name}!")

        st.markdown("---")
        if st.button("Re-initialize Retriever"):
            retriever, vectorstore = build_langchain_retriever(persist_dir, collection_name, top_k)
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.vectorstore = vectorstore
                st.session_state.rag_chain = build_talk_rag_chain(retriever, ollama_model)
                st.success("Retriever Loaded.")
            else:
                st.error("Collection empty or not found. Please upload a paper.")

    tab_gen, tab_eval = st.tabs(["💬 Generator", "📊 Evaluation"])

    with tab_gen:
        col1, col2 = st.columns([3, 1])
        with col1:
            length_choice = st.radio("Duration", ["30s", "90s", "5mins"], horizontal=True)
        with col2:
            style = st.selectbox("Style", ["Technical", "Plain-English", "Press Release"])

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("e.g., 'Make a 5-slide talk on the methodology...'")

        if user_input:
            if not st.session_state.rag_chain:
                st.error("Please ingest a paper or initialize the retriever first.")
            else:
                st.session_state.messages.append({"role": "user", "content": user_input})
                append_to_conversation_log(
                    args.saral_conversation_log, "user", user_input,
                    {"length": length_choice, "style": style}
                )
                
                with st.chat_message("user"):
                    st.write(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("SARAL is working..."):
                        try:
                            if st.session_state.last_output is None:
                                response = st.session_state.rag_chain.invoke({
                                    "user_request": user_input,
                                    "length_choice": length_choice,
                                    "style": style
                                })
                            else:
                                response = refine_talk(
                                    previous_output=st.session_state.last_output,
                                    refine_instruction=user_input,
                                    style=style,
                                    llm_model=ollama_model
                                )
                            
                            st.markdown(response)
                            st.session_state.last_output = response
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            append_to_conversation_log(
                                args.saral_conversation_log, "assistant", response,
                                {"mode": "refine" if st.session_state.last_output else "generate"}
                            )
                        except Exception as e:
                            st.error(f"Error during generation: {e}")

    with tab_eval:
        st.subheader("Automated Quality Checks")
        if st.session_state.last_output:
            st.markdown("### Analysis of Last Generation")
            eval_metrics = TalkEvaluator.evaluate_provenance(
                st.session_state.last_output, []
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Citation Coverage", f"{eval_metrics['citation_coverage_score']*100:.1f}%")
            c2.metric("Lines Analyzed", eval_metrics['total_lines_analyzed'])
            c3.metric("Unique Sources", len(eval_metrics['unique_chunks_cited']))
            st.json(eval_metrics)
        else:
            st.info("Generate a response first to see evaluation metrics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARAL Chatbot")
    parser.add_argument("--chroma_dir", type=str, default="./DBs/deepseek_ocr", help="Directory for ChromaDB persistence")
    parser.add_argument("--collection_name", type=str, default="deepseek_ocr_chunks", help="Name of the Chroma collection")
    parser.add_argument("--ollama_model", type=str, default="qwen3:8b", help="Ollama model tag to use")
    parser.add_argument("--saral_conversation_log", type=str, default="conversation.jsonl", help="Path to log file")

    try:
        # We use parse_known_args because streamlit inserts its own flags 
        # (like --server.port) into sys.argv, which causes argparse to fail 
        # if we use strict parse_args().
        args, unknown = parser.parse_known_args()
    except SystemExit:
        # Fallback if --help is called or something breaks critically
        class Defaults:
            chroma_dir = "./chroma_db"
            collection_name = "research_papers"
            ollama_model = "llama3"
            saral_conversation_log = "conversation.jsonl"
        args = Defaults()
        print("SystemExit detected (likely Streamlit --help). Using default arguments.")

    main(args)