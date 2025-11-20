import os
import argparse
import json
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings

os.environ['TORCH_HOME'] = "/Users/namanmishra/Documents/Code/saral_chatbot/pretrained_models"
os.environ['HF_HOME'] = "/Users/namanmishra/Documents/Code/saral_chatbot/pretrained_models"

def load_chunks_from_json(json_file):

    data = json.load(open(json_file, 'r'))

    title = data.get('title', '')
    author = data.get('author', '')

    documents = []
    metadatas = []
    ids = []

    abstract_chunks = data.get('abstract_chunks', [])
    for i, chunk in enumerate(abstract_chunks):
        if not isinstance(chunk, str) or not chunk.strip():
            continue
        doc_id = f"abstract_{i}"
        documents.append(chunk)
        metadatas.append(
            {
                "section": "abstract",
                "section_title": "Abstract",
                "chunk_index": i,
                "paper_title": title,
                "paper_author": author,
            }
        )
        ids.append(doc_id)

    content = data.get('content', [])
    for sec_idx, section in enumerate(content):
        section_title = section.get('title', f"section_{sec_idx}")
        level = section.get('level', None)
        sec_chunks = section.get('chunks', [])

        for ch_idx, chunk in enumerate(sec_chunks):
            if not isinstance(chunk, str) or not chunk.strip():
                continue
            doc_id = f"sec_{sec_idx}_chunk_{ch_idx}"
            documents.append(chunk)
            metadatas.append(
                {
                    "section": "content",
                    "section_title": section_title,
                    "section_level": level,
                    "section_index": sec_idx,
                    "chunk_index": ch_idx,
                    "paper_title": title,
                    "paper_author": author,
                }
            )
            ids.append(doc_id)

    return documents, metadatas, ids


def main(args):

    print(f"[info] Loading chunks from {args.input_json_file}")
    documents, metadatas, ids = load_chunks_from_json(args.input_json_file)

    if not documents:
        raise RuntimeError("No chunks found to embed. Did you run the chunking script first?")
    print(f"[info] Found {len(documents)} chunks to embed.")

    print("[info] Loading sentence-transformers/all-MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("[info] Encoding chunks...")
    embeddings = []

    for i in tqdm(range(0, len(documents), args.batch_size), desc="Embedding Chunks"):
        batch = documents[i : i + args.batch_size]
        batch_embedding = model.encode(
            batch,
            batch_size=args.batch_size,
            normalize_embeddings=True,
        )
        embeddings.extend(batch_embedding.tolist())

    print(f"[info] Initializing ChromaDB at {args.persist_directory}")
    client = chromadb.PersistentClient(path=args.persist_directory)
    collection = client.get_or_create_collection(name=args.collection_name)

    print(f"[info] Adding {len(documents)} chunks to collection '{args.collection_name}'...")

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("[info] Done. ChromaDB persisted.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ChromaDB collection from chunked JSON using MiniLM embeddings.")
    parser.add_argument("input_json_file", type=str, help="Path to the chunked JSON file (with abstract_chunks and section['chunks']).")
    parser.add_argument("--persist_directory", type=str, default="./chroma_deepseek_ocr", help="Directory where ChromaDB stores its data.")
    parser.add_argument("--collection_name", type=str, default="deepseek_ocr_chunks", help="Name of the Chroma collection.",)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for MiniLM encoding.",)
    args = parser.parse_args()
    main(args)