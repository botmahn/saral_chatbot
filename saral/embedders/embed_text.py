#!/usr/bin/env python3
import os
import argparse
import json
from typing import List, Tuple, Dict, Any

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import chromadb

# Keep your cache dirs (you can also move these into the class if you want)
os.environ['TORCH_HOME'] = "/Users/namanmishra/Documents/Code/saral_chatbot/pretrained_models"
os.environ['HF_HOME'] = "/Users/namanmishra/Documents/Code/saral_chatbot/pretrained_models"


class ChromaJSONIndexer:
    """
    Build a ChromaDB collection from a chunked JSON file using a SentenceTransformer model.

    Expects JSON of the form:
    {
        "title": "...",
        "author": "...",
        "abstract_chunks": [...],
        "content": [
            {
                "title": "...",
                "level": 1,
                "chunks": ["...", ...]
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_deepseek_ocr",
        collection_name: str = "deepseek_ocr_chunks",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        verbose: bool = True,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.verbose = verbose

        if self.verbose:
            print(f"[info] Loading embedding model: {self.model_name}")

        self.model = SentenceTransformer(self.model_name)

        if self.verbose:
            print(f"[info] Initializing ChromaDB at {self.persist_directory}")

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    # ---------------------- JSON → chunks ----------------------

    @staticmethod
    def load_chunks_from_json(json_file: str) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Load abstract and content chunks + metadata from a JSON file.
        Returns:
            documents: list of chunk strings
            metadatas: list of metadata dicts
            ids:       list of unique IDs for Chroma
        """
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        title = data.get("title", "")
        author = data.get("author", "")

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        # ---- Abstract chunks ----
        abstract_chunks = data.get("abstract_chunks", [])
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

        # ---- Content chunks ----
        content = data.get("content", [])
        for sec_idx, section in enumerate(content):
            section_title = section.get("title", f"section_{sec_idx}")
            level = section.get("level", None)
            sec_chunks = section.get("chunks", [])

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

    # ---------------------- Core indexing ----------------------

    def index_file(self, input_json_file: str):
        """
        Read one chunked JSON file, compute embeddings, and add to Chroma collection.
        """
        if self.verbose:
            print(f"[info] Loading chunks from {input_json_file}")

        documents, metadatas, ids = self.load_chunks_from_json(input_json_file)

        if not documents:
            raise RuntimeError(
                "No chunks found to embed. Did you run the chunking script first?"
            )

        if self.verbose:
            print(f"[info] Found {len(documents)} chunks to embed.")
            print("[info] Encoding chunks...")

        embeddings = []

        for i in tqdm(range(0, len(documents), self.batch_size), desc="Embedding Chunks"):
            batch = documents[i : i + self.batch_size]
            batch_embedding = self.model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
            )
            embeddings.extend(batch_embedding.tolist())

        if self.verbose:
            print(
                f"[info] Adding {len(documents)} chunks to collection '{self.collection_name}'..."
            )

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        if self.verbose:
            print("[info] Done. ChromaDB persisted.")

        # Optionally return summary
        return {
            "num_chunks": len(documents),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }

    def index_many(self, input_json_files: list[str]):
        """
        Index multiple JSON files into the same collection.
        """
        results = []
        for path in input_json_files:
            if self.verbose:
                print(f"\n[info] Processing file: {path}")
            res = self.index_file(path)
            results.append((path, res))
        return results


# ---------------------- CLI wrapper ----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create a ChromaDB collection from chunked JSON using MiniLM embeddings."
    )
    parser.add_argument(
        "input_json_file",
        type=str,
        help="Path to the chunked JSON file (with abstract_chunks and section['chunks']).",
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="./chroma_deepseek_ocr",
        help="Directory where ChromaDB stores its data.",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="deepseek_ocr_chunks",
        help="Name of the Chroma collection.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for MiniLM encoding.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logging.",
    )

    args = parser.parse_args()

    indexer = ChromaJSONIndexer(
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        verbose=not args.quiet,
    )

    indexer.index_file(args.input_json_file)


if __name__ == "__main__":
    main()
