import argparse
from sentence_transformers import SentenceTransformer
import chromadb


def pretty_print_results(query, results):
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0] if "distances" in results else [None] * len(ids)

    if not ids:
        print("No results found.")
        return

    for rank, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        section_title = meta.get("section_title", "Unknown") if isinstance(meta, dict) else "Unknown"
        section = meta.get("section", "Unknown") if isinstance(meta, dict) else "Unknown"
        chunk_index = meta.get("chunk_index", "N/A") if isinstance(meta, dict) else "N/A"
        paper_title = meta.get("paper_title", "") if isinstance(meta, dict) else ""
        paper_author = meta.get("paper_author", "") if isinstance(meta, dict) else ""

        similarity = None
        if dist is not None:
            # For cosine distance in [0, 2], convert to cosine similarity in [-1, 1]
            # cos_sim = 1 - dist / 2
            similarity = 1.0 - (dist / 2.0)
            # If you want a clean [0, 1] score, clamp:
            similarity = max(-1.0, min(1.0, similarity))

        print(f"\n--- Result #{rank} ---")
        print(f"ID: {doc_id}")
        print(f"Section: {section} | Title: {section_title} | Chunk: {chunk_index}")
        if paper_title:
            print(f"Paper: {paper_title}")
        if paper_author:
            print(f"Author: {paper_author}")

        if dist is not None:
            print(f"Distance: {dist:.4f}")
            print(f"Cosine similarity: {similarity:.4f}")

        snippet = doc.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        print(f"Text: {snippet}")



def main(args):
    # --- Load embedding model (same as used for indexing) ---
    print("[info] Loading sentence-transformers/all-MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # --- Connect to Chroma persistent DB ---
    print(f"[info] Connecting to ChromaDB at {args.persist_directory}")
    client = chromadb.PersistentClient(path=args.persist_directory)

    print(f"[info] Getting collection '{args.collection_name}'")
    collection = client.get_collection(args.collection_name)

    # --- Embed query ---
    query = args.query
    print(f"[info] Embedding query: {query!r}")
    query_emb = model.encode([query], normalize_embeddings=True).tolist()

    # --- Query Chroma ---
    print(f"[info] Retrieving top-{args.top_k} results...")
    results = collection.query(
        query_embeddings=query_emb,
        n_results=args.top_k,
        include=["documents", "metadatas", "distances"],
    )

    # --- Print results nicely ---
    pretty_print_results(query, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test retrieval from a ChromaDB collection built with MiniLM embeddings."
    )
    parser.add_argument(
        "query",
        type=str,
        help="Text query to search for.",
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="./chroma_deepseek_ocr",
        help="Directory where ChromaDB stores its data (same as used during embedding).",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="deepseek_ocr_chunks",
        help="Name of the Chroma collection (same as used during embedding).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to retrieve.",
    )

    args = parser.parse_args()
    main(args)