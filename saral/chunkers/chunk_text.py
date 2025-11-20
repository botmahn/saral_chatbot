#!/usr/bin/env python3
import json
import argparse

import tiktoken
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter


class JSONTextChunker:
    """
    Class to chunk abstract + section text from a LaTeX-parsed JSON file
    using a RecursiveCharacterTextSplitter with tiktoken length function.

    Usage:
        chunker = JSONTextChunker(chunk_size=256, chunk_overlap=64)
        chunker.process("paper.json", "paper_chunked.json")
    """

    def __init__(self, chunk_size=256, chunk_overlap=64, encoding="cl100k_base", verbose=True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = encoding
        self.verbose = verbose

        # tiktoken tokenizer
        self.tokenizer = tiktoken.get_encoding(encoding)

        # token length function for the splitter
        def token_length(text):
            return len(self.tokenizer.encode(text))

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=token_length,
        )

    # ------------------------------
    # Utility
    # ------------------------------

    def _chunk_text(self, text):
        """Return text chunks using the configured splitter."""
        if not text or not isinstance(text, str):
            return []
        return self.text_splitter.split_text(text)

    # ------------------------------
    # Core Processing
    # ------------------------------

    def process(self, input_json_file, output_json_file):
        """Load JSON, chunk text fields, and save new JSON."""

        if self.verbose:
            print(f"[INFO] Loading {input_json_file}")

        with open(input_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if self.verbose:
            print(f"[INFO] Chunking with chunk_size={self.chunk_size}, "
                  f"overlap={self.chunk_overlap}, encoding={self.encoding}")

        # ----------- Chunk abstract -----------
        abstract_text = data.get("abstract")
        if isinstance(abstract_text, str) and abstract_text.strip():
            data["abstract_chunks"] = self._chunk_text(abstract_text)
            if self.verbose:
                print(f"Chunked 'abstract' → {len(data['abstract_chunks'])} chunks")
        else:
            data["abstract_chunks"] = []
            if self.verbose:
                print("No valid abstract found; skipping.")

        # ----------- Chunk each content section -----------
        content_list = data.get("content")

        if content_list and isinstance(content_list, list):
            if self.verbose:
                print("Chunking content sections...")

            for section in tqdm(content_list, desc="Chunking sections"):
                section_text = section.get("text")
                section["chunks"] = self._chunk_text(section_text)

            if self.verbose:
                print(f"Finished chunking {len(content_list)} sections.")
        else:
            if self.verbose:
                print("No 'content' list found; skipping section chunking.")

        # ----------- Save output JSON -----------
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        if self.verbose:
            print(f"[✓] Saved chunked output → {output_json_file}")

        return data  # Optional return value


# -------------------------------------------------
# Optional CLI wrapper
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chunk text in a JSON file using a RecursiveCharacterTextSplitter."
    )
    parser.add_argument("input_json_file", type=str, help="Path to the input .json file.")
    parser.add_argument("output_json_file", type=str, help="Path to save the chunked .json output.")
    parser.add_argument("-c", "--chunk_size", type=int, default=256)
    parser.add_argument("-o", "--chunk_overlap", type=int, default=64)
    parser.add_argument("-e", "--encoding", type=str, default="cl100k_base")
    parser.add_argument("--quiet", action="store_true", help="Disable print logs")

    args = parser.parse_args()

    chunker = JSONTextChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        encoding=args.encoding,
        verbose=not args.quiet,
    )

    chunker.process(args.input_json_file, args.output_json_file)


if __name__ == "__main__":
    main()
