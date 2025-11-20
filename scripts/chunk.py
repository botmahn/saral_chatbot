import json
import argparse

import tiktoken
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text_with_splitter(text, splitter):
    if not text:
        return []
    splits = splitter.split_text(text)
    return splits


def main(args):

    tokenizer = tiktoken.get_encoding(args.encoding)

    def token_length(text):
        return len(tokenizer.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=token_length,
    )

    with open(args.input_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Starting chunking with chunk_size={args.chunk_size} and overlap={args.chunk_overlap}...")

    abstract_text = data.get("abstract")
    if isinstance(abstract_text, str) and abstract_text.strip():
        data["abstract_chunks"] = chunk_text_with_splitter(abstract_text, text_splitter)
        print(f"Chunked 'abstract' into {len(data['abstract_chunks'])} chunks.")
    else:
        data["abstract_chunks"] = []
        print("No valid abstract text found; 'abstract_chunks' set to empty list.")

    content_list = data.get("content")
    if content_list and isinstance(content_list, list):
        for i, section in enumerate(
            tqdm(content_list, desc="Chunking content sections"),
            start=1,
        ):
            section_text = section.get("text")
            if isinstance(section_text, str) and section_text.strip():
                section["chunks"] = chunk_text_with_splitter(section_text, text_splitter)
            else:
                section["chunks"] = []
        print(f"Finished chunking {len(content_list)} content sections.")
    else:
        print("No 'content' list found; skipping section chunking.")

    with open(args.output_json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved chunked JSON to {args.output_json_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk text in a JSON file using a RecursiveCharacterTextSplitter.")
    parser.add_argument("input_json_file", type=str, help="Path to the input .json file.")
    parser.add_argument("output_json_file", type=str, help="Path to save the output .json file with chunks.",)
    parser.add_argument("-c", "--chunk_size", type=int, default=256, help="The target number of tokens for each chunk (default: 256).",)
    parser.add_argument("-o", "--chunk_overlap", type=int, default=64, help="The number of tokens to overlap between chunks (default: 64).",)
    parser.add_argument("-e", "--encoding", type=str, default="cl100k_base", help="The tiktoken encoding to use (default: cl100k_base, used by gpt-4).",)
    args = parser.parse_args()
    main(args)
