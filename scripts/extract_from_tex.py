import re
import os
import json
import argparse
import sys

def clean_text_block(text):
    """
    Applies a series of regex substitutions to clean a block of LaTeX text.
    This function is applied *after* the document is sectioned.
    """
    if not text:
        return ""

    # Remove comments (already done in main parser, but good for safety)
    # Use negative lookbehind to avoid matching escaped \%
    text = re.sub(r'(?<!\\)%.*\n', '\n', text)

    # Remove non-greedy environments: equation, align, gather, verbatim, lstlisting, etc.
    # We already removed figure and table in the main parser.
    text = re.sub(
        r'\\begin\{(equation|align|gather|verbatim|lstlisting)\*?\}[\s\S]*?\\end\{\1\*?\}',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )
    # Also remove figure/table environments (which were extracted *before* this)
    text = re.sub(
        r'\\begin\{(figure|table|tabular)\*?\}[\s\S]*?\\end\{\1\*?\}',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove math environments (inline and display)
    text = re.sub(r'\$.*?\$', '', text, flags=re.DOTALL)  # Non-greedy inline
    text = re.sub(r'\\\([\s\S]*?\\\)', '', text, flags=re.DOTALL)  # Non-greedy \( ... \)
    text = re.sub(r'\\\[[\s\S]*?\\\]', '', text, flags=re.DOTALL)  # Non-greedy \[ ... \]

    # Handle formatting commands (keep the text inside)
    # e.g., \textbf{Hello} -> Hello
    text = re.sub(r'\\(textbf|textit|emph|texttt|textsc|sc|bf|it)\{([^}]+)\}', r'\2', text, flags=re.IGNORECASE)

    # Remove commands with one argument that we want to discard
    # e.g., \cite{...}, \ref{...}, \label{...}, \url{...}, \footnote{...}
    # Also handle non-breaking space ~ prefix
    text = re.sub(r'~?\\(cite|citep|citet|ref|label|url|footnote|thanks)\[[^\]]*?\]\{[^}]+\}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'~?\\(cite|citep|citet|ref|label|url|footnote|thanks)\{([^}]+)\}', '', text, flags=re.IGNORECASE)

    # Remove simple commands (no arguments or with simple options)
    # e.g., \item, \par, \maketitle, \small, \large, \etc, \ie
    text = re.sub(r'\\[a-zA-Z]+(\*|\[[^\]]*?\])?', ' ', text)

    # Convert escaped LaTeX special characters to their symbol
    text = re.sub(r'\\%', '%', text)

    # Remove forced line breaks
    text = re.sub(r'\\\\', ' ', text)

    # Remove remaining curly braces
    text = re.sub(r'[\{\}]', '', text)

    # Clean up whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Consolidate spaces

    # Join lines that are not paragraph breaks (i.e., not \n\n)
    # This turns hard-wrapped text into single paragraphs.
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Consolidate horizontal spaces again, in case joining lines created new ones
    text = re.sub(r'[ ]+', ' ', text)

    # Normalize paragraph breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Consolidate paragraphs
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Remove leading whitespace on lines
    text = re.sub(r'\s+$', '', text, flags=re.MULTILINE) # Remove trailing whitespace on lines
    text = re.sub(r'\n\n+', '\n\n', text)  # Collapse multiple blank lines

    return text.strip()

def resolve_inputs(tex_content, base_dir):
    """
    Recursively finds and replaces \input and \include commands
    with the content of the referenced .tex file.
    """
    input_pattern = re.compile(r'\\(input|include)\{([^}]+)\}')
    
    def replacer(match):
        filename = match.group(2)
        if not filename.endswith('.tex'):
            filename += '.tex'
        
        filepath = os.path.join(base_dir, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Recursively resolve inputs in the new content
                return resolve_inputs(content, os.path.dirname(filepath))
            except Exception as e:
                print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
                return ''
        else:
            print(f"Warning: File not found for input: {filepath}", file=sys.stderr)
            return ''

    return input_pattern.sub(replacer, tex_content)

def extract_metadata(full_content):
    """
    Extracts image, equation, and table metadata *before* cleaning.
    """
    image_data = []
    equation_data = []
    table_data = []

    # Common patterns
    caption_pattern = re.compile(r'\\caption\{([\s\S]*?)\}', re.DOTALL)
    label_pattern = re.compile(r'\\label\{([^}]+)\}')

    # 1. Extract Figures
    # This finds \begin{figure} ... \end{figure} blocks
    figure_pattern = re.compile(r'\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}', re.DOTALL | re.IGNORECASE)
    # Inside a figure block, find includegraphics
    graphicx_pattern = re.compile(r'\\includegraphics(?:\[[^\]]*?\])?\{([^}]+)\}')

    for fig_block in figure_pattern.finditer(full_content):
        block_text = fig_block.group(0)
        
        image_meta = {
            "filename": None,
            "caption": None,
            "label": None,
            "raw_latex": block_text
        }
        
        img_match = graphicx_pattern.search(block_text)
        if img_match:
            image_meta["filename"] = img_match.group(1)
            
        cap_match = caption_pattern.search(block_text)
        if cap_match:
            # Simple clean of caption (remove nested commands, etc.)
            image_meta["caption"] = clean_text_block(cap_match.group(1))
            
        lbl_match = label_pattern.search(block_text)
        if lbl_match:
            image_meta["label"] = lbl_match.group(1)
            
        image_data.append(image_meta)

    # 2. Extract Equations
    # This finds various equation environments
    equation_pattern = re.compile(r'\\begin\{(equation|align|gather|multline)\*?\}[\s\S]*?\\end\{\1\*?\}', re.DOTALL | re.IGNORECASE)
    
    for eq_block in equation_pattern.finditer(full_content):
        block_text = eq_block.group(0)
        env_type = eq_block.group(1)
        
        equation_meta = {
            "type": env_type,
            "label": None,
            "raw_latex": block_text
        }
        
        lbl_match = label_pattern.search(block_text)
        if lbl_match:
            equation_meta["label"] = lbl_match.group(1)
            
        equation_data.append(equation_meta)

    # 3. Extract Tables
    # This finds \begin{table} ... \end{table} blocks
    table_pattern = re.compile(r'\\begin\{(table|table\*)\}[\s\S]*?\\end\{\1\}', re.DOTALL | re.IGNORECASE)
    # Inside a table block, find the tabular content
    tabular_pattern = re.compile(r'\\begin\{tabular\*?\}[\s\S]*?\\end\{tabular\*?\}', re.DOTALL | re.IGNORECASE)

    for tbl_block in table_pattern.finditer(full_content):
        block_text = tbl_block.group(0)
        
        table_meta = {
            "caption": None,
            "label": None,
            "raw_tabular_content": None,
            "raw_latex": block_text
        }
            
        cap_match = caption_pattern.search(block_text)
        if cap_match:
            table_meta["caption"] = clean_text_block(cap_match.group(1))
            
        lbl_match = label_pattern.search(block_text)
        if lbl_match:
            table_meta["label"] = lbl_match.group(1)
        
        tab_match = tabular_pattern.search(block_text)
        if tab_match:
            table_meta["raw_tabular_content"] = tab_match.group(0)
            
        table_data.append(table_meta)

    return image_data, equation_data, table_data


def parse_tex_to_json(main_tex_path):
    """
    Main function to parse the LaTeX project.
    Returns four separate data structures: text, images, equations, and tables.
    """
    base_dir = os.path.dirname(os.path.abspath(main_tex_path))
    
    try:
        with open(main_tex_path, 'r', encoding='utf-8') as f:
            initial_content = f.read()
    except FileNotFoundError:
        print(f"Error: Main file not found: {main_tex_path}", file=sys.stderr)
        return None, None, None, None
    except Exception as e:
        print(f"Error reading main file: {e}", file=sys.stderr)
        return None, None, None, None

    # 1. Resolve all \input commands to get the full document
    full_content = resolve_inputs(initial_content, base_dir)

    # 2. Extract metadata (images, equations, tables) BEFORE cleaning
    image_data, equation_data, table_data = extract_metadata(full_content)

    # 3. Pre-cleaning for text extraction: Remove comments
    # Use negative lookbehind to avoid matching escaped \%
    full_content_cleaned = re.sub(r'(?<!\\)%.*\n', '\n', full_content)
    
    # Note: Figure/Table/Equation environments are removed *inside* clean_text_block
    # This is fine since we've already extracted them.

    data = {
        "title": "",
        "author": "",
        "abstract": "",
        "content": []
    }

    # 4. Extract text metadata (Title, Author, Abstract)
    try:
        title_match = re.search(r'\\title\{([\s\S]*?)\}', full_content_cleaned, re.DOTALL)
        if title_match:
            data['title'] = clean_text_block(title_match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse title. {e}", file=sys.stderr)

    try:
        author_match = re.search(r'\\author\{([\s\S]*?)\}', full_content_cleaned, re.DOTALL)
        if author_match:
            # \and is common in author blocks
            author_text = author_match.group(1).replace('\\and', ' and ')
            data['author'] = clean_text_block(author_text)
    except Exception as e:
        print(f"Warning: Could not parse author. {e}", file=sys.stderr)

    try:
        abstract_match = re.search(r'\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}', full_content_cleaned, re.DOTALL | re.IGNORECASE)
        if abstract_match:
            data['abstract'] = clean_text_block(abstract_match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse abstract. {e}", file=sys.stderr)

    # 5. Extract main document content
    main_content_match = re.search(r'\\begin\{document\}([\s\S]*?)\\end\{document\}', full_content_cleaned, re.DOTALL | re.IGNORECASE)
    if not main_content_match:
        print("Error: Could not find \\begin{document}...\\end{document} block.", file=sys.stderr)
        return data, image_data, equation_data, table_data  # Return data collected so far

    main_content = main_content_match.group(1)

    # 6. Split text content by sectioning commands
    # We use re.split() which keeps the delimiters
    section_pattern = re.compile(r'\\(section|subsection|subsubsection)\*?\{([^}]+)\}', re.IGNORECASE)
    
    # `parts` will be [text_before_first_section, 'section', 'Title 1', 'Text 1', 'subsection', 'Title 1.1', 'Text 1.1', ...]
    parts = section_pattern.split(main_content)
    
    # The first part is text before any section (e.g., \maketitle, intro paragraph)
    intro_text = clean_text_block(parts[0])
    if intro_text:
        data['content'].append({
            "level": 0,
            "title": "Introduction (Preamble)",
            "text": intro_text
        })

    # Iterate over the rest of the parts, which come in groups of 3
    # (section_type, title, text)
    for i in range(1, len(parts), 3):
        try:
            section_type = parts[i].lower()
            title = parts[i+1].strip()
            text = parts[i+2]
            
            level = 0
            if section_type == 'section':
                level = 1
            elif section_type == 'subsection':
                level = 2
            elif section_type == 'subsubsection':
                level = 3

            data['content'].append({
                "level": level,
                "title": title,
                "text": clean_text_block(text)
            })
        except IndexError:
            # This can happen if the document ends right after a section command
            continue
            
    return data, image_data, equation_data, table_data

def main():
    parser = argparse.ArgumentParser(description='Convert an arXiv LaTeX project into clean JSON files.')
    parser.add_argument('main_tex_file', type=str, help='Path to the main .tex file of the project.')
    parser.add_argument('output_json_file', type=str, help='Path to save the main text output JSON file. Other files (images, equations) will be derived from this name.')
    
    args = parser.parse_args()
    
    # Derive output filenames
    base_output_path = args.output_json_file
    if base_output_path.endswith('.json'):
        base_output_path = base_output_path[:-5]
        
    text_output_file = base_output_path + '.json'
    image_output_file = base_output_path + '_images.json'
    equation_output_file = base_output_path + '_equations.json'
    table_output_file = base_output_path + '_tables.json'
    
    print(f"Starting parsing of {args.main_tex_file}...")
    text_data, image_data, equation_data, table_data = parse_tex_to_json(args.main_tex_file)
    
    # Write text data
    if text_data:
        try:
            with open(text_output_file, 'w', encoding='utf-8') as f:
                json.dump(text_data, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved text data to {text_output_file}")
        except Exception as e:
            print(f"Error writing text JSON file: {e}", file=sys.stderr)

    # Write image data
    if image_data:
        try:
            with open(image_output_file, 'w', encoding='utf-8') as f:
                json.dump(image_data, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved image data to {image_output_file}")
        except Exception as e:
            print(f"Error writing image JSON file: {e}", file=sys.stderr)

    # Write equation data
    if equation_data:
        try:
            with open(equation_output_file, 'w', encoding='utf-8') as f:
                json.dump(equation_data, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved equation data to {equation_output_file}")
        except Exception as e:
            print(f"Error writing equation JSON file: {e}", file=sys.stderr)
            
    # Write table data
    if table_data:
        try:
            with open(table_output_file, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved table data to {table_output_file}")
        except Exception as e:
            print(f"Error writing table JSON file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()