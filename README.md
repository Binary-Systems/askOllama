# askOllama(1)

## NAME
<p><strong>askOllama</strong>  -  Command‐line interface for batch querying Ollama models with RAG support</p>
       
## SYNOPSIS

       askOllama [-o outfile] [-p personafile] [--rag RAGDIR] prompt [arg]

## DESCRIPTION

<p><strong>askOllama</strong> is a command‐line tool for sending prompts and arguments to Ollama models and retrieving responses. It supports batch queries, persona files, Retrieval‐Augmented Generation (RAG) directories, and multiple output formats.</p>

<p><strong>askOllama</strong> supports Retrieval-Augmented Generation (RAG) by indexing documents in a specified directory or a
single file. The vector database used for RAG is isolated in a hidden subdirectory `.rag_db` inside the
RAG path (or the file's parent directory), and that directory is excluded from the corpus so it does
not interfere with which files are indexed.</p>

<p>Prompts and arguments can be read from files: if the positional <em>prompt</em> or <em>arg</em> looks like a
file path, it will be parsed. Multiple prompts and multiple arguments are combined in a Cartesian
product (N×M) and executed, with results written in order.</p>

## OPTIONS
       -o outfile
              Specify an output file for results. Supported  formats:  .jsonl,
              .csv,  .rtf,  or  plain text. If omitted, results are printed to
              stdout.

       -p personafile
              Specify a file containing a persona/system prompt to set for the
              model.

       --rag RAGDIR
              Directory (or a single file) to build a RAG index from. The index is stored in `.rag_db` under the
              RAG directory. The `.rag_db` directory is ignored while crawling.

       prompt A prompt string or a file containing prompts (one per line).

       arg    An  optional  argument  string or file containing arguments (one
              per line).

       ‐h     Show usage/options summary.

       ‐‐help  Show this help/man page.

       --verbose  Enable verbose logging (DEBUG level). Default logging is INFO.


## SUPPORTED FILETYPES

       RAG context files:
              PDF (.pdf), text (.txt, .log, .py, .json), HTML  (.html,  .htm),
              Markdown  (.md, .markdown), RTF (.rtf), Word (.doc, .docx), Pow‐
              erPoint (.ppt, .pptx), Excel (.xls, .xlsx)

       Prompt files:
              Text (.txt), JSONL (.jsonl), CSV (.csv), RTF (.rtf)

       Argfiles:
              Text (.txt), JSONL (.jsonl), CSV (.csv), RTF (.rtf)

       Outfiles:
              JSONL (.jsonl), CSV (.csv), RTF (.rtf), plain text (.txt)


## EXAMPLES
<p>Query with a single prompt and print the result to stdout:</p>

       > askOllama "Who is the 47th president of the USA?"

<p>Query with prompts from a file and save results to JSONL:</p>

       > askOllama -o outfile.jsonl prompts.txt

<p>Query with persona and RAG context:</p>

       > askOllama -o outfile.txt -p Grogu.txt --rag RAGDIR prompts.txt argfile.txt


## INSTALLATION
<p><strong>askOllama</strong> is an all-in-one Python executable.   Just put it in your executable path or current working directory.  However, like most Python programs, askOllama requires dependencies to be loaded via pip.</p>

    python -m pip install --upgrade pip setuptools wheel

    # Core
    pip install \
      requests \
      langchain langchain-community langchain-chroma langchain-ollama chromadb \
      unstructured pypdf \
      python-docx python-pptx openpyxl


## ENVIRONMENT
<p>Heads-up: you still need an Ollama server running locally (default http://localhost:11434), and --llm overrides OLLAMA_MODEL if you set that env var.</p>

       OLLAMA_API_BASE
              Base URL for the Ollama API (default: http://localhost:11434).

       OLLAMA_MODEL
              Model name for Ollama (default: llama2).

## AUTHOR
<p>Binary Systems, Inc.  <br>
http://www.binary-systems.com</p>


