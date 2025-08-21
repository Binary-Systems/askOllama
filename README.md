# askOllama(1)                General Commands Manual               askOllama(1)

## NAME
<p>askOllama  -  Command‐line interface for batch querying Ollama models with RAG support</p>
       
## SYNOPSIS

       askOllama [-o outfile] [-p personafile] [--rag RAGDIR] prompt [arg]

## DESCRIPTION

<p>askOllama is a command‐line tool for sending prompts and  arguments  to
       Ollama models and retrieving responses. It supports batch queries, per‐
       sona  files, retrieval‐augmented generation (RAG) directories, and mul‐
       tiple output formats.</p>

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

## OPTIONS
       -o outfile<br>
              Specify an output file for results. Supported  formats:  .jsonl,
              .csv,  .rtf,  or  plain text. If omitted, results are printed to
              stdout.

       -p personafile<br>
              Specify a file containing a persona/system prompt to set for the
              model.

       --rag RAGDIR<br>
              Specify a directory or file to upload as context  documents  for
              retrieval‐augmented generation.

       prompt A prompt string or a file containing prompts (one per line).

       arg    An  optional  argument  string or file containing arguments (one
              per line).

       ‐h     Show usage/options summary.

       ‐‐help Show this help/man page.


## ENVIRONMENT

       OLLAMA_API_BASE<br>
              Base URL for the Ollama API (default: http://localhost:11434).

       OLLAMA_MODEL<br>
              Model name for Ollama (default: llama2).


## EXAMPLES

<p>Query with a single prompt and print result to stdout:</p>

       > askOllama "Who is the 47th president of the USA?"

<p>Query with prompts from a file and save results to JSONL:</p>

       > askOllama -o outfile.jsonl prompts.txt

<p>Query with persona and RAG context:</p>

       > askOllama -o outfile.txt -p Grogu.txt --rag RAGDIR prompts.txt argfile.txt

## AUTHOR
<p>Binary Systems, Inc.  <br>
http://www.binary-systems.com</p>


