#!/usr/bin/env python3

# Copyright (c) 2025 Binary Systems, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0        
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.    

"""
askOllama — batch query Ollama models with optional RAG, persona, prompt/arg files, and rich outputs.

Additions in this version:
  • Vector DB is stored in a hidden subdir `.rag_db` inside the RAG path (or its parent if a single file is used). This dir is ignored when building the index, so it never pollutes the RAG corpus.
  • `--llm` lets you choose the Ollama model; if missing, the tool attempts to pull it automatically via the Ollama HTTP API.
  • `--verbose` toggles chatty logs (DEBUG level). Default is INFO.
  • Restored original capabilities: persona file (-p), prompt file / argument file (positional), NxM cartesian execution, and multi-format output (-o) including .jsonl, .csv, .rtf, or plain text.

Environment variables (kept for compatibility):
  • OLLAMA_API_BASE (default http://localhost:11434)
  • OLLAMA_MODEL (default 'llama2', overridden by --llm if provided)

Requires: langchain, langchain-community, langchain-chroma, langchain-ollama, unstructured, pypdf, openpyxl, python-docx, python-pptx, requests, chromadb, aiohttp, striprtf, beautifulsoup4, lxml, markdown
"""

import argparse
import os
import sys
import json
import csv
import shutil
import logging
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import re
from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup
import lxml.etree as ET
from markdown import markdown
from itertools import cycle

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredRTFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredXMLLoader,
)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import openpyxl
except ImportError:
    openpyxl = None


ASKOLLAMA_MAN = r"""
.TH ASKOLLAMA 1 "August 2025" "User Commands"
.SH NAME
askOllama \- Command-line interface for batch querying Ollama models with optional RAG support
.SH SYNOPSIS
.B askOllama
[\-o outfile] [\-p personafile] [\-\-rag RAGDIR] [\-\-llm MODEL] [\-\-verbose] [\-\-separator SEPARATOR] [\-\-merge] [\-\-farm FARM] [\-\-transpose] [\-\-timeout TIMEOUT] prompt [arg]
.SH DESCRIPTION
.B askOllama
is a command-line tool for sending prompts and arguments to Ollama models and retrieving responses.
It supports Retrieval-Augmented Generation (RAG) by indexing documents in a specified directory or a
single file. The vector database used for RAG is isolated in a hidden subdirectory `.rag_db` inside the
RAG path (or the file's parent directory), and that directory is excluded from the corpus so it does
not interfere with which files are indexed.

Prompts and arguments can be read from files: if the positional \fIprompt\fR or \fIarg\fR looks like a
file path, it will be parsed. Multiple prompts and multiple arguments are combined in a Cartesian
product (N×M) and executed, with results written in order.

.TP
.BR -p " PERSONA", \--persona " PERSONA"
Use the given file as a persona/system prompt (prepended as the system message). If RAG is enabled,
a short instruction is appended to answer based solely on the provided context.

.TP
.BR --rag " RAGDIR"
Directory (or a single file) to build a RAG index from. The index is stored in `.rag_db` under the
RAG directory. The `.rag_db` directory is ignored while crawling. Supports .pdf, .txt, .html, .md, .rtf, .doc, .docx, .ppt, .pptx, .xls, .xlsx, .xml.

.TP
.BR --llm " MODEL"
Specify the Ollama model to use (e.g., `llama3:8b`). If the model is not installed, askOllama will
attempt to download it via the Ollama HTTP API before running.

.TP
.BR --verbose
Enable verbose logging (DEBUG level). Default logging is INFO.

.TP
.BR --separator " SEPARATOR"
Specify a string to place after each result in text-style outputs (.txt, .rtf, .doc, .docx, .md, .html, .pages). Ignored for tabular formats (.csv, .xlsx, .numbers, .jsonl). Default: three newlines.

.TP
.BR --merge
Include the prompt and argument before each result in the output.

.TP
.BR --farm " FARM"
Specify Ollama servers for workload distribution: comma-separated URLs or a file with URLs (each on separate lines or comma-separated on lines). If a file is given, only those servers are used. If comma-separated URLs are given, only those are used. If no --farm, defaults to OLLAMA_API_BASE (or localhost). If using a file, it can be dynamically updated during runtime.

.TP
.BR --transpose
Transpose the Cartesian product order: process all prompts for each argument (arguments outer loop).

.TP
.BR --timeout " TIMEOUT"
Timeout for server response in seconds (default: 30).

.TP
.BR -o " OUTFILE"
Write results to OUTFILE. Supported extensions: `.jsonl`, `.csv`, `.rtf`, `.txt`, `.md`, `.html`, `.doc`, `.docx`, `.pages`, `.xlsx`, `.numbers`.

.SH ENVIRONMENT
.TP
.B OLLAMA_API_BASE
Base URL for the Ollama API (default: http://localhost:11434).
.TP
.B OLLAMA_MODEL
Model name for Ollama (default: llama2). Overridden by --llm.

.SH EXAMPLES
.TP
Query one prompt and print to stdout:
.B
askOllama "Who is the president of the USA?"
.TP
Prompt file × argument file to CSV using RAG:
.B
askOllama -o results.csv --rag ./docs prompts.txt args.txt --llm llama3:8b

.SH SEE ALSO
Ollama API
"""


def print_usage():
    print(
        """
Usage: askOllama [options] <prompt> [<arg>]

Options:
  -o OUTFILE        Output file (.jsonl, .csv, .rtf, or text)
  -p PERSONA        Persona/system prompt file
  --persona PERSONA Alias for -p
  --rag RAGDIR      RAG context directory or single file (index saved in .rag_db)
  --llm MODEL       Ollama model name (pulled if missing)
  --verbose         Verbose logging
  --separator SEP   Separator string for text outputs (after each result)
  --merge           Include prompt and arg before each result
  --farm FARM       Ollama servers: URLs or farmfile
  --transpose       Transpose order (args outer)
  --timeout TIMEOUT Timeout for server response (default 30s)
  -h                Show concise usage
  --help            Show embedded man page
        """.strip()
    )


def print_man_page():
    import tempfile
    import subprocess
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.man') as tmp:
        tmp.write(ASKOLLAMA_MAN)
        tmp_path = tmp.name
    try:
        try:
            # Use groff if available for nicer formatting
            subprocess.run(["groff", "-Tutf8", "-man", tmp_path], check=True)
        except Exception:
            # Fallback: just cat the man text
            with open(tmp_path, 'r') as f:
                sys.stdout.write(f.read())
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# -------- RAG helpers --------

def _loader_for(path: Path):
    s = path.suffix.lower()
    if s == ".pdf":
        return UnstructuredPDFLoader(str(path))
    if s in {".txt", ".log", ".py", ".json"}:
        return TextLoader(str(path))
    if s in {".html", ".htm"}:
        return UnstructuredHTMLLoader(str(path))
    if s in {".md", ".markdown"}:
        return UnstructuredMarkdownLoader(str(path))
    if s == ".rtf":
        return UnstructuredRTFLoader(str(path))
    if s in {".doc", ".docx"}:
        return UnstructuredWordDocumentLoader(str(path))
    if s in {".ppt", ".pptx"}:
        return UnstructuredPowerPointLoader(str(path))
    if s in {".xls", ".xlsx"}:
        return UnstructuredExcelLoader(str(path))
    if s == ".xml":
        return UnstructuredXMLLoader(str(path))
    return None


def _collect_rag_files(rag_target: Path) -> List[Path]:
    if rag_target.is_file():
        return [rag_target]
    files: List[Path] = []
    for p in rag_target.rglob('*'):
        if p.is_dir():
            # Skip the embedded vector DB dir
            if p.name == ".rag_db":
                continue
            continue
        if p.parent.name == ".rag_db":
            continue
        if _loader_for(p) is not None:
            files.append(p)
    return files


def build_vector_db(rag_target: Path) -> Optional[Chroma]:
    """Build (or rebuild) the vector DB under rag_target/.rag_db and return a Chroma instance.

    This rebuilds the index each run for simplicity and freshness. The index lives in a hidden
    `.rag_db` folder adjacent to the RAG sources and is excluded from future crawls.
    """
    persist_dir = (rag_target.parent if rag_target.is_file() else rag_target) / ".rag_db"
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Clear only the index directory, never source files
    try:
        # Remove existing index to maintain a clean, deterministic state
        if any(persist_dir.iterdir()):
            shutil.rmtree(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.warning(f"Could not reset vector DB at {persist_dir}: {e}")

    files = _collect_rag_files(rag_target)
    logging.info(f"RAG: found {len(files)} file(s) to index under {rag_target}")

    docs = []
    for fp in files:
        loader = _loader_for(fp)
        if loader is None:
            logging.debug(f"Skipping unsupported file: {fp}")
            continue
        try:
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            logging.warning(f"Failed to load {fp}: {e}")

    logging.info(f"Loaded {len(docs)} documents. Splitting into chunks…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    logging.info(f"Created {len(chunks)} chunks for embedding.")

    if not chunks:
        logging.warning("No documents indexable for RAG. Exiting vector DB build.")
        return None

    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=str(persist_dir))
    

    #vectordb.persist()
    logging.info(f"Vector DB stored at {persist_dir}")
    return vectordb


def retrieve_context(db: Chroma, query: str, k: int = 4) -> str:
    results = db.similarity_search(query, k=k)
    return "\n\n".join(r.page_content for r in results)


# -------- Ollama helpers --------

def _list_models_via_api(base_url: str) -> List[str]:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", [])]
    except Exception as e:
        logging.debug(f"/api/tags failed: {e}")
    return []


def _pull_model_via_api(base_url: str, model: str) -> bool:
    try:
        logging.info(f"Pulling model via API: {model}")
        with requests.post(f"{base_url}/api/pull", json={"name": model}, stream=True, timeout=600) as r:
            if r.status_code not in (200, 201):
                logging.error(f"pull API returned {r.status_code}")
                return False
            # Stream progress lines if any
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    evt = json.loads(line.decode('utf-8'))
                    status = evt.get('status') or evt.get('digest') or ''
                    if status:
                        logging.info(f"pull: {status}")
                except Exception:
                    pass
        # give the daemon a second to index newly pulled model
        time.sleep(1)
        return True
    except Exception as e:
        logging.error(f"Model pull failed: {e}")
        return False


def ensure_model_available(model: str, base_url: str) -> None:
    installed = _list_models_via_api(base_url)
    if any(model == m or model.split(":")[0] == m.split(":")[0] for m in installed):
        logging.info(f"Model present: {model}")
        return
    logging.info(f"Model not present: {model}")
    if not _pull_model_via_api(base_url, model):
        logging.error(
            "Unable to pull model automatically. Ensure Ollama is running and the model name is correct."
        )
        sys.exit(1)


# -------- I/O helpers --------

def read_file(file_path: str) -> List[str]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    text = ""
    if suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            content = []
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get('prompt') or data.get('text') or data.get('argument') or str(data)
                    content.append(text)
                except json.JSONDecodeError:
                    content.append(line.strip())
            return content
    elif suffix == '.csv':
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            content = [row[0] for row in reader if row]
            return content
    elif suffix == '.rtf':
        with open(path, 'r', encoding='utf-8') as f:
            rtf_text = f.read()
            text = rtf_to_text(rtf_text)
    elif suffix in ['.doc', '.docx']:
        if Document is None:
            logging.error("python-docx required for .doc/.docx")
            sys.exit(1)
        doc = Document(path)
        text = '\n\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    elif suffix in ['.html', '.htm']:
        with open(path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            text = soup.get_text(separator='\n\n', strip=True)
    elif suffix == '.xml':
        with open(path, 'r', encoding='utf-8') as f:
            tree = ET.parse(f)
            text = ET.tostring(tree.getroot(), encoding='unicode', method='text')
    elif suffix == '.md':
        with open(path, 'r', encoding='utf-8') as f:
            md_text = f.read()
            text = markdown(md_text, output_format='plaintext')
    else:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

    # Split by paragraphs (blank lines)
    content = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return content


def append_result(outfile: Optional[str], prompt: str, arg: Optional[str], result: str, suf: str, merge: bool, separator: Optional[str]) -> None:
    arg_str = arg if arg is not None else "None"

    content = ""
    if merge:
        content += f"Prompt: {prompt}\nArg: {arg_str}\n\n"
    content += result

    sep = (separator + "\n") if separator is not None else "\n\n\n"

    tabular_formats = ['.csv', '.jsonl', '.xlsx', '.numbers']
    text_formats = ['.txt', '.md', '.html', '.rtf', '.doc', '.docx', '.pages']

    is_tabular = suf in tabular_formats if outfile else False
    is_text = not is_tabular

    if outfile is None:
        print(content)
        print(sep)
        return

    out = Path(outfile)
    mode = 'a' if out.exists() else 'w'
    is_new = not out.exists()

    if suf == '.jsonl':
        with open(out, mode, encoding='utf-8') as f:
            entry = {"prompt": prompt, "arg": arg or "", "result": result} if merge else {"result": result}
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    elif suf == '.csv' or suf == '.numbers':
        with open(out, mode, encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            if is_new:
                header = ["prompt", "arg", "result"] if merge else ["result"]
                w.writerow(header)
            row = [prompt, arg or "", result] if merge else [result]
            w.writerow(row)

    elif suf == '.xlsx':
        if openpyxl is None:
            logging.error("openpyxl required for .xlsx support")
            sys.exit(1)
        wb = openpyxl.load_workbook(out) if out.exists() else openpyxl.Workbook()
        ws = wb.active
        if is_new:
            header = ["prompt", "arg", "result"] if merge else ["result"]
            ws.append(header)
        row = [prompt, arg or "", result] if merge else [result]
        ws.append(row)
        wb.save(out)

    elif suf == '.rtf':
        if is_new:
            with open(out, 'w', encoding='utf-8') as f:
                f.write(r'{\rtf1\ansi\deff0 {\fonttbl {\f0 Courier;}}' + "\n")
                f.write("}")
        with open(out, 'r', encoding='utf-8') as f:
            rtf_content = f.read().rstrip('}')
        safe_content = content.replace('\\', r'\\').replace('{', r'\{').replace('}', r'\}')
        safe_sep = sep.replace('\\', r'\\').replace('{', r'\{').replace('}', r'\}')
        new_rtf = rtf_content + safe_content.replace('\n', '\\par\n') + '\\par\n' + safe_sep.replace('\n', '\\par\n') + "}"
        with open(out, 'w', encoding='utf-8') as f:
            f.write(new_rtf)

    elif suf in ['.doc', '.docx']:
        if Document is None:
            logging.error("python-docx required for .docx support")
            sys.exit(1)
        doc = Document(out) if out.exists() else Document()
        lines = content.split('\n')
        for line in lines:
            if line.strip():
                doc.add_paragraph(line)
        doc.add_paragraph(sep)
        doc.save(out)

    elif suf == '.html':
        if is_new:
            with open(out, 'w', encoding='utf-8') as f:
                f.write("<html><body>\n")
                f.write("</body></html>")
        with open(out, 'r', encoding='utf-8') as f:
            html_content = f.read().rstrip('</body></html>')
        html_content += "<p>" + content.replace('\n', '<br>') + "</p>\n"
        html_content += "<p>" + sep.replace('\n', '<br>') + "</p>\n" if sep else "<br><br><br>\n"
        html_content += "</body></html>"
        with open(out, 'w', encoding='utf-8') as f:
            f.write(html_content)

    elif suf in ['.txt', '.md', '.pages']:
        with open(out, mode, encoding='utf-8') as f:
            f.write(content + "\n" + sep)

    else:
        with open(out, mode, encoding='utf-8') as f:
            f.write(content + "\n" + sep)


# -------- main --------

async def process_query(current_url: str, payload: Dict, headers: Dict, timeout: int) -> str:
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(f"{current_url}/api/chat", headers=headers, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get('message', {}).get('content', '')
            else:
                raise Exception(f"Failed with status {resp.status}")

async def worker(queue: asyncio.Queue, results: Dict[int, Tuple[str, str]], farmfile, farm_urls, base_url, headers: Dict, model: str, semaphore: asyncio.Semaphore, total: int, offline_servers: set, lock: asyncio.Lock, timeout: int):
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            break
        idx, payload, prompt_text, arg_text = task
        success = False
        async with lock:
            servers = get_servers(farmfile, farm_urls, base_url)
            servers = [s for s in servers if s not in offline_servers]
        num_servers = len(servers)
        if num_servers == 0:
            logging.info(f"All servers offline for prompt index {idx}, re-queueing for retry.")
            await queue.put((idx, payload, prompt_text, arg_text))
            queue.task_done()
            await asyncio.sleep(1)  # Prevent tight loop
            continue
        start_server = idx % num_servers
        for attempt in range(num_servers):
            server_idx = (start_server + attempt) % num_servers
            current_url = servers[server_idx]
            async with semaphore:
                try:
                    logging.info(f"Querying model ({model}) on {current_url} [{idx+1}/{total}]:")
                    result = await process_query(current_url, payload, headers, timeout)
                    results[idx] = (result, current_url)
                    success = True
                    break
                except asyncio.TimeoutError:
                    async with lock:
                        offline_servers.add(current_url)
                    logging.warning(f"Server {current_url} timed out and marked offline.")
                except Exception as e:
                    logging.warning(f"Server {current_url} failed: {e}")
        if not success:
            logging.info(f"All servers offline for prompt index {idx}, re-queueing for retry.")
            await queue.put((idx, payload, prompt_text, arg_text))
        queue.task_done()

async def server_checker(offline_servers: set, lock: asyncio.Lock, done_event: asyncio.Event):
    while not done_event.is_set():
        await asyncio.sleep(60)  # Check every 60 seconds
        to_remove = []
        async with lock:
            for server in list(offline_servers):
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                        async with session.get(f"{server}/api/tags") as resp:
                            if resp.status == 200:
                                to_remove.append(server)
                except Exception:
                    pass
            for server in to_remove:
                offline_servers.remove(server)
                logging.info(f"Server {server} is back online.")

async def output_loop(results, pairs, total, skip, args, suf, resume_file):
    processed = skip
    while processed < total:
        if processed in results:
            prompt_text, arg_text = pairs[processed]
            result, server_url = results.pop(processed)
            logging.info(f"Prompt [{processed+1}/{total}]: {prompt_text}")
            logging.info(f"Arg [{processed+1}/{total}]: {arg_text if arg_text else 'None'}")
            logging.info(f"Result [{processed+1}/{total}] from {server_url}:\n\n{result}\n")
            append_result(args.o, prompt_text, arg_text, result, suf, args.merge, args.separator)
            processed += 1
            if resume_file:
                with open(resume_file, 'w', encoding='utf-8') as f:
                    json.dump({"processed": processed}, f)
        else:
            await asyncio.sleep(0.5)
    return processed

async def main_async(args, pairs, total, base_url, model, system_prompt, db, resume_file, skip, suf, farmfile, farm_urls, timeout):
    start_time = time.time()
    servers = get_servers(farmfile, farm_urls, base_url)
    if not servers:
        servers = [base_url]
    num_servers = len(servers)
    headers = {'Content-Type': 'application/json'}

    semaphore = asyncio.Semaphore(num_servers)
    queue = asyncio.Queue()
    results: Dict[int, Tuple[str, str]] = {}
    offline_servers = set()
    lock = asyncio.Lock()
    done_event = asyncio.Event()
    checker_task = asyncio.create_task(server_checker(offline_servers, lock, done_event))
    workers = [asyncio.create_task(worker(queue, results, farmfile, farm_urls, base_url, headers, model, semaphore, total, offline_servers, lock, timeout)) for _ in range(num_servers)]

    for idx in range(skip, total):
        prompt_text, arg_text = pairs[idx]
        message = prompt_text if arg_text is None else f"{prompt_text}\n{arg_text}"
        context = retrieve_context(db, message) if db else ""
        user_content = f"Context:\n{context}\n\nQuestion:\n{message}" if context else message
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        }
        await queue.put((idx, payload, prompt_text, arg_text))

    output_task = asyncio.create_task(output_loop(results, pairs, total, skip, args, suf, resume_file))

    for _ in range(num_servers):
        await queue.put(None)

    await queue.join()
    for w in workers:
        await w

    processed = await output_task
    done_event.set()
    await checker_task
    elapsed_time = time.time() - start_time
    return processed, elapsed_time

def get_servers(farmfile, farm_urls, base_url):
    if farmfile:
        try:
            lines = read_file(str(farmfile))
            urls = []
            for line in lines:
                urls.extend([u.strip() for u in line.split(',') if u.strip()])
            return urls
        except Exception as e:
            logging.warning(f"Failed to read farmfile {farmfile}: {e}")
            return []
    elif farm_urls:
        return farm_urls
    else:
        return [base_url]

def main():
    parser = argparse.ArgumentParser(description="askOllama", add_help=False)
    parser.add_argument('-o', default=None, help="outfile")
    parser.add_argument('-p', '--persona', default=None, help="personafile")
    parser.add_argument('--rag', default=None, help="RAGDIR or single file")
    parser.add_argument('--llm', default=None, help="Specify Ollama LLM to use (pulled if missing)")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose (DEBUG) logging")
    parser.add_argument('--separator', default=None, help="Separator string for text outputs")
    parser.add_argument('--merge', action='store_true', help="Include prompt and arg in output")
    parser.add_argument('--farm', default=None, help="Additional Ollama servers: URLs or farmfile")
    parser.add_argument('--transpose', action='store_true', help="Transpose order (args outer)")
    parser.add_argument('--timeout', default=30, type=int, help="Timeout for server response in seconds")
    parser.add_argument('-h', action='store_true', help="Show usage/options summary")
    parser.add_argument('--help', action='store_true', help="Show embedded man page")
    parser.add_argument('prompt', nargs='?', help="Prompt string or promptfile path")
    parser.add_argument('arg', nargs='?', default=None, help="Argument string or argfile path")
    args = parser.parse_args()

    # Logging level
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='[%(levelname)s] %(message)s')

    if args.h:
        print_usage()
        return
    if args.help or args.prompt is None:
        print_man_page()
        return

    os.environ['ANONYMIZED_TELEMETRY'] = 'False'

    base_url = os.environ.get('OLLAMA_API_BASE', 'http://localhost:11434')
    model = args.llm or os.environ.get('OLLAMA_MODEL', 'llama2')

    farmfile = None
    farm_urls = []
    if args.farm:
        farm_path = Path(args.farm)
        if farm_path.exists():
            farmfile = farm_path
        else:
            farm_urls = [u.strip() for u in args.farm.split(',') if u.strip()]

    # Ensure models on initial servers
    initial_servers = get_servers(farmfile, farm_urls, base_url)
    for s in initial_servers:
        ensure_model_available(model, s)
        if args.rag:
            ensure_model_available('nomic-embed-text', s)

    # Persona
    system_prompt = ''
    if args.persona:
        try:
            with open(args.persona, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
        except Exception as e:
            logging.error(f"Failed to read persona file {args.persona}: {e}")
            sys.exit(1)

    # RAG (using default base_url for embeddings)
    db = None
    if args.rag:
        if system_prompt:
            system_prompt += '\n\n'
        system_prompt += 'Answer the question based solely on the context provided.'
        rag_target = Path(args.rag)
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
        db = build_vector_db(rag_target)

    # Inputs: prompts
    prompt_path = Path(args.prompt)
    if prompt_path.exists():
        prompts = read_file(str(prompt_path))
    else:
        prompts = [args.prompt]
    logging.info(f"Loaded {len(prompts)} prompt(s)")

    # Inputs: arguments
    args_list: List[Optional[str]]
    if args.arg is None:
        args_list = [None]
    else:
        arg_path = Path(args.arg)
        if arg_path.exists():
            args_list = read_file(str(arg_path))
        else:
            args_list = [args.arg]
    logging.info(f"Loaded {len(args_list)} argument value(s)")

    # Cartesian product
    if args.transpose:
        pairs: List[Tuple[str, Optional[str]]] = [(p, a) for a in args_list for p in prompts]
    else:
        pairs = [(p, a) for p in prompts for a in args_list]
    total = len(pairs)
    if total == 0:
        logging.error("No prompts or arguments to process.")
        sys.exit(1)

    # Resume setup
    suf = Path(args.o).suffix.lower() if args.o else ''
    resume_file = None
    skip = 0
    if args.o:
        resume_file = Path(args.o).parent / f".{Path(args.o).name}.resume"
        if not Path(args.o).exists() and resume_file.exists():
            resume_file.unlink()
            logging.info(f"Output file does not exist; deleted existing resume file {resume_file}")
        if resume_file.exists():
            try:
                with open(resume_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    skip = data.get('processed', 0)
            except Exception as e:
                logging.warning(f"Failed to load resume file {resume_file}: {e}")
                skip = 0
        logging.info(f"Resuming from {skip} processed entries")
    else:
        logging.info("Output to stdout; no resume support.")

    processed, elapsed_time = asyncio.run(main_async(args, pairs, total, base_url, model, system_prompt, db, resume_file, skip, suf, farmfile, farm_urls, args.timeout))

    if resume_file:
        try:
            resume_file.unlink()
            logging.info(f"Completed successfully; removed resume file {resume_file}")
        except Exception as e:
            logging.warning(f"Failed to remove resume file {resume_file}: {e}")

    logging.info(f"Completed {processed} of {total} request(s) in {elapsed_time:.2f} seconds.")
    logging.info("Done.")


if __name__ == '__main__':
    main()