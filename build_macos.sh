#!/bin/bash
# askOllama Portable macOS Build

echo "=== askOllama Portable macOS Build ==="
echo "[1/6] Creating venv..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/6] Installing dependencies..."
pip install --upgrade pip
pip install pyinstaller langchain langchain-community langchain-chroma langchain-ollama unstructured pypdf openpyxl python-docx python-pptx requests chromadb aiohttp striprtf beautifulsoup4 lxml markdown

echo "[3/6] Building universal executable..."
pyinstaller --onefile --clean --name askOllama askOllama.py

echo "[4/6] Creating launcher..."
echo '#!/bin/bash' > dist/askOllama
echo 'exec "$0.exe" "$@"' >> dist/askOllama
chmod +x dist/askOllama

echo "[5/6] Copying script..."
cp askOllama.py dist/askOllama

echo "[6/6] Creating examples..."
echo "http://localhost:11434" > dist/farm.example.txt
echo "http://ollama1:11434" >> dist/farm.example.txt
echo "http://ollama2:11434" >> dist/farm.example.txt
echo "Tell me a joke." > dist/prompts.example.txt
echo "Why did the chicken cross the road?" > dist/args.example.txt

echo "=== BUILD COMPLETE! ==="
echo "Folder: dist/askOllama-portable"
echo "Run: cd dist && ./askOllama --version"
echo "Zip: zip -r askOllama-portable.zip dist/"
