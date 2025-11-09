@echo off
echo === askOllama Portable Windows Build ===
echo [1/6] Creating venv...
python -m venv .venv
call .venv\Scripts\activate

echo [2/6] Installing dependencies...
pip install --upgrade pip
pip install pyinstaller langchain langchain-community langchain-chroma langchain-ollama unstructured pypdf openpyxl python-docx python-pptx requests chromadb aiohttp striprtf beautifulsoup4 lxml markdown

echo [3/6] Building executable...
pyinstaller --onefile --clean --name askOllama askOllama.py

echo [4/6] Creating launcher...
echo @echo off > dist\askOllama.bat
echo set "SCRIPT_DIR=%%~dp0" >> dist\askOllama.bat
echo "%%SCRIPT_DIR%%askOllama.exe" %%* >> dist\askOllama.bat

echo [5/6] Copying script...
copy askOllama.py dist\askOllama

echo [6/6] Creating examples...
echo http://localhost:11434 > dist\farm.example.txt
echo http://ollama1:11434 >> dist\farm.example.txt
echo http://ollama2:11434 >> dist\farm.example.txt
echo Tell me a joke. > dist\prompts.example.txt
echo Why did the chicken cross the road? > dist\args.example.txt

echo === BUILD COMPLETE! ===
echo Folder: dist/askOllama-portable
echo Run: cd dist && .\askOllama.bat --version
pause
