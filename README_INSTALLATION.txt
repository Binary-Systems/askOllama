

1. Build a stand-alone executable with PyInstaller

# 1. Create a clean venv (optional but recommended)
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install everything askOllama needs
pip install --upgrade pip
pip install askOllama  # or pip install -r requirements.txt if you have one

# 3. Install PyInstaller
pip install pyinstaller

# 4. Build a single-file executable
pyinstaller --onefile ^
            --add-data "C:\path\to\.venv\Lib\site-packages\langchain;langchain" ^
            --add-data "C:\path\to\.venv\Lib\site-packages\chromadb;chromadb" ^
            askOllama.py


Flag,Why you need it
--onefile,Packs everything into one askOllama.exe
"--add-data ""path;dest""","Copies heavy wheels that PyInstaller can’t auto-detect (langchain, chromadb, torch, etc.)"
--clean,Removes old build cache (use every time)


2. Keep the shebang and make Windows respect it
Windows ignores #!/usr/bin/env python3, but we can trick it with a tiny batch wrapper.
Create askOllama.bat in the same folder as the exe:


@echo off
REM askOllama.bat — makes the shebang work on Windows
set "SCRIPT_DIR=%~dp0"
"%SCRIPT_DIR%askOllama.exe" %*


Now rename your script to askOllama (no extension) and make it look like a real script:

#!/usr/bin/env python3
# askOllama — your normal script (exactly the same as before)




3. Final folder layout (copy this entire folder to any PC)

askOllama-portable\
├── askOllama          ← the Python script with shebang
├── askOllama.bat      ← tiny launcher
├── askOllama.exe      ← the big PyInstaller binary
└── (any farmfiles, prompts.txt, etc.)


Windows sees askOllama → runs askOllama.bat → launches the exe → done.



6. One-liner for your build server (copy-paste)

python -m venv .venv && .\.venv\Scripts\activate && `
pip install --upgrade pip && `
pip install pyinstaller langchain chromadb torch ollama aiohttp && `
pyinstaller --onefile --clean --name askOllama askOllama.py && `
echo @echo off > dist\askOllama.bat && `
echo "%~dp0askOllama.exe" %%* >> dist\askOllama.bat && `
copy askOllama.py dist\askOllama && `
echo "=== DONE === Portable folder is in dist/"


Drop the dist folder on any Windows machine and you’re good to go — no Python, no admin, no hassle.
Enjoy your truly portable askOllama!





=== askOllama — Portable Edition ===

Just double-click askOllama.bat or run in terminal:

    askOllama prompts.txt args.txt --farm farm.txt --llm llama3.1:8b

Or drag-and-drop files onto askOllama.bat!

Features:
- No Python needed
- No admin rights
- No PATH changes
- Works offline after first run
- Auto-reconnects to servers
- Real-time ordered output
- Resume support (-o output.jsonl)

Example:
    askOllama "Tell me a joke" --llm llama3.2
    askOllama prompts.txt --farm my_servers.txt -o results.jsonl

Enjoy!