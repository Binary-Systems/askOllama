#!/bin/bash
# askOllama Portable Linux AppImage Build

echo "=== askOllama Portable Linux AppImage Build ==="
echo "[1/6] Creating venv..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/6] Installing dependencies..."
pip install --upgrade pip
pip install pyinstaller langchain langchain-community langchain-chroma langchain-ollama unstructured pypdf openpyxl python-docx python-pptx requests chromadb aiohttp striprtf beautifulsoup4 lxml markdown fuse2 (if on Ubuntu/Debian)

echo "[3/6] Building executable..."
pyinstaller --onefile --clean --name askOllama askOllama.py

echo "[4/6] Creating AppDir..."
mkdir -p dist/AppDir/usr/bin
cp dist/askOllama dist/AppDir/usr/bin/
cp askOllama.py dist/AppDir/usr/bin/askOllama.py
echo '[Desktop Entry]
Type=Application
Name=askOllama
Exec=askOllama %U
Icon=terminal' > dist/AppDir/askOllama.desktop

echo "[5/6] Creating AppImage..."
APPIMAGE_NAME="askOllama-x86_64.AppImage"
wget -q https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage -O dist/appimagetool
chmod +x dist/appimagetool
chmod +x dist/AppDir/usr/bin/askOllama
dist/appimagetool dist/AppDir/ $APPIMAGE_NAME

echo "[6/6] Creating examples..."
echo "http://localhost:11434" > dist/farm.example.txt
echo "http://ollama1:11434" >> dist/farm.example.txt
echo "http://ollama2:11434" >> dist/farm.example.txt
echo "Tell me a joke." > dist/prompts.example.txt
echo "Why did the chicken cross the road?" > dist/args.example.txt

echo "=== BUILD COMPLETE! ==="
echo "AppImage: $APPIMAGE_NAME"
echo "Run: chmod +x $APPIMAGE_NAME && ./$APPIMAGE_NAME --version"
