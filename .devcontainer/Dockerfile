# Wir starten vom selben Python-Basis-Image, das Sie bereits verwenden.
FROM mcr.microsoft.com/devcontainers/python:3.11

# --- TORCH UND REQUIREMENTS INSTALLATION ---

# Erstellen Sie einen Arbeitsordner im Image.
WORKDIR /app

# Kopieren Sie die requirements.txt vom Host in den Arbeitsordner des Images.
# Der erste Pfad ist relativ zum Dockerfile. ../ geht eine Ebene hoch.
COPY requirements.txt .

# Führen Sie pip install aus. Dies geschieht nur einmal beim Bauen
# und wird dann dauerhaft zwischengespeichert (gecacht).
RUN pip --no-cache-dir install -r requirements.txt

# Setzen Sie den Arbeitsordner für den Container zurück auf den Standard-Workspace,
# damit das Terminal im richtigen Verzeichnis startet.
WORKDIR /workspaces