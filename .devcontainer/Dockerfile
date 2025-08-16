# Start from the official Microsoft dev container image for Python 3.12
# This base image comes with many useful tools pre-installed.
FROM mcr.microsoft.com/devcontainers/python:3.12

# The dev container images have a non-root user 'vscode' pre-configured.
# It's best practice to use it to avoid running everything as root.
USER vscode

# --- Optional: Install project dependencies ---
# If you have a requirements.txt file in your project root, uncomment
# the following lines to install dependencies when the container is built.

# COPY --chown=vscode:vscode requirements.txt /tmp/pip-tmp/
# RUN pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt