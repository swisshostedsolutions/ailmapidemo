Dynamic Hugging Face Pipeline API 🚀
This project is a FastAPI-based backend that provides a dynamic, self-documenting API for running various Hugging Face transformers pipelines. It's designed to be a robust foundation for building custom AI-powered applications.

The key feature of this backend is its ability to programmatically discover supported tasks and their required input parameters, making it highly flexible and adaptable to the evolving transformers library.

## Key Features
✅ Dynamic Task Discovery: Automatically generates a list of all pipelines supported by the installed transformers library.

🧠 Intelligent Parameter Introspection: Programmatically inspects pipeline classes to determine the exact input arguments required for each task.

🛠️ Automated Inconsistency Correction: Includes a discovery script that finds and corrects discrepancies between a pipeline's documented and runtime parameter names.

⚙️ Flexible Inference Endpoint: A single, powerful POST /run-pipeline endpoint that can execute a wide variety of tasks.

📦 Local Model Caching: Downloads and saves models to a local backend/local_model directory to avoid repeated downloads and enable offline use.

🧪 Automated API Testing: A robust test suite built with pytest to ensure the API endpoints are working correctly.

📚 Self-Documenting API: Automatically generates interactive API documentation (Swagger UI and ReDoc) via FastAPI.

## Project Structure
The project is organized into distinct directories for the backend application, utility scripts, and tests.

.
├── backend/
│   ├── main.py                 # The FastAPI application entry point
│   ├── utils/                  # Utility functions (model loading, task analysis)
│   ├── tests/                  # Automated tests for the API
│   ├── local_model/            # Directory for storing downloaded HF models
│   └── requirements.txt        # Python dependencies for the backend
├── scripts/
│   └── discover_param_names.py # Standalone script for pre-caching and discovery
├── test_assets/
│   ├── sample.jpg
│   ├── sample.wav
│   └── sample.mp4
└── README.md                   # This file
## Setup and Installation
Follow these steps to set up and run the project locally.

1. Clone the Repository

Bash

git clone <your-repository-url>
cd <your-repository-name>
2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

Bash

# Create the environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
.\venv\Scripts\activate
3. Install Dependencies
Install all the required Python libraries from the requirements.txt file located in the project root.

Bash

pip install -r requirements.txt
This will install fastapi, uvicorn, transformers, torch, pytest, httpx, and other necessary packages.

## Usage
The project has three main functions: running the development server, running the test suite, and running the discovery script.

Running the Development Server
This command starts the FastAPI application with auto-reload enabled.

Bash

# Run from the project root directory
uvicorn backend.main:app --reload
Once running, you can access:

The API: http://127.0.0.1:8000

Interactive Docs (Swagger UI): http://127.0.0.1:8000/docs

Running the Automated Tests
The test suite verifies that the API endpoints are functioning as expected. The development server must be running in a separate terminal for the tests to work.

Bash

# Run from the project root directory
pytest
Pre-caching Models & Discovering Parameters
This is a one-time utility script that performs two actions:

It programmatically tests every supported pipeline to find parameter name inconsistencies.

It downloads the default model for every task into the backend/local_model/ directory.

Note: This script will take a long time to run and will download several gigabytes of models.

Bash

# Run from the project root directory
python -m scripts.discover_param_names
The script will output a PARAMETER_NAME_MAP dictionary. Copy this dictionary and paste it into backend/utils/task_analyzer.py to apply the corrections.

## API Endpoints
Method	Path	Description
GET	/tasks	Returns a detailed JSON object of all supported tasks, their default models, and required parameters.
POST	/run-pipeline/	Executes a specified pipeline. Requires a JSON body with task_name and an inputs dictionary.
GET	/docs	Provides the interactive Swagger UI for testing the API.
GET	/redoc	Provides the alternative ReDoc API documentation.