# in tests/test_all_pipelines.py
import pytest
import httpx
import pandas as pd


API_URL = "http://127.0.0.1:8000"


@pytest.mark.asyncio
async def test_pipeline_audio_classification():
    """
    Tests the audio-classification pipeline.
    """
    task_name = "audio-classification"
    inputs = {
        # The pipeline expects the path to our sample audio file.
        "inputs": "test_assets/sample.wav"
    }
    
    payload = {
        "task_name": task_name,
        "inputs": inputs
    }
    
    print(f"\n--- Testing pipeline: {task_name} ---")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
    
    # Assert that the API call was successful
    assert response.status_code == 200, f"API call for {task_name} failed: {response.text}"
    
    # Assert that the response contains a valid, non-empty result
    response_data = response.json()
    assert "result" in response_data
    assert isinstance(response_data["result"], list)
    assert len(response_data["result"]) > 0
    
    print(f"✅ Pipeline '{task_name}' succeeded.")


@pytest.mark.asyncio
async def test_pipeline_automatic_speech_recognition():
    """
    Tests the automatic-speech-recognition pipeline.
    """
    task_name = "automatic-speech-recognition"
    inputs = {
        "inputs": "test_assets/sample.wav"
    }
    
    payload = {
        "task_name": task_name,
        "inputs": inputs
    }
    
    print(f"\n--- Testing pipeline: {task_name} ---")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
    
    # Assert that the API call was successful
    assert response.status_code == 200, f"API call for {task_name} failed: {response.text}"
    
    # Assert that the response contains a valid result
    response_data = response.json()
    assert "result" in response_data
    # Add a more specific check for the expected output format of this pipeline
    assert "text" in response_data["result"]
    
    print(f"✅ Pipeline '{task_name}' succeeded.")


@pytest.mark.asyncio
async def test_pipeline_table_question_answering():
    """
    Tests the table-question-answering pipeline.
    """
    task_name = "table-question-answering"
    
    # This pipeline requires a table, which we create as a Pandas DataFrame.
    table_data = {
        "actors": ["Brad Pitt", "Leonardo DiCaprio", "Morgan Freeman"],
        "movies": ["Seven", "The Revenant", "The Shawshank Redemption"]
    }
    table = pd.DataFrame.from_dict(table_data)
    
    # The inputs dictionary must match what the pipeline expects.
    inputs = {
        "query": "Which movie did Morgan Freeman star in?",
        "table": table.to_dict(orient='list') # Convert DataFrame to a JSON-friendly format
    }
    
    payload = {
        "task_name": task_name,
        "inputs": inputs
    }
    
    print(f"\n--- Testing pipeline: {task_name} ---")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
        
    assert response.status_code == 200, f"API call for {task_name} failed: {response.text}"
    
    response_data = response.json()
    assert "result" in response_data
    # A specific check to ensure we get an answer
    assert "answer" in response_data["result"]
    
    print(f"✅ Pipeline '{task_name}' succeeded.")


@pytest.mark.asyncio
async def test_pipeline_feature_extraction():
    """
    Tests the feature-extraction pipeline.
    """
    task_name = "feature-extraction"
    inputs = {
        "inputs": "This is a sentence to be embedded."
    }
    
    payload = {
        "task_name": task_name,
        "inputs": inputs
    }
    
    print(f"\n--- Testing pipeline: {task_name} ---")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
        
    assert response.status_code == 200, f"API call for {task_name} failed: {response.text}"
    
    response_data = response.json()
    assert "result" in response_data
    
    # Specific checks for the embedding structure
    result = response_data["result"]
    assert isinstance(result, list), "Result should be a list (batch)"
    assert isinstance(result[0], list), "First element should be a list (tokens)"
    assert isinstance(result[0][0], list), "Second element should be a list (embedding vector)"
    assert isinstance(result[0][0][0], float), "Embedding values should be floats"
    
    print(f"✅ Pipeline '{task_name}' succeeded.")

@pytest.mark.asyncio
async def test_pipeline_fill_mask():
    """
    Tests the fill-mask pipeline.
    """
    task_name = "fill-mask"
    
    # The input must contain the pipeline's mask_token.
    inputs = {
        "inputs": "The capital of France is <mask>."
    }
    
    payload = {
        "task_name": task_name,
        "inputs": inputs
    }
    
    print(f"\n--- Testing pipeline: {task_name} ---")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
        
    assert response.status_code == 200, f"API call for {task_name} failed: {response.text}"
    
    response_data = response.json()
    assert "result" in response_data
    
    # Specific checks for the fill-mask output structure
    result = response_data["result"]
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], dict)
    assert "score" in result[0]
    assert "token_str" in result[0]
    
    print(f"✅ Pipeline '{task_name}' succeeded.")


@pytest.mark.asyncio
async def test_pipeline_summarization():
    """
    Tests the summarization pipeline.
    """
    task_name = "summarization"
    
    # Provide a longer text suitable for summarization.
    long_text = (
        "Jupiter is the fifth planet from the Sun and the largest in the Solar System. "
        "It is a gas giant with a mass more than two and a half times that of all the other "
        "planets in the Solar System combined, but slightly less than one-thousandth the mass of the Sun. "
        "Jupiter is the third brightest natural object in the Earth's night sky after the Moon and Venus. "
        "It has been known to astronomers since antiquity and is named after the Roman god Jupiter."
    )
    
    inputs = {
        "inputs": long_text
    }
    
    payload = {
        "task_name": task_name,
        "inputs": inputs
    }
    
    print(f"\n--- Testing pipeline: {task_name} ---")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
        
    assert response.status_code == 200, f"API call for {task_name} failed: {response.text}"
    
    response_data = response.json()
    assert "result" in response_data
    
    # Specific checks for the summarization output structure
    result = response_data["result"]
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], dict)
    assert "summary_text" in result[0]
    
    print(f"✅ Pipeline '{task_name}' succeeded.")

@pytest.mark.asyncio
async def test_pipeline_text2text_generation():
    """
    Tests the text2text-generation pipeline (e.g., for translation).
    """
    task_name = "text2text-generation"
    
    # We use a common prefix for T5 models to instruct them on the task.
    inputs = {
        "inputs": "translate English to German: The house is wonderful."
    }
    
    payload = {
        "task_name": task_name,
        "inputs": inputs
    }
    
    print(f"\n--- Testing pipeline: {task_name} ---")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
        
    assert response.status_code == 200, f"API call for {task_name} failed: {response.text}"
    
    response_data = response.json()
    assert "result" in response_data
    
    # Specific checks for the text2text output structure
    result = response_data["result"]
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], dict)
    assert "generated_text" in result[0]
    
    print(f"✅ Pipeline '{task_name}' succeeded.")

@pytest.mark.asyncio
async def test_pipeline_text_generation():
    """
    Tests the text-generation pipeline.
    """
    task_name = "text-generation"
    
    prompt = "In a world where AI companions are common,"
    inputs = {
        "text_inputs": prompt
    }
    
    payload = {
        "task_name": task_name,
        "inputs": inputs
    }
    
    print(f"\n--- Testing pipeline: {task_name} ---")
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
        
    assert response.status_code == 200, f"API call for {task_name} failed: {response.text}"
    
    response_data = response.json()
    assert "result" in response_data
    
    # --- CORRECTED ASSERTIONS FOR NESTED STRUCTURE ---
    result = response_data["result"]
    assert isinstance(result, list)
    assert len(result) > 0
    # Check for the inner list
    assert isinstance(result[0], list)
    assert len(result[0]) > 0
    # Check for the dictionary inside the inner list
    assert isinstance(result[0][0], dict)
    assert "generated_text" in result[0][0]
    assert result[0][0]["generated_text"].startswith(prompt)
    
    print(f"✅ Pipeline '{task_name}' succeeded.")

# @pytest.mark.asyncio
# async def test_pipeline_token_classification():
#     """
#     Tests the token-classification (NER) pipeline.
#     """
#     task_name = "token-classification"
    
#     # Use the 'inputs' key as discovered by our script.
#     # Provide a sentence with clear entities (a person and a location).
#     inputs = {
#         "inputs": "My name is Wolfgang and I live in Berlin."
#     }
    
#     payload = {
#         "task_name": task_name,
#         "inputs": inputs
#     }
    
#     print(f"\n--- Testing pipeline: {task_name} ---")
#     async with httpx.AsyncClient(timeout=60) as client:
#         response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
        
#     assert response.status_code == 200, f"API call for {task_name} failed: {response.text}"
    
#     response_data = response.json()
#     assert "result" in response_data
    
#     # Specific checks for the NER output structure
#     result = response_data["result"]
#     assert isinstance(result, list)
#     assert len(result) > 0 # We should find at least one entity
#     assert isinstance(result[0], dict)
#     assert "entity" in result[0]
#     assert "word" in result[0]
    
#     print(f"✅ Pipeline '{task_name}' succeeded.")


# @pytest.mark.asyncio
# async def test_pipeline_text_to_audio():
#     """
#     Tests the text-to-audio pipeline.
#     This is a complex model and may be slow.
#     """
#     task_name = "text-to-audio"
    
#     # Per our discovery script, the corrected parameter name is 'text_inputs'
#     inputs = {
#         "text_inputs": "Hello, this is a test of the text to audio pipeline."
#     }
    
#     payload = {
#         "task_name": task_name,
#         "inputs": inputs
#     }
    
#     print(f"\n--- Testing pipeline: {task_name} ---")
#     # This model is large and slow, so we'll use a longer timeout.
#     async with httpx.AsyncClient(timeout=120) as client:
#         response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
        
#     assert response.status_code == 200, f"API call for {task_name} failed: {response.text}"
    
#     response_data = response.json()
#     assert "result" in response_data
#     # Specific checks for the expected TTS output structure
#     assert isinstance(response_data["result"], dict)
#     assert "audio" in response_data["result"]
#     assert "sampling_rate" in response_data["result"]
    
#     print(f"✅ Pipeline '{task_name}' succeeded.")