# in tests/test_all_pipelines.py
import pytest
import httpx

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