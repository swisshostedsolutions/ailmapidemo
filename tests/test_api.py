import pytest
import httpx

API_URL = "http://127.0.0.1:8000"

@pytest.mark.asyncio
async def test_get_tasks_endpoint():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/tasks")
        assert response.status_code == 200
        assert len(response.json()) > 0

@pytest.mark.parametrize(
    "task_name, inputs",
    [
        ("text-classification", {"inputs": "This is a great movie!"}),
        ("zero-shot-classification", {"sequences": "The new budget was announced.", "candidate_labels": ["politics"]}),
        ("question-answering", {"question": "What is the capital of Switzerland?", "context": "The capital is Bern."}),
        ("document-question-answering", {"image": "test_assets/sample.jpg", "question": "What is the invoice number?"}),
        ("visual-question-answering", {"image": "test_assets/sample.jpg", "question": "What is in this image?"})
    ]
)
@pytest.mark.asyncio
async def test_run_pipeline_endpoint(task_name, inputs):
    payload = {"task_name": task_name, "inputs": inputs}
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
    
    assert response.status_code == 200, f"API call for {task_name} failed: {response.text}"
    response_data = response.json()
    assert "result" in response_data
    assert response_data["result"] is not None