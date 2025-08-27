# in tests/test_api.py
import pytest
import httpx

API_URL = "http://127.0.0.1:8000"

@pytest.mark.asyncio
async def test_get_tasks_endpoint():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/tasks")
        
        assert response.status_code == 200
        response_data = response.json()
        assert isinstance(response_data, dict)
        assert len(response_data) > 0
        print(f"\n✅ GET /tasks returned {len(response_data)} tasks.")


@pytest.mark.parametrize(
    "task_name, inputs",
    [
        (
            "text-classification",
            {"inputs": "This course is amazing!"}
        ),
        (
            "zero-shot-classification",
            {
                "sequences": "The new budget was announced by the government today.",
                "candidate_labels": ["sports", "politics", "technology"]
            }
        ),
        (
            "question-answering",
            {                
                "question": "What is the capital of Switzerland?",
                "context": "The capital of Switzerland is Bern."
            }
        )
    ]
)
@pytest.mark.asyncio
async def test_run_pipeline_endpoint(task_name, inputs):
    payload = {
        "task_name": task_name,
        "inputs": inputs
    }
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{API_URL}/run-pipeline/", json=payload)
        assert response.status_code == 200, f"API call failed with status {response.status_code}: {response.text}"
        response_data = response.json()
        assert "result" in response_data
        assert response_data["result"] is not None
        print(f"✅ POST /run-pipeline for '{task_name}' succeeded.")