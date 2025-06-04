import requests
import time
from typing import List, Optional, Dict, Any
from langchain_core.embeddings import Embeddings
from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.io import BoolInput, DictInput, DropdownInput, FloatInput, IntInput, MessageTextInput, SecretStrInput


class RunPodEmbeddings(Embeddings):
    """Custom embeddings class for RunPod API."""
    
    def __init__(
        self,
        api_key: str,
        endpoint_id: str,
        model: str = "BAAI/bge-small-en-v1.5",
        max_retries: int = 3,
        timeout: float = 60.0,
        chunk_size: int = 1000,
        **kwargs
    ):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to RunPod API with retry logic."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        for attempt in range(self.max_retries):
            try:
                # Submit the job
                response = requests.post(
                    f"{self.base_url}/run",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                
                if "id" in result:
                    # Poll for results
                    job_id = result["id"]
                    return self._poll_for_result(job_id, headers)
                else:
                    # Synchronous response
                    return result
                    
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"RunPod API request failed after {self.max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("Max retries exceeded")
    
    def _poll_for_result(self, job_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Poll RunPod API for job completion."""
        max_poll_attempts = 30  # 5 minutes with 10 second intervals
        
        for _ in range(max_poll_attempts):
            try:
                status_response = requests.get(
                    f"{self.base_url}/status/{job_id}",
                    headers=headers,
                    timeout=10
                )
                status_response.raise_for_status()
                status_result = status_response.json()
                
                if status_result.get("status") == "COMPLETED":
                    return status_result.get("output", {})
                elif status_result.get("status") == "FAILED":
                    raise Exception(f"RunPod job failed: {status_result.get('error', 'Unknown error')}")
                
                time.sleep(10)  # Wait 10 seconds before next poll
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to poll job status: {str(e)}")
        
        raise Exception("Job polling timeout - job did not complete within expected time")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        all_embeddings = []
        
        # Process in chunks
        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i:i + self.chunk_size]
            
            for text in chunk:
                payload = {
                    "input": {
                        "model": self.model,
                        "input": text
                    }
                }
                
                result = self._make_request(payload)
                
                # Extract embedding from result
                # Adjust this based on actual RunPod response format
                if "embedding" in result:
                    embedding = result["embedding"]
                elif "embeddings" in result:
                    embedding = result["embeddings"][0] if isinstance(result["embeddings"], list) else result["embeddings"]
                elif "data" in result and len(result["data"]) > 0:
                    embedding = result["data"][0].get("embedding", [])
                else:
                    raise Exception(f"Unexpected response format from RunPod API: {result}")
                
                all_embeddings.append(embedding)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        payload = {
            "input": {
                "model": self.model,
                "input": text
            }
        }
        
        result = self._make_request(payload)
        
        # Extract embedding from result
        if "embedding" in result:
            return result["embedding"]
        elif "embeddings" in result:
            return result["embeddings"][0] if isinstance(result["embeddings"], list) else result["embeddings"]
        elif "data" in result and len(result["data"]) > 0:
            return result["data"][0].get("embedding", [])
        else:
            raise Exception(f"Unexpected response format from RunPod API: {result}")


class RunPodEmbeddingsComponent(LCEmbeddingsModel):
    display_name = "RunPod Embeddings"
    description = "Generate embeddings using RunPod API with custom models."
    icon = "cloud"
    name = "RunPodEmbeddings"

    inputs = [
        SecretStrInput(
            name="runpod_api_key", 
            display_name="RunPod API Key", 
            value="RUNPOD_API_KEY", 
            required=True,
            info="Your RunPod API key for authentication."
        ),
        MessageTextInput(
            name="endpoint_id", 
            display_name="Endpoint ID", 
            value="988yije0lagyo3",
            required=True,
            info="RunPod endpoint ID (e.g., 988yije0lagyo3)."
        ),
        MessageTextInput(
            name="model",
            display_name="Model",
            value="BAAI/bge-small-en-v1.5",
            required=True,
            info="The embedding model to use (e.g., BAAI/bge-small-en-v1.5)."
        ),
        IntInput(
            name="chunk_size", 
            display_name="Chunk Size", 
            advanced=True, 
            value=100,
            info="Number of texts to process in parallel."
        ),
        IntInput(
            name="max_retries", 
            display_name="Max Retries", 
            value=3, 
            advanced=True,
            info="Maximum number of retry attempts for failed requests."
        ),
        FloatInput(
            name="request_timeout", 
            display_name="Request Timeout", 
            advanced=True, 
            value=60.0,
            info="Timeout for API requests in seconds."
        ),
        BoolInput(
            name="show_progress_bar", 
            display_name="Show Progress Bar", 
            advanced=True,
            value=False,
            info="Whether to show progress bar during processing."
        ),
        DictInput(
            name="additional_headers",
            display_name="Additional Headers",
            advanced=True,
            info="Additional headers to include in API requests."
        ),
    ]

    def build_embeddings(self) -> Embeddings:
        return RunPodEmbeddings(
            api_key=self.runpod_api_key,
            endpoint_id=self.endpoint_id,
            model=self.model,
            chunk_size=self.chunk_size,
            max_retries=self.max_retries,
            timeout=self.request_timeout or 60.0,
        )