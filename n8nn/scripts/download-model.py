from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "michelgi/LLaVA-proto"
FILENAME = "sklearn_model.joblib"

model = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
)