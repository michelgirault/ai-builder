from langflow import CustomComponent

class CustomModelComponent(CustomComponent):
    display_name: str = "Custom Model"
    description: str = "Custom LLM model."

    def build_config(self):
        return {
            "model_id": {
                "display_name": "Model Id",
                "options": [
                    "llava-hf/llava-1.5-7b-hf",  # Add your custom model name here
                    # Add more custom model names if needed
                ],
            },
            "credentials_profile_name": {"display_name": "Credentials Profile Name"},
        }