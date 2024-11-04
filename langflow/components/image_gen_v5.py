import os
import base64
import requests
import uuid
from datetime import datetime
from langflow.custom import Component
from langflow.inputs import StrInput
from langflow.io import MessageTextInput, Output
from langflow.field_typing import Text
from langflow.schema import Data
from langflow.schema.message import Message
from langflow.inputs import StrInput, MultilineInput, SecretStrInput, IntInput, DropdownInput
from langflow.template import Output, Input

class CustomComponent(Component):
    display_name = "Custom component test input output"
    description = "Use as a template to create your own component."
    icon = "custom_components"
    name = "TestComponent"
    inputs = [
        MessageTextInput(
            name="input_prompt",
            required=True,
            display_name="Input Prompt",
            placeholder="Enter text to analyze",
            input_types=["Message"]
        ),
        Input(
            name="input_apikey", 
            display_name="Input ApiKey",
            password="true",
            input_types=["Message"]
        ),
        Input(
            name="input_timoutduration",
            display_name="Timeout Duration",
            field_type="int",
            required=False,
            value=10,
        ),
        Input(
            name="output_directory",
            display_name="Output Directory",
            field_type="str",
            required=True,
            value="06186778-309d-42e6-a12b-85c5db9bf3e4",
        ),
    ]
    outputs = [
        Output(display_name="Output Message", name="output_message", method="build_output_message"),
    ]
        
    def build_output_message(self) -> Message:
        url = "http://194.247.183.148:4000/sdapi/v1/txt2img"
        headers = {
            "Authorization": f"Bearer {self.input_apikey}",
            "Content-Type": "application/json",
        }
        data = {
            "prompt": self.input_prompt,
            "num_images": 1,
            "image_height": 512,
            "image_width": 512,
            "sampler_name": "Euler",
            "token_merging_ratio": 0.2,
            "steps": 15,
            "override_settings": {"sd_model_checkpoint": "sd3_medium_incl_clips_t5xxlfp8"},
        }

        response = requests.post(url, headers=headers, json=data, timeout=self.input_timoutduration)
        response.raise_for_status()
        result = response.json()
        image_data = result["images"][0]
            
        # Generate a unique filename
        BASE_IMAGE_URL = "http://builder.apps.cluster01.lumimai.com/api/v1/files/images/"
        directory_base = "/data/config"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{len(os.listdir(self.output_directory)) + 1}.jpg"
        relative_path = os.path.join(self.output_directory, filename)
        full_path = os.path.join(directory_base, relative_path)
        full_url_path = BASE_IMAGE_URL+relative_path
        
            
        # Save the image
        with open(full_path, "wb") as image_file:
            image_file.write(base64.b64decode(image_data))
            
        return Message(
                text=full_url_path
            )