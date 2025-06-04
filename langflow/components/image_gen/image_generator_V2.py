import os
import logging
import base64
import requests
from datetime import datetime
from langflow.custom import Component
from langflow.io import MessageTextInput
from langflow.schema.message import Message
from langflow.schema.content_block import ContentBlock
from langflow.schema.content_types import MediaContent
from langflow.inputs import Input, DropdownInput
from langflow.template import Output

class CustomComponent(Component):
    display_name = "Image generator"
    description = "Custom component to create image using RunPod."
    icon = "custom_components"
    name = "TestComponent"
    inputs = [
        MessageTextInput(
            name="input_prompt",
            required=True,
            display_name="Input Prompt",
            placeholder="Enter prompt for image generation",
            input_types=["Message"]
        ),
        Input(
            name="input_steps", 
            display_name="Steps",
            required=True,
            input_types=["Message"]
        ),
        Input(
            name="input_base_url", 
            display_name="Base Url",
            required=True,
            input_types=["Message"]
        ),
        DropdownInput(
            name="input_height",
            display_name="height",
            required=True,
            options=["512", "1024"],
            input_types=["Message"]
        ),
        DropdownInput(
            name="input_width",
            display_name="width",
            required=True,
            options=["512", "1024"],
            input_types=["Message"]
        ),
        Input(
            name="input_apikey", 
            display_name="Api Key",
            password="true",
            input_types=["Message"]
        ),
        Input(
            name="input_timoutduration",
            display_name="Timeout Duration",
            field_type="int",
            required=False,
            value=120,
        ),
        Input(
            name="output_directory",
            display_name="Current flowID",
            field_type="str",
            required=True,
        ),
    ]
    outputs = [
        Output(display_name="Output Message", name="output_message", method="build_output_message"),
    ]
        
    def build_output_message(self) -> Message:
        # Prepare the URL - ensure it uses runsync
        url = self.input_base_url.strip()
        if 'runsync' not in url:
            url = url.replace('/run', '/runsync')
            if not url.endswith('/runsync'):
                url = url.rstrip('/') + '/runsync'
        
        # Prepare the request
        headers = {
            "Authorization": f"Bearer {self.input_apikey}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": {
                "prompt": self.input_prompt,
                "negative_prompt": "no toon",
                "height": int(self.input_height),
                "width": int(self.input_width),
                "num_inference_steps": int(self.input_steps),
                "guidance_scale": 3.5
            }
        }
        
        # Send the request to the API
        response = requests.post(
            url, 
            headers=headers, 
            json=payload, 
            timeout=self.input_timoutduration
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract image data from the expected location in the response
        image_data = None
        if "output" in result and isinstance(result["output"], dict):
            output = result["output"]
            if "data" in output and isinstance(output["data"], dict):
                data = output["data"]
                if "image" in data and isinstance(data["image"], str):
                    image_data = data["image"]
        
        # If not found in the expected location, try alternatives
        if not image_data or len(image_data) < 100:
            if "image" in result:
                image_data = result["image"]
            elif "output" in result and isinstance(result["output"], dict):
                if "image" in result["output"]:
                    image_data = result["output"]["image"]
        
        # Save the image
        ROOTDIR = os.environ['LANGFLOW_CONFIG_DIR']
        LANGFLOW_BASE_URL = os.environ['LANGFLOW_BASE_URL']
        
        os.makedirs(self.output_directory, exist_ok=True)
        
        BASE_IMAGE_URL = os.path.join(LANGFLOW_BASE_URL, "api/v1/files/images/")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.jpg"
        
        relative_path = os.path.join(filename)
        full_path = os.path.join(ROOTDIR, self.output_directory, relative_path)
        full_url_path = os.path.join(BASE_IMAGE_URL, self.output_directory, relative_path)
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, "wb") as image_file:
            image_file.write(base64.b64decode(image_data))
        
        # Return the image
        image_to_chat = Message(
            text=full_url_path,
            sender="image_gen",
            sender_name="Image",
            content_blocks=[
                ContentBlock(
                    title="Media Block",
                    contents=[
                        MediaContent(type="media", urls=[full_url_path])
                    ]
                )
            ],
        )
        return image_to_chat