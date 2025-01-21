import os
import io
import logging
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
from langflow.schema.content_block import ContentBlock
from langflow.schema.content_types import TextContent, MediaContent
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
            value=10,
        ),
        Input(
            name="output_directory",
            display_name="Output Directory",
            field_type="str",
            required=True,
        ),
    ]
    outputs = [
        Output(display_name="Output Message", name="output_message", method="build_output_message"),
    ]
        
    def build_output_message(self) -> Message:
        url = self.input_base_url
        headers = {
            "Authorization": f"Bearer {self.input_apikey}",
            "Content-Type": "application/json",
        }
        data = {
            "prompt": self.input_prompt,
            "negative_prompt": "cartoon",
            "height": self.input_height,
            "width": self.input_width,
            "num_inference_steps": self.input_steps,
            "guidance_scale": 3.5
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=self.input_timoutduration)
        response.raise_for_status()
        result = response.json()
        image_data = result
        #define variable
        ROOTDIR = os.environ['LANGFLOW_CONFIG_DIR']

        #create repository if non existant
        os.makedirs(self.output_directory, exist_ok=True)

        # Generate a unique filename
        BASE_IMAGE_URL = "http://builder.apps.lumimai.com:8080/api/v1/files/images/"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{len(os.listdir(self.output_directory)) + 1}.jpg"
 
        relative_path = os.path.join(filename)
        full_path = os.path.join(ROOTDIR, self.output_directory, relative_path)
        full_url_path = os.path.join(BASE_IMAGE_URL, self.output_directory, relative_path)
        # test the variable
        print(full_url_path)
        print(full_path)

        # Ensure the directory for the file exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
            
        # Save the image
        with open(full_path, "wb") as image_file:
            image_file.write(base64.b64decode(image_data))
        #display in chat the image output / link
        image_to_chat = Message(
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
        