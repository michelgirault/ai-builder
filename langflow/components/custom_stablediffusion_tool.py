# from langflow.field_typing import Data
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
            required="False",
            value="10",

        ),
    ]

    outputs = [
        Output(display_name="Output data", name="output_data", method="build_output_data"),
    ]
        
    def build_output_data(self) -> Data:

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
            "steps": 20,
            "override_settings": {"sd_model_checkpoint": "sd3_medium_incl_clips"},
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=self.input_timoutduration)
            response.raise_for_status()
            result = response.json()
            test_response = result["images"][0]
            return result["images"][0]  # Return the base64 string
        except requests.exceptions.RequestException as e:
            raise Exception(f"Image generation failed: {str(e)}")
        return data()
        

        
