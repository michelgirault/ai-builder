from langchain.tools import Tool
from custom_tools import StableDiffusionTool  # Assuming you saved your tool here

def register_custom_tools():
    stable_diffusion_tool = StableDiffusionTool(api_key="YOUR_API_KEY")
    return [Tool.from_function(func=stable_diffusion_tool.run, name="StableDiffusion", description="Generates images with Stable Diffusion")]