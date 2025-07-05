from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

access_token = ""  # Replace with your own token

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    use_auth_token=access_token,
    torch_dtype=torch.float16
).to("cuda")

def generate_sd_image(prompt):
    image = pipe(prompt).images[0]
    return image

gr.Interface(
    fn=generate_sd_image,
    inputs=gr.Textbox(placeholder="A panda playing guitar under moonlight...", label="Prompt"),
    outputs="image",
    title="🖼️ Text To Image Generator",
    description="Generates images using Stable Diffusion. GPU required.",
).launch(share=True)
