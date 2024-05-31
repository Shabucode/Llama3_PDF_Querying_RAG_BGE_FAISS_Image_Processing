# sc.py

import numpy as np
import gradio as gr
from PIL import Image
import seam_carving
import json
from llm import get_dimensions  # Import the function from llm.py


# dimensions=get_dimensions(marketplace="steamworks documentation", context="main capsule")
# d=json.loads(dimensions) 
# print(d)
# width_resizing = d[0]
# height_resizing =d[1]

# Content Aware Image resizing (seam carving) function to accept resizing values with image as input
def image_resizing(image, marketplace, context):
    dimensions = get_dimensions(marketplace, context)  # Get width and height from llm.py
    # Assuming dimensions is a JSON string, parse it to a dictionary
    #d = json.loads(dimensions)
    print(type(dimensions))
    # Access width and height from the dictionary
    try:
        width_resizing = dimensions["width"]
        height_resizing = dimensions["height"]
    except ValueError:
        return None, "Invalid dimensions received.", None
    print(width_resizing, height_resizing)
    src = np.array(image)  # Convert the input image to a numpy array
    src_h, src_w, _ = src.shape  # Assigning the variables src_h and src_w to the shape of the source image
    original_size = f"Original image size: {src_w}x{src_h}"  # Assigning source image size to the original_size variable

    try:
        dst = seam_carving.resize(
            src,  # source image (rgb or gray)
            size=(width_resizing, height_resizing),  # target size
            energy_mode="backward",  # choose from {backward, forward}
            order="width-first",  # choose from {width-first, height-first}
            keep_mask=None,  # object mask to protect from removal
        )
        dst_h, dst_w, _ = dst.shape  # Assigning the destination image values to the dst_h and dst_w
        resized_size = f"Resized image size: {dst_w}x{dst_h}"  # Assigning the resized destination image size to the resized_size variable
        return dst, original_size, resized_size  # return the destination image, source image size, and the resized destination image size
    except Exception as e:
        error_message = f"Error during resizing: {e}"
        return None, error_message, None



# Gradio Interface
demo = gr.Interface(
    fn=image_resizing,
    inputs=[
        gr.Image(), 
        gr.Dropdown(["steamworks documentation", "epic game store"], label="Select Marketplace"), 
        gr.Textbox(label="Context")
    ],  # it accepts image and the marketplace and context as input
    outputs=[
        gr.Image(), 
        gr.Textbox(label="Original Size"), 
        gr.Textbox(label="Resized Size")
    ],  # it outputs the resized image, and the source and destination image sizes
    live=False  # False for submit button and True for live results
)

demo.launch(share=True)
