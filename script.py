import os
import json
import re
import numpy as np
import base64
from openai import OpenAI
import io
from PIL import Image

client = OpenAI()

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

image_path = "./apples.png"

def get_gpt4_detections(image_path, classname):
    image = Image.open(image_path)
    base64_image = encode_image_to_base64(image)

    QUERY = """You are performing object detection on the provided image. 
    You are looking for {classname}s. 
    Return a JSON object with `x` (center y-coordinate), `y` (center x-coordinate), `width` and `height` attributes of boundings boxes drawn around each {classname} in the image. 
    Normalize these values to be between 0 and 1.
    
    Ex 1. An example output for an image with 3 `bananas` detected looks like this:

    [
        {
            "x": 0.32,
            "y": 0.55,
            "width": 0.25,
            "height": 0.35
        },
        {
            "x": 0.68,
            "y": 0.55,
            "width": 0.25,
            "height": 0.35
        },
        {
            "x": 0.5,
            "y": 0.35,
            "width": 0.26,
            "height": 0.34
        }
    ]

    Ex 2. An example output for an image with 1 `hat` detected looks like this:

    [
        {
            "x": 0.6,
            "y": 0.32,
            "width": 0.14,
            "height": 0.45
        }
    ]

    Ex 3. An example output for an image with 0 `bird` objects detected looks like this:

    []
    
    """

    response = client.chat.completions.create(
    model="gpt-4o",
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": QUERY,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
    max_tokens=300,
    )

    response_msg = response.choices[0].message.content

    json_start = response_msg.find("```json")
    json_end = response_msg.find("```", json_start + 6)
    json_content = response_msg[json_start + 7:json_end].strip()

    print("BERFORE: ", response_msg)
    print("HIIII: ", json_content)

    return json_content


def visualize_detection(detections, image_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    detections = json.loads(detections)
    img_width, img_height = image.size

    for detection in detections:
        x_center = detection['x'] * img_width
        y_center = detection['y'] * img_height
        width = detection['width'] * img_width
        height = detection['height'] * img_height
        x = x_center - width / 2
        y = y_center - height / 2

        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.savefig("result.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)


bounding_boxes = get_gpt4_detections(image_path, "apple")

visualize_detection(bounding_boxes, image_path)
