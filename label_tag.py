import os
import json
import base64
from PIL import Image
import io
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from src.encrypt import decrypt_data
from src.auth_helpers import get_access_token
from utils.chat_model import AIGatewayLangchainChatOpenAI
import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

# Load configuration from JSON and environment
with open('secret/url.json', 'r') as file:
    url_data = json.load(file)
AI_GATEWAY_BASE_URL = url_data.get("ai_gateway")
issuer_url = url_data.get("issuer_url")

# Load the .env file from the specified directory
load_dotenv('secret/.env')  # Specify the full path to the .env file

# Retrieve and decrypt the environment variables
loaded_fernet_key = os.getenv('FERNET_KEY').encode()
loaded_encrypted_client_id = os.getenv('ENCRYPTED_CLIENT_ID').encode()
loaded_encrypted_client_secret = os.getenv('ENCRYPTED_CLIENT_SECRET').encode()

client_id = decrypt_data(loaded_encrypted_client_id, loaded_fernet_key)
client_secret = decrypt_data(loaded_encrypted_client_secret, loaded_fernet_key)

access_token = get_access_token(client_id, client_secret, issuer_url)

model = AIGatewayLangchainChatOpenAI(
    access_token=access_token, base_url=AI_GATEWAY_BASE_URL, model="o1-2024-12-17", deere_ai_gateway_registration_id="graphics-quality-check")

def invoke_model(image_base64, prompt):
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ],
    )
    response = model.invoke([message])
    return response.content

def convert_image(image_path):
    image = Image.open(image_path)
    if image.format not in ["PNG", "JPEG", "GIF", "WEBP", "BMP"]:
        print(f"Unsupported image format: {image.format}")
        return None
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    image_data = buffered.getvalue()
    return base64.b64encode(image_data).decode("utf-8")

def process_image_with_prompt(image_path, prompt):
    image_base64 = convert_image(image_path)
    if image_base64 is None:
        return "Image conversion failed due to unsupported format."
    result = invoke_model(image_base64, prompt)
    return result

def identify_part_number(graphics_folder, actual_images_folder):
    # Loop through each graphics image for training
    for part_number in os.listdir(graphics_folder):
        part_folder_path = os.path.join(graphics_folder, part_number)
        if os.path.isdir(part_folder_path):
            graphics_images = [os.path.join(part_folder_path, img) for img in os.listdir(part_folder_path) if img.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
            # Here we could enhance the training process if needed by passing multiple images.

            for actual_image in os.listdir(actual_images_folder):
                actual_image_path = os.path.join(actual_images_folder, actual_image)
                if actual_image.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):  # Check for BMP files
                    prompt = f"""
                    Analyze the provided image of a machine part and identify its corresponding part number from the available graphics for {part_number}. 
                    If there are no exact matches, please provide the closest possible match along with a confidence level or similarity score. 
                    Include any relevant details that may help in understanding the match, such as distinguishing features or characteristics.
                    """
                    result = process_image_with_prompt(actual_image_path, prompt)
                    print(f"Result for {actual_image}: {result}")

# Set your folders paths here
graphics_folder = r"C:\Users\W4FGXUV\Downloads\Graphics_Quality_Check\Graphics_Quality_Check\dataset\dataset"
actual_images_folder = r"C:\Users\W4FGXUV\Downloads\Graphics_Quality_Check\Graphics_Quality_Check\Actual Part Images\Actual Part Images"

# Call the function to identify part numbers
identify_part_number(graphics_folder, actual_images_folder)


"""So i will give you example code which is developed for some other problem statement, and my problem statement, i want you to use the code
given and frame it for my problem statement, so my problem statement is i have few 12 folders with folder name as part__number and there are
approximately 150 graphics rendered images in each folder of those part_numbers, my goal is to identify the part_numberv of the images from 
actual machine photos which hae gone rusty or failed, so based on graphics images i want my model to learn the which part number is of which image 
and then identify those failed or faulty parts from actual image photo and give me part number 
"""

