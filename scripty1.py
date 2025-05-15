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
    access_token=access_token, base_url=AI_GATEWAY_BASE_URL, model="gpt-4o-2024-05-13", deere_ai_gateway_registration_id="graphics-quality-check")

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
    if image.format not in ["PNG", "JPEG", "GIF", "WEBP"]:
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

# Example usage

image_path = r"C:\Users\W4FGXUV\Downloads\Graphics_Quality_Check\Graphics_Quality_Check\graphics\graphics\output_schematic.jpg"
# prompt = '''Extract the following parameters from the provided graphics in json format along with parameter and value:
#         1. callout
#         2. dpi
#         3. Callout Font
#         4. Image Size '''

#prompt = '''Match the legends with the actual componenets in the schematic image point out if anything is missing in the legend 
#         extract the matched legend and component list
#'''

prompt = '''Match the legends with the actual callouts in the image and let me know if any of the callouts are missing
'''
result = process_image_with_prompt(image_path, prompt)
print(result)
