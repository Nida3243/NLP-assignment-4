import os
import json
import base64
import io
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.encrypt import decrypt_data
from src.auth_helpers import get_access_token
from utils.chat_model import AIGatewayLangchainChatOpenAI

# Load configuration from JSON and environment
with open('secret/url.json', 'r') as file:
    url_data = json.load(file)
AI_GATEWAY_BASE_URL = url_data.get("ai_gateway")
issuer_url = url_data.get("issuer_url")

# Load the .env file from the specified directory
load_dotenv('secret/.env')

# Retrieve and decrypt the environment variables
loaded_fernet_key = os.getenv('FERNET_KEY').encode()
loaded_encrypted_client_id = os.getenv('ENCRYPTED_CLIENT_ID').encode()
loaded_encrypted_client_secret = os.getenv('ENCRYPTED_CLIENT_SECRET').encode()

client_id = decrypt_data(loaded_encrypted_client_id, loaded_fernet_key)
client_secret = decrypt_data(loaded_encrypted_client_secret, loaded_fernet_key)

access_token = get_access_token(client_id, client_secret, issuer_url)

model = AIGatewayLangchainChatOpenAI(
    access_token=access_token, base_url=AI_GATEWAY_BASE_URL, model="gpt-4o-2024-05-13", deere_ai_gateway_registration_id="graphics-quality-check")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

# Create the uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Check if the uploaded file is a valid image
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Your existing image processing functions
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            prompt = request.form.get('prompt')
            result = process_image_with_prompt(file_path, prompt)
            return render_template('result.html', result=result, image_url=file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
