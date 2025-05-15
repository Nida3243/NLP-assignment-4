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

# model = AIGatewayLangchainChatOpenAI(
#     access_token=access_token, base_url=AI_GATEWAY_BASE_URL, model="gpt-4o-2024-05-13", deere_ai_gateway_registration_id="graphics-quality-check")

 
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

image_path_1 = r"""C:\Users\W4FGXUV\Downloads\Graphics_Quality_Check\Graphics_Quality_Check\Schematics\Defective\Screenshot 2025-01-31 133644.png"""

#Graphics Quality Check Prompt for Schematics
prompt_1 ='''You are a Graphics Quality Check Expert specializing in schematic validation. 
Your task is to thoroughly inspect the provided schematic image and verify its correctness based on the following three checkpoints:

Checkpoint 1: 
Extract the following parameters from the provided graphics in json format along with parameter and value:

DPI (Dots Per Inch): Determine the resolution of the image.
Callout Font: Identify the font type used for callouts.
Image Size: Extract the width and height of the image in pixels.

Expected Output Format:
Callouts: ["A1", "B2", "C3", "D4"]
DPI: 300Callout 
Font: ArialImage 
Size: Width: 1920px, Height: 1080px

Checkpoint 2: 
Legend and Component MatchingCross-check the legend (key) with the actual components in the schematic.Identify any missing components in the legend that are present
 in the schematic.Extract the matched list of legends and corresponding components.If any legend entry does not have a matching component or vice versa, report it as 
 missing.
 Expected Output Format:
 Matched Legends and Components:Resistor: R1, R2, R3Capacitor: C1, C2Diode: D1, D2IC: U1
 Missing Legends: Transformer, Inductor
 Missing Components: C3
 
Checkpoint 3: 
You are a validation expert for schematics. Your task is to verify whether the colors assigned to the borders of all wires in the schematic are correct according to the following list:

Color Code List:

Black - 0
Brown - 1
Red - 2
Orange - 3
Yellow - 4
Green - 5
Blue - 6
Purple - 7
Grey - 8
White - 9
Hex Codes for Colors:

BLACK-HEX: #231F20
BROWN-HEX: #CF8B2D
RED-HEX: #ED1846
ORANGE-HEX: #F58220
YELLOW-HEX: #FFF200
GREEN-HEX: #008C44
BLUE-HEX: #00C0F3
PURPLE-HEX: #524FA1
GREY-HEX: #BCBECO
WHITE-HEX: #FFFFFF
 Each wire code may contain alphanumeric characters or only numbers. Your responsibility is to check the last digit of each wire code (before any alphabetic character) to verify the assigned border color against the expected color from the color code list.


Validation Process:
Each wire in the schematic has an alphanumeric code (e.g., 41400, 4263E, 6715A).The last digit of the code determines the expected wire color.
Extract all wire codes and compare their actual colors with the expected colors.Provide a list of incorrectly assigned colors and highlight any missing colors from the provided standard list.
Expected Output Format:
Correct Wire Colors:

6571 (Brown)
Last Digit: 1 
Expected: Brown 
Actual: Brown →
Status: Correct

Incorrect Wire Colors:

0002 (Green) 
Last Digit: 2 
Expected: Red 
Actual: Green 
Status: Incorrect

0001 (Black) 
Last Digit: 1 
Expected: Brown 
Actual: Black 
Status: Incorrect

6506 (Red) 
Last Digit: 6 
Expected: Blue 
Actual: Red 
Status: Incorrect


Final DeliverableYour final report should include:
Extracted Graphics Parameters (Callouts, DPI, Callout Font, Image Size).Legend-to-Component Matching Report (Matched and Missing Legends and Components).
Wire Color Validation Report (Validate colours with colour code).
All discrepancies must be clearly highlighted, ensuring that no details are missed in the schematic validation process.

Follow a standard format for the report as shown below

Checkpoint 1: Extraction of Graphics Parameters

**Extracted Parameters:**
{
  "Callouts": ["A5505", "B5501", "B5506", "GND201", "B5109", "B5502", "B5503", "B5500", "R5603", "W0018", "W0008", "W0009", "W0010", "W0026", "W0028", "W0031"],
  "DPI": 300,
  "Callout Font": "Arial",
  "Image Size": {
    "Width": 1920,
    "Height": 1080
  }
}

Checkpoint 2: Legend and Component Matching

**Legend:**

- A5505: Engine Control Unit (ECU)
- B5501: Selective Catalytic Reduction (SCR) Supply Module
- B5506: Diesel Exhaust Fluid (DEF) Quality Sensor
- GND201: Battery Box Ground
- B5109: Diesel Particulate Filter (DPF) Differential Pressure Sensor
- B5502: NOx Sensor, Diesel Particulate Filter (DPF) Outlet
- B5503: NOx Sensor, Selective Catalytic Reduction (SCR) Outlet
- B5500: Tri Comp Inlet Humidity/Press/Temp Controller Area Network (CAN)
- R5603: CAN Terminator
- W0018, W0008, W0009, W0010, W0026, W0028, W0031: Wiring Connectors

**Matched Legends and Components:**

- **Matched Legends and Components:**
  - Engine Control Unit (ECU): A5505
  - Selective Catalytic Reduction (SCR) Supply Module: B5501
  - Diesel Exhaust Fluid (DEF) Quality Sensor: B5506
  - Battery Box Ground: GND201
  - Diesel Particulate Filter (DPF) Differential Pressure Sensor: B5109
  - NOx Sensor, Diesel Particulate Filter (DPF) Outlet: B5502
  - NOx Sensor, Selective Catalytic Reduction (SCR) Outlet: B5503
  - Tri Comp Inlet Humidity/Press/Temp Controller Area Network (CAN): B5500
  - CAN Terminator: R5603
  - Wiring Connectors: W0018, W0008, W0009, W0010, W0026, W0028, W0031

- **Missing Legends:**
  - None

- **Missing Components:**
  - None

Checkpoint 3: Wire Color Validation

**Extracted Wire Codes and Colors:**

1. 5305 (Black)
   - Last Digit: 5
   - Expected: Green
   - Actual: Black
   - **Status: Incorrect**

2. 5301 (Black)
   - Last Digit: 1
   - Expected: Brown
   - Actual: Black
   - **Status: Incorrect**

3. 5331 (Orange)
   - Last Digit: 1
   - Expected: Brown
   - Actual: Orange
   - **Status: Incorrect**

4. 5804 (Yellow)
   - Last Digit: 4
   - Expected: Yellow
   - Actual: Yellow
   - **Status: Correct**

5. 5803 (Yellow)
   - Last Digit: 3
   - Expected: Orange
   - Actual: Yellow
   - **Status: Incorrect**

6. 5805 (Green)
   - Last Digit: 5
   - Expected: Green
   - Actual: Green
   - **Status: Correct**

**Incorrect Wire Colors:**

- 5305 (Black)
  - Last Digit: 5
  - Expected: Green
  - Actual: Black
  - **Status: Incorrect**

- 5301 (Black)
  - Last Digit: 1
  - Expected: Brown
  - Actual: Black
  - **Status: Incorrect**

- 5331 (Orange)
  - Last Digit: 1
  - Expected: Brown
  - Actual: Orange
  - **Status: Incorrect**

- 5803 (Yellow)
  - Last Digit: 3
  - Expected: Orange
  - Actual: Yellow
  - **Status: Incorrect**

### Summary of Discrepancies

- **Graphics Parameters:** Extracted accurately.
- **Legend and Component Matching:** All legends and components are matched correctly.
- **Wire Color Validation:** Several wire colors do not match the expected standard color codes.

**Recommendations:**

- Correct the wire colors for the codes 5305, 5301, 5331, and 5803 to match the expected color standards.
- Ensure future schematics adhere to the color code standards for consistency and accuracy.
'''

result = process_image_with_prompt(image_path_1, prompt_1)
print(result)

# image_path_2 = r"""C:\Users\W4FGXUV\Downloads\Graphics_Quality_Check\Graphics_Quality_Check\Schematics\AfterTreatment.jpg"""

# prompt_2 ='''You are a Graphics Quality Check Expert specializing in graphics image validation. 
# Your task is to thoroughly inspect the provided schematic image and verify its correctness based on the following three checkpoints:

# Checkpoint 1: 
# Extraction of Graphics ParametersExtract the following key parameters from the schematic image and provide the output in JSON format, 
# ensuring accuracy in parameter values:Callout Labels: Extract all callout labels present in the schematic, including numbers, alphabets, or 
# combinations.
# DPI (Dots Per Inch): Determine the resolution of the image.
# Callout Font: Identify the font type used for callouts.
# Image Size: Extract the width and height of the image in pixels.
# Expected Output Format:
# Callouts: ["A1", "B2", "C3", "D4"]
# DPI: 300Callout 
# Font: ArialImage 
# Size: Width: 1920px, Height: 1080px'''

# result = process_image_with_prompt(image_path_1, prompt)
# print(result)