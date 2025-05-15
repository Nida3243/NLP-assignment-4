# Required Libraries
# Make sure to install these using:
# pip install Pillow python-dotenv langchain streamlit

import os
import json

# Define the color code mapping along with their hex values
COLOR_CODE_MAP = {
    '0': {'name': 'Black', 'hex': '#000000'},
    '1': {'name': 'Brown', 'hex': '#A52A2A'},
    '2': {'name': 'Red', 'hex': '#FF0000'},
    '3': {'name': 'Orange', 'hex': '#FFA500'},
    '4': {'name': 'Yellow', 'hex': '#FFFF00'},
    '5': {'name': 'Green', 'hex': '#008000'},
    '6': {'name': 'Blue', 'hex': '#0000FF'},
    '7': {'name': 'Purple', 'hex': '#800080'},
    '8': {'name': 'Grey', 'hex': '#808080'},
    '9': {'name': 'White', 'hex': '#FFFFFF'}
}

# Function to extract the last digit before any alphabets
def extract_last_digit(wire_code):
    for char in reversed(wire_code):
        if char.isdigit():
            return char
        elif char.isalpha():
            continue
    return None

# Function to validate wire colors
def validate_wire_colors(wire_codes, assigned_colors):
    validation_results = []
    incorrect_colors = []

    for code, actual_color in zip(wire_codes, assigned_colors):
        last_digit = extract_last_digit(code)
        expected_color_info = COLOR_CODE_MAP.get(last_digit, None)

        if expected_color_info is not None:
            expected_color = expected_color_info['name']
            expected_hex = expected_color_info['hex']
            validation_results.append({
                'code': code,
                'last_digit': last_digit,
                'expected_color': expected_color,
                'expected_hex': expected_hex,
                'actual_color': actual_color,
                'status': 'Correct' if actual_color.lower() == expected_color.lower() else 'Incorrect'
            })
            if actual_color.lower() != expected_color.lower():
                incorrect_colors.append(f"{code} (Expected: {expected_color}, Actual: {actual_color})")

    return validation_results, incorrect_colors

# Example usage
if __name__ == "__main__":
    # Sample wire codes and their assigned colors for testing
    wire_codes = ["1234A", "5678B", "91011C"]
    assigned_colors = ["Red", "Green", "Blue"]  # Replace with actual colors assigned in the schematic

    # Validate wire colors
    validation_results, incorrect_colors = validate_wire_colors(wire_codes, assigned_colors)

    # Display results
    for validation in validation_results:
        print(f"Code: {validation['code']}, Last Digit: {validation['last_digit']}, "
              f"Expected Color: {validation['expected_color']} (Hex: {validation['expected_hex']}), "
              f"Actual Color: {validation['actual_color']}, Status: {validation['status']}")

    if incorrect_colors:
        print("\nSome wires have incorrect colors:")
        for color in incorrect_colors:
            print(color)
    else:
        print("\nAll wire colors are correct!")
