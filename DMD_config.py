#!/bin/env python3

import xml.etree.ElementTree as ET
import base64
import argparse
from PIL import Image
import numpy as np
import io
import cv2
import os
import yaml
from pathlib import Path


def save_base64_image(image_data, output_path):
    """
    Save a base64 encoded image to a file with automatic format detection
    """
    try:
        # Handle data URL format if present
        if image_data.startswith("data:"):
            image_data = image_data.split(",")[1]

        # Decode base64 data
        image_bytes = base64.b64decode(image_data)

        # Open image with PIL for format detection
        img = Image.open(io.BytesIO(image_bytes))

        # Get base path without extension
        base_path = os.path.splitext(output_path)[0]

        # Create new path with detected format
        new_output_path = f"{base_path}.{img.format.lower()}"

        print(f"\nSaving image:")
        print(f"Detected format: {img.format}")
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        print(f"Output path: {new_output_path}")

        # Save image in its original format
        with open(new_output_path, "wb") as f:
            f.write(image_bytes)

        print(f"Successfully saved to {new_output_path}")
        return new_output_path

    except Exception as e:
        print(f"Error saving image {output_path}: {str(e)}")
        return None


def find_dmd_rectangle(img, display=None, config=None):
    """
    Find the DMD rectangle using grayscale detection and applying configuration constraints

    Args:
        img: Input image as numpy array
        display: Boolean flag to display detection results
        config: Configuration dictionary containing detection parameters

    Returns:
        Tuple (x, y, w, h) of detected rectangle or None if not found
    """
    # Convert to grayscale if image is in color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Create binary image through thresholding
    _, thresh = cv2.threshold(gray, 20, 64, 0)

    # Find contours in binary image
    contours, _ = cv2.findContours(thresh, 1, 2)

    # Initialize variables for largest valid contour
    largest_area = 0
    largest_cnt = None
    total_area = img.shape[0] * img.shape[1]

    # Process each detected contour
    for cnt in contours:
        # Approximate contour as polygon
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        # Skip if not a quadrilateral
        if len(approx) != 4:
            continue

        # Get contour area
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Apply configuration constraints if provided
        if config:
            # Check aspect ratio constraints
            aspect_ratio = w / h
            if aspect_ratio < config["min_aspect_ratio"] or aspect_ratio > config["max_aspect_ratio"]:
                continue

            # Check area ratio constraints
            area_ratio = area / total_area
            if area_ratio < config["min_area_ratio"] or area_ratio > config["max_area_ratio"]:
                continue

            # Check minimum dimension constraints
            if w < config["min_width"] or h < config["min_height"]:
                continue

        # Update largest valid contour if current is larger
        if area > largest_area:
            largest_area = area
            largest_cnt = cnt

    if isinstance(largest_cnt, np.ndarray):
        x, y, w, h = cv2.boundingRect(largest_cnt)
        if not display:
            return x, y, w, h

        # Add "DMD" label to visualization
        cv2.putText(img, "DMD", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Draw contour in green
        cv2.drawContours(img, [largest_cnt], -1, (0, 255, 0), 3)
        # Display resulting image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Image.fromarray(image).show()
        return x, y, w, h
    else:
        return None


def process_xml(xml_path, save_images=False, display=None, config_data=None):
    """
    Process DirectB2S XML file to extract and analyze images

    Args:
        xml_path: Path to the XML file
        save_images: Boolean flag to save extracted images
        display: Boolean flag to display detection results
        config_data: Configuration dictionary with screen and detection parameters
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        output_dir = os.path.dirname(xml_path)

        backglass_rect = None
        dmd_rect = None

        # Process both BackglassImage and DMDImage elements
        for elem_type in ["BackglassImage", "DMDImage"]:
            element = root.find(f".//{elem_type}")
            if element is not None and "Value" in element.attrib:
                print(f"\nAnalyzing {elem_type}...")

                # Save images if requested
                if save_images:
                    base_output_path = os.path.join(output_dir, f"{elem_type}")
                    saved_path = save_base64_image(element.get("Value"), base_output_path)
                    if saved_path:
                        print(f"Image saved as: {os.path.basename(saved_path)}")

                # Process base64 image data
                image_data = element.get("Value")
                if image_data.startswith("data:"):
                    image_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(image_data)

                # Convert to PIL Image for metadata
                pil_image = Image.open(io.BytesIO(image_bytes))
                img_width, img_height = pil_image.size
                print(f"Processing image:")
                print(f"Format: {pil_image.format}")
                print(f"Size: {img_width}x{img_height}")

                # Convert to OpenCV format for processing
                img_array = np.array(pil_image)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Find DMD rectangle using configuration parameters
                detection_config = config_data.get("defaults") if config_data else None
                rect = find_dmd_rectangle(img_array, display, detection_config)

                if rect:
                    x, y, w, h = rect
                    print(f"DMD rectangle found:")
                    print(f"Position: x={x}, y={y}")
                    print(f"Dimensions: {w}x{h} pixels")

                    if elem_type == "BackglassImage":
                        backglass_rect = (x, y, w, h, img_width, img_height)
                    else:
                        dmd_rect = (x, y, w, h, img_width, img_height)
                else:
                    print("No DMD rectangle found")

        # Generate INI configuration if config data is provided
        if config_data:
            ini_path = str(Path(xml_path).with_suffix(".ini"))
            generate_ini_config(backglass_rect, dmd_rect, config_data, ini_path)

    except ET.ParseError as e:
        print(f"XML parsing error: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def generate_ini_config(backglass_rect, dmd_rect, config_data, output_path):
    """
    Generate INI configuration file for PinMAME and FlexDMD setup

    Args:
        backglass_rect: Tuple containing detected DMD rectangle in backglass image
        dmd_rect: Tuple containing detected DMD rectangle in DMD image
        config_data: Configuration dictionary with screen parameters
        output_path: Path where INI file will be saved
    """
    screens = config_data["screens"]
    is_three_screens = screens["DMD"]["size_x"] > 0

    # Initialize INI content with standard settings
    ini_content = "[Standalone]\n"
    ini_content += "PinMAMEPath =\n"
    ini_content += "PinMAMEWindow = 1\n"

    # Calculate DMD window position and size based on available data
    if is_three_screens and dmd_rect:
        # Use detected DMD position from DMD image in 3-screen setup
        x, y, w, h, img_width, img_height = dmd_rect
        scale_x = screens["DMD"]["size_x"] / img_width
        scale_y = screens["DMD"]["size_y"] / img_height

        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        scaled_width = int(w * scale_x)
        scaled_height = int(h * scale_y)

        dmd_x = screens["Playfield"]["size_x"] + scaled_x
        dmd_y = scaled_y
        dmd_width = scaled_width
        dmd_height = scaled_height

        print("Using detected DMD position from DMDImage")

    elif is_three_screens:
        # Use default DMD values for 3-screen setup
        print("No DMD rectangle detected, using DMD screen default values")
        dmd_x = screens["Playfield"]["size_x"] + screens["DMD"]["default_x"]
        dmd_y = screens["DMD"]["default_y"]
        dmd_width = screens["DMD"]["default_width"]
        dmd_height = screens["DMD"]["default_height"]

    elif backglass_rect:
        # Use detected DMD position from backglass in 2-screen setup
        x, y, w, h, img_width, img_height = backglass_rect
        scale_x = screens["BackGlass"]["size_x"] / img_width
        scale_y = screens["BackGlass"]["size_y"] / img_height

        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        scaled_width = int(w * scale_x)
        scaled_height = int(h * scale_y)

        dmd_x = screens["Playfield"]["size_x"] + scaled_x
        dmd_y = scaled_y
        dmd_width = scaled_width
        dmd_height = scaled_height

        print("Using detected DMD position from BackglassImage")

    else:
        # Use default backglass values for 2-screen setup
        print("No rectangles detected, using BackGlass default values")
        dmd_x = screens["Playfield"]["size_x"] + screens["BackGlass"]["dmd_default_x"]
        dmd_y = screens["BackGlass"]["dmd_default_y"]
        dmd_width = screens["BackGlass"]["dmd_default_width"]
        dmd_height = screens["BackGlass"]["dmd_default_height"]

    # Configure PinMAME window settings
    ini_content += f"PinMAMEWindowX = {dmd_x}\n"
    ini_content += f"PinMAMEWindowY = {dmd_y}\n"
    ini_content += f"PinMAMEWindowWidth = {dmd_width}\n"
    ini_content += f"PinMAMEWindowHeight = {dmd_height}\n"
    ini_content += "PinMAMEWindowRotation =\n"

    # Configure FlexDMD window settings
    ini_content += "FlexDMDWindow = 1\n"
    ini_content += f"FlexDMDWindowX = {dmd_x}\n"
    ini_content += f"FlexDMDWindowY = {dmd_y}\n"
    ini_content += f"FlexDMDWindowWidth = {dmd_width}\n"
    ini_content += f"FlexDMDWindowHeight = {dmd_height}\n"

    # Configure B2S window settings
    ini_content += "B2SHideGrill =\n"
    ini_content += "B2SHideB2SDMD =\n"
    ini_content += "B2SHideB2SBackglass =\n"
    ini_content += "B2SHideDMD =\n"
    ini_content += "B2SDualMode =\n"
    ini_content += "B2SWindows = 1\n"

    # Configure B2S backglass window
    ini_content += f"B2SBackglassX = {screens['Playfield']['size_x'] + screens['DMD']['size_x']}\n"
    ini_content += "B2SBackglassY = 0\n"
    ini_content += f"B2SBackglassWidth = {screens['BackGlass']['size_x']}\n"
    ini_content += f"B2SBackglassHeight = {screens['BackGlass']['size_y']}\n"
    ini_content += "B2SBackglassRotation =\n"

    # Configure B2S DMD window
    ini_content += f"B2SDMDX = {screens['Playfield']['size_x']}\n"
    ini_content += "B2SDMDY = 0\n"
    ini_content += f"B2SDMDWidth = {screens['DMD']['size_x']}\n"
    ini_content += f"B2SDMDHeight = {screens['DMD']['size_y']}\n"
    ini_content += "B2SDMDRotation =\n"
    ini_content += "B2SDMDFlipY =\n"
    ini_content += "B2SPlugins =\n"

    # Save configuration to file
    with open(output_path, "w") as f:
        f.write(ini_content)
    print(f"Configuration saved to {os.path.basename(output_path)}")


def main():
    """
    Main entry point for the DMD Configuration Generator
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DMD Configuration Generator")
    parser.add_argument("files", nargs="+", help="DirectB2S files to process")
    parser.add_argument("-d", "--display", action="store_true", help="Display result of DMD placement")
    parser.add_argument("-s", "--save", action="store_true", help="Save all images found in DirectB2S")
    parser.add_argument("-c", "--config", default="DMD_config.yaml", help="Path to screen configuration YAML file")
    args = parser.parse_args()

    # Load screen configuration
    try:
        with open(args.config, "r") as f:
            config_data = yaml.safe_load(f)
            # Add debug output flag
            config_data["debug_output"] = True
    except Exception as e:
        print(f"Error loading configuration file: {str(e)}")
        return

    # Process each input file
    for xml_path in args.files:
        print(f"\nProcessing {xml_path}")
        process_xml(xml_path, args.save, args.display, config_data)


if __name__ == "__main__":
    main()
