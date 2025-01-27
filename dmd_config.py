#!/bin/env python3

import xml.etree.ElementTree as ET
import base64
import sys
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
    Save a base64 encoded image to a file
    
    Args:
        image_data: Base64 encoded image data
        output_path: Path where to save the image
    """
    try:
        if image_data.startswith('data:image'):
            # Remove the data URL prefix if present
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        print(f"Saved {output_path}")
    except Exception as e:
        print(f"Error saving image {output_path}: {str(e)}")

def calculate_screen_positions(screens):
    """
    Calculate screen positions based on their IDs.
    Windows joins screens in ID order, so we need to sum up widths
    of all screens that come before in ID sequence.
    """
    # Sort screens by ID
    sorted_screens = sorted(screens.items(), key=lambda x: x[1]['id'])
    
    # Calculate x position for each screen based on previous screens
    positions = {}
    current_x = 0
    
    for screen_name, screen_data in sorted_screens:
        positions[screen_name] = {
            'x': current_x,
            'width': screen_data['size_x']
        }
        current_x += screen_data['size_x']
    
    return positions

def generate_ini_config(backglass_rect, dmd_rect, config_data, output_path):
    """
    Generate the ini configuration file based on detected rectangles and screen config.
    Screen positions are based on Windows screen IDs.
    """
    screens = config_data['screens']
    is_three_screens = screens['DMD']['size_x'] > 0
    
    # Calculate screen positions based on IDs
    screen_positions = calculate_screen_positions(screens)
    
    ini_content = "[Standalone]\n"
    
    # PinMAME Settings
    ini_content += "PinMAMEPath =\n"
    ini_content += "PinMAMEWindow = 1\n"
    
    # Calculate DMD position based on mode and detected rectangle
    if is_three_screens and dmd_rect:
        # Calculate scaling factors for DMD
        scale_x = screens['DMD']['size_x'] / dmd_rect['img_width']
        scale_y = screens['DMD']['size_y'] / dmd_rect['img_height']
        
        # Scale positions and dimensions
        scaled_x = int(dmd_rect['x'] * scale_x)
        scaled_y = int(dmd_rect['y'] * scale_y)
        scaled_width = int(dmd_rect['width'] * scale_x)
        scaled_height = int(dmd_rect['height'] * scale_y)
        
        # 3-screen mode: use scaled DMDImage rectangle
        pinmame_x = screen_positions['DMD']['x'] + scaled_x
        pinmame_y = scaled_y
        pinmame_width = scaled_width
        pinmame_height = scaled_height
    elif backglass_rect:
        # Similar scaling for BackglassImage in 2-screen mode
        scale_x = screens['BackGlass']['size_x'] / backglass_rect['img_width']
        scale_y = screens['BackGlass']['size_y'] / backglass_rect['img_height']
        
        scaled_x = int(backglass_rect['x'] * scale_x)
        scaled_y = int(backglass_rect['y'] * scale_y)
        scaled_width = int(backglass_rect['width'] * scale_x)
        scaled_height = int(backglass_rect['height'] * scale_y)
        
        pinmame_x = screen_positions['BackGlass']['x'] + scaled_x
        pinmame_y = scaled_y
        pinmame_width = scaled_width
        pinmame_height = scaled_height
    
    # Set PinMAME window position and size
    ini_content += f"PinMAMEWindowX = {pinmame_x}\n"
    ini_content += f"PinMAMEWindowY = {pinmame_y}\n"
    ini_content += f"PinMAMEWindowWidth = {pinmame_width}\n"
    ini_content += f"PinMAMEWindowHeight = {pinmame_height}\n"
    ini_content += "PinMAMEWindowRotation =\n"
    
    # FlexDMD Settings (same as PinMAME)
    ini_content += "FlexDMDWindow = 1\n"
    ini_content += f"FlexDMDWindowX = {pinmame_x}\n"
    ini_content += f"FlexDMDWindowY = {pinmame_y}\n"
    ini_content += f"FlexDMDWindowWidth = {pinmame_width}\n"
    ini_content += f"FlexDMDWindowHeight = {pinmame_height}\n"
    
    # B2S Settings
    ini_content += "B2SHideGrill =\n"
    ini_content += "B2SHideB2SDMD =\n"
    ini_content += "B2SHideB2SBackglass =\n"
    ini_content += "B2SHideDMD =\n"
    ini_content += "B2SDualMode =\n"
    ini_content += "B2SWindows = 1\n"
    
    # Backglass position based on ID
    ini_content += f"B2SBackglassX = {screen_positions['BackGlass']['x']}\n"
    ini_content += "B2SBackglassY = 0\n"
    ini_content += f"B2SBackglassWidth = {screens['BackGlass']['size_x']}\n"
    ini_content += f"B2SBackglassHeight = {screens['BackGlass']['size_y']}\n"
    ini_content += "B2SBackglassRotation =\n"
    
    # B2S DMD settings based on ID
    ini_content += f"B2SDMDX = {screen_positions['DMD']['x']}\n"
    ini_content += "B2SDMDY = 0\n"
    ini_content += f"B2SDMDWidth = {screens['DMD']['size_x']}\n"
    ini_content += f"B2SDMDHeight = {screens['DMD']['size_y']}\n"
    ini_content += "B2SDMDRotation =\n"
    ini_content += "B2SDMDFlipY =\n"
    ini_content += "B2SPlugins =\n"
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(ini_content)
    print(f"Configuration saved to {os.path.basename(output_path)}")

def find_uniform_rectangle(image_data):
    """
    Find the largest uniform colored rectangle in an image.
    Ensures the rectangle stays within any borders by focusing on the inner area.
    """
    try:
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array and handle different color formats
        img_array = np.array(img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
        img_height, img_width = img_array.shape[:2]
        
        # Calculate gradients to detect color changes
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Create mask for uniform areas
        uniform_mask = (gradient_magnitude < 1).astype(np.uint8) * 255
        
        # Apply morphological operations to eliminate border effects
        kernel = np.ones((3,3), np.uint8)
        eroded_mask = cv2.erode(uniform_mask, kernel, iterations=5)
        
        # Find contours in the eroded mask
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process all contours to find valid rectangles
        rectangles = []
        for contour in contours:
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small contours
            if w < 10 or h < 5:
                continue
                
            # Get the region and check if it's uniform
            roi = eroded_mask[y:y+h, x:x+w]
            if roi.size == 0:
                continue
                
            # Calculate fill ratio of the rectangle
            fill_ratio = np.sum(roi == 255) / (w * h)
            
            # Only consider rectangles that are mostly filled (uniform)
            if fill_ratio > 0.95:  # 95% uniform
                area_ratio = (w * h) / (img_width * img_height)
                if 0.00001 <= area_ratio <= 0.75:
                    rectangles.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': w * h
                    })
        
        # Return the largest valid rectangle found
        if rectangles:
            best_rect = max(rectangles, key=lambda r: r['area'])
            return {
                'x': best_rect['x'],
                'y': best_rect['y'],
                'width': best_rect['width'],
                'height': best_rect['height'],
                'img_width': img_width,
                'img_height': img_height
            }
        
        return None
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return None

def process_xml(xml_path, save_images=False, config_data=None):
    """
    Process an XML file to find uniform rectangles in its images and generate ini config.
    
    Args:
        xml_path: Path to the XML file to process
        save_images: Boolean, whether to save images or not
        config_data: Dictionary with screen configurations
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        output_dir = os.path.dirname(xml_path)
        
        backglass_rect = None
        dmd_rect = None
        
        # Process BackglassImage and DMDImage
        for elem_type in ["BackglassImage", "DMDImage"]:
            element = root.find(f".//{elem_type}")
            if element is not None and 'Value' in element.attrib:
                print(f"\nAnalyzing {elem_type}...")
                
                # Save image if requested
                if save_images:
                    output_path = os.path.join(output_dir, f"{elem_type}.png")
                    save_base64_image(element.get('Value'), output_path)
                
                # Find uniform rectangle
                rect = find_uniform_rectangle(element.get('Value'))
                if rect:
                    print(f"Uniform rectangle found:")
                    print(f"Position: x={rect['x']}, y={rect['y']}")
                    print(f"Dimensions: {rect['width']}x{rect['height']} pixels")
                    print(f"Image dimensions: {rect['img_width']}x{rect['img_height']} pixels")
                    
                    if elem_type == "BackglassImage":
                        backglass_rect = rect
                    else:
                        dmd_rect = rect
                else:
                    print("No uniform rectangle found")
        
        # Save other images if requested
        if save_images:
            thumb = root.find(".//ThumbnailImage")
            if thumb is not None and 'Value' in thumb.attrib:
                output_path = os.path.join(output_dir, "ThumbnailImage.png")
                save_base64_image(thumb.get('Value'), output_path)
            
            bulb = root.find(".//Illumination/Bulb")
            if bulb is not None and 'Image' in bulb.attrib:
                output_path = os.path.join(output_dir, "Bulb.png")
                save_base64_image(bulb.get('Image'), output_path)
        
        # Generate ini configuration if config data is provided
        if config_data:
            ini_path = str(Path(xml_path).with_suffix('.ini'))
            generate_ini_config(backglass_rect, dmd_rect, config_data, ini_path)

    except ET.ParseError as e:
        print(f"XML parsing error: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='DMD Configuration Generator')
    parser.add_argument('files', nargs='+', help='DirectB2S files to process')
    parser.add_argument('-s', '--save', action='store_true', help='Save all images found in DirectB2S')
    parser.add_argument('-c', '--config', default='dmd_config.yaml', help='Path to screen configuration YAML file')
    args = parser.parse_args()
    
    # Load screen configuration
    try:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration file: {str(e)}")
        return
    
    for xml_path in args.files:
        print(f"\nProcessing {xml_path}")
        process_xml(xml_path, args.save, config_data)

if __name__ == "__main__":
    main()
