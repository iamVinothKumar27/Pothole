import cv2
import numpy as np

def detect_cracks(image_path, pixels_per_meter=1000):  # Adjust pixels_per_meter based on your image scale
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to enhance crack detection
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours and calculate lengths
    result = img.copy()
    crack_lengths_meters = []
    
    for contour in contours:
        # Draw the contour
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        
        # Calculate contour length in meters
        perimeter_pixels = cv2.arcLength(contour, True)
        length_meters = perimeter_pixels / pixels_per_meter
        crack_lengths_meters.append(length_meters)
        
        # Add length text to image
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result, f'{length_meters:.3f}m', (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save results
    cv2.imwrite('gray_image.jpg', gray)
    cv2.imwrite('binary_image.jpg', binary)
    cv2.imwrite('crack_detection.jpg', result)
    
    return len(contours), crack_lengths_meters

def calibrate_pixels_per_meter(known_distance_meters, pixels_in_image):
    """Calculate pixels per meter using a known reference distance"""
    return pixels_in_image / known_distance_meters

if __name__ == "__main__":
    image_path = "D:\Project - 2\Crack detect\Testing\cracked-1.jpg"
    
    # Calibration example: if you know a 1-meter reference object is 1000 pixels in your image
    pixels_per_meter = calibrate_pixels_per_meter(1, 1000)  # Adjust these values based on your reference
    
    num_cracks, lengths = detect_cracks(image_path, pixels_per_meter)
    print(f"Number of potential cracks detected: {num_cracks}")
    print(f"Crack lengths (in meters): {[f'{length:.3f}' for length in lengths]}")
    if lengths:
        print(f"Total crack length: {sum(lengths):.3f} meters")
        print(f"Average crack length: {sum(lengths)/len(lengths):.3f} meters")
