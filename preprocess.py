import numpy as np
import cv2

def contour_coordinates(contour):
    '''
    Returns x-coordinate of a contour's centroid.
    Only returns coordinates for contours with area > 10.
    '''
    try:
        area = cv2.contourArea(contour)
        if area > 10:
            M = cv2.moments(contour)
            if M['m00'] != 0:  # Avoid division by zero error
                return int(M['m10'] / M['m00'])
    except Exception as e:
        print(f"Error in contour_coordinates: {e}")
    return 0  # Return 0 instead of None to avoid potential None handling issues

def drawSquare(image):
    '''
    Draws a square around the found digits.
    If the input image is already a square, returns as-is.
    Otherwise, resizes the image to a square.
    '''
    try:
        # Ensure image is 2D
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        b = [0, 0, 0]  # Black padding
        height, width = image.shape[0], image.shape[1]
        
        if height == width:
            return image  # No resize if already square
        
        # Resize to make the smaller dimension match the larger
        max_dim = max(height, width)
        d_size = cv2.resize(image, (max_dim, max_dim), interpolation=cv2.INTER_CUBIC)
        
        # If height > width, pad width
        if height > width:
            padding = (height - width) // 2
            return cv2.copyMakeBorder(d_size, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=b)
        else:
            padding = (width - height) // 2
            return cv2.copyMakeBorder(d_size, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=b)
    
    except Exception as e:
        print(f"Error in drawSquare: {e}")
        return image  # Return original image if processing fails

def resize(image, dim):
    '''
    Resizes the image to the specified square dimension `dim` with padding to keep it centered.
    '''
    try:
        # Ensure image is 2D
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        b = [0, 0, 0]  # Black padding
        dim = dim - 4  # Adjust the dimension to leave room for padding
        height, width = image.shape[:2]
        
        # Calculate resize ratio
        r = float(dim) / max(width, height)
        resized = cv2.resize(image, (int(width * r), int(height * r)), interpolation=cv2.INTER_AREA)
        
        h, w = resized.shape[:2]
        
        # Create a square canvas
        square = np.zeros((dim, dim), dtype=np.uint8)
        
        # Calculate start positions to center the image
        start_x = (dim - w) // 2
        start_y = (dim - h) // 2
        
        # Place resized image in the center of the square
        square[start_y:start_y+h, start_x:start_x+w] = resized
        
        # Adding 2-pixel padding around the final resized image
        return cv2.copyMakeBorder(square, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=b)
    
    except Exception as e:
        print(f"Error in resize: {e}")
        # Return a blank image of specified dimension if processing fails
        return np.zeros((dim, dim), dtype=np.uint8)

# Test the functions
def main():
    # Create a test image
    test_image = np.zeros((50, 30), dtype=np.uint8)
    cv2.rectangle(test_image, (10, 10), (20, 40), 255, -1)
    
    print("Original image shape:", test_image.shape)
    
    # Test drawSquare
    squared_image = drawSquare(test_image)
    print("Squared image shape:", squared_image.shape)
    
    # Test resize
    resized_image = resize(test_image, 20)
    print("Resized image shape:", resized_image.shape)

if __name__ == "__main__":
    main()
