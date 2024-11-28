import numpy as np
import cv2
from preprocess import drawSquare, resize
from basic_knn import classifier_knn  # Import trained KNN model

def contour_coordinates(contour):
    '''Returns x-coordinate of a contour'''
    (x, y, w, h) = cv2.boundingRect(contour)
    return x

# Load the image
image = cv2.imread('images/text.jpg', cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("The file 'images/text.jpg' was not found. Please check the path.")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
cv2.waitKey(500)
cv2.destroyAllWindows()

# Preprocessing
blur = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blur, 30, 150)
contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found:", len(contours))

# Sort contours from left to right
contours = sorted(contours, key=contour_coordinates)

# Process each contour
detected_digits = []
raw_images = []  # Store original digit images for inspection
confidence_scores = []

for i, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    if w >= 5 and h >= 20:  # Filter by size
        roi = blur[y:y+h, x:x+w]
        _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Preprocessing the digit
        square = drawSquare(roi)  # Draw a square around the found digits
        digit_resized = resize(square, 20)  # Resize to 20x20 for the KNN model
        
        # Save the raw digit image for debugging
        raw_images.append(digit_resized)
        
        # Reshape the digit into a 1D array of 400 elements for the KNN model
        result = digit_resized.reshape((1, 400)).astype(np.float32)
        
        # Classify the digit using KNN with more detailed output
        ret, res, neighbours, distance = classifier_knn.findNearest(result, k=3)
        
        # Convert to digit and calculate confidence
        detected_digit = str(int(float(res[0])))
        
        # Calculate confidence based on nearest neighbors
        unique, counts = np.unique(neighbours[0], return_counts=True)
        confidence = counts[0] / len(neighbours[0]) * 100
        
        detected_digits.append(detected_digit)
        confidence_scores.append(confidence)
        
        # Annotate image with digit and confidence
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        label = f"{detected_digit} ({confidence:.1f}%)"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Additional debugging print
        print(f"Digit {i}: {detected_digit}, Confidence: {confidence:.1f}%, Neighbours: {neighbours[0]}")

# Save and display final result
cv2.imshow("Final Image", image)
cv2.imwrite("output_image_with_confidence.jpg", image)
cv2.waitKey(500)
cv2.destroyAllWindows()

print("\nDetected digits:", detected_digits)
print("Confidence scores:", confidence_scores)

# Debug: Save individual digit images with more context
for i, digit in enumerate(raw_images):
    cv2.imwrite(f'debug_digit_{i}_value_{detected_digits[i]}.jpg', digit)
    print(f"Saved debug image for digit_{i} (value: {detected_digits[i]})")
