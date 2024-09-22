import cv2
import pytesseract
import re  # Import regex library to validate number plate

# Path to your Haar Cascade model
harcascade = "model/haarcascade_russian_plate_number.xml"

# Initialize video capture (0 for webcam)
cap = cv2.VideoCapture(0)

# Set width and height
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Minimum area of the detected plate region
min_area = 500
count = 0

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Regular expression for validating number plates (customize based on your country's format)
plate_regex = re.compile(r'^TN\d{2}[A-Z]{1,2}\d{1,4}$')


def preprocess_image(img_roi):
    """Preprocess the image to improve OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to make the text stand out
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return thresh

detected = False  # Flag to track if a valid plate has been detected

while True:
    success, img = cap.read()  # Capture frame from the webcam

    if detected:  # If plate has been detected, exit the loop
        break

    # Convert frame to grayscale for Haar Cascade
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect plates
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            # Draw rectangle around the detected plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extract the region of interest (ROI) - the plate itself
            img_roi = img[y: y + h, x: x + w]
            img_roi = cv2.resize(img_roi, (300, 100))  # Adjust the size as necessary

            # Preprocess the image for better OCR results
            preprocessed_img = preprocess_image(img_roi)

            # Extract text from the number plate using pytesseract
            plate_text = pytesseract.image_to_string(preprocessed_img, config='--psm 6')

            plate_text = plate_text.strip()  # Remove extra spaces
            plate_text = re.sub(r'\W+', '', plate_text)  # Remove all non-alphanumeric characters

            # Validate if the text matches a number plate pattern
            if plate_regex.match(plate_text):
                # Only save the plate image if it is valid
                plate_filename = "plates/scanned_img_" + str(count) + ".jpg"
                cv2.imwrite(plate_filename, preprocessed_img)
                print(f"Saved: {plate_filename}")
                print(f"Detected Number Plate Text: {plate_text}")

                # Optional: Display the detected plate (ROI) in a separate window
                cv2.imshow("ROI", preprocessed_img)

                # Show a message that the plate is saved
                cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                cv2.imshow("Results", img)
                cv2.waitKey(500)
                count += 1

                detected = True  # Set flag to true to stop detection
                break  # Break out of the plate detection loop

    # Display the live video feed with detected plates
    cv2.imshow("Result", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
