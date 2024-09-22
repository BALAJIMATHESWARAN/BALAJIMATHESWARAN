import cv2
import pytesseract

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
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Modify this path if necessary

while True:
    success, img = cap.read()  # Capture frame from the webcam

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

            # Save the plate image automatically
            plate_filename = "plates/scanned_img_" + str(count) + ".jpg"
            cv2.imwrite(plate_filename, img_roi)
            print(f"Saved: {plate_filename}")

            # Extract text from the number plate using pytesseract
            plate_text = pytesseract.image_to_string(img_roi, config='--psm 8')  # '--psm 8' is for single word detection
            print(f"Detected Number Plate Text: {plate_text.strip()}")  # Output the extracted text

            # Optional: Display the detected plate (ROI) in a separate window
            cv2.imshow("ROI", img_roi)

            # Show a message that the plate is saved
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Results", img)
            cv2.waitKey(500)
            count += 1

    # Display the live video feed with detected plates
    cv2.imshow("Result", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
