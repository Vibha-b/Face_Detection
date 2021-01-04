import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Choose an image to detect faces in - image read function
#img = cv2.imread('rdj.jpg')
# To capture video from webcam.
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Read current time frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    # cvtColor() : convert an image from one color space to another
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # detectMultiScale() : Detects objects of different sizes in the input image.
    # The detected objects are returned as a list of rectangles.
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    # rectangle(image, start_point, end_point, color, thickness)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    # ASCII character for q:113 and Q:81
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
webcam.release()

print("CODE COMPLETED")
