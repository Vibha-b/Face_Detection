import cv2

# Face and smile classifiers
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')

# Grab Webcam feed
webcam = cv2.VideoCapture(0)

while True:
    # Read current frame from webcam
    successful_frame_read, frame = webcam.read()

    # If there's an error, abort
    if not successful_frame_read:
        break

    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)
    # smiles = smile_detector.detectMultiScale(frame_grayscale) # scaleFactor=1.7, minNeighbors=20)

    # Run face detection within each of these faces
    for (x, y, w, h) in faces:

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get the sub frame (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]

        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # Find all the smiles in the face
        # for (x_, y_, w_, h_) in smiles:

        # Draw a rectangle around the smile
        #cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0, 255, 0), 4)

        # Label this face as smiling
        if(len(smiles) > 0):
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))

        cv2.imshow('Face Detector', frame)
        key = cv2.waitKey(1)

    # Stop if Q key is pressed
    # ASCII character for q:113 and Q:81
    if key == 81 or key == 113:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
print("CODE COMPLETED")
