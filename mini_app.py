def take_img():
    enrollment = txt.get()
    name = txt2.get()
    if enrollment == '' or name == '':
        Notification.configure(text="Enrollment & Name required!", bg="red", fg="white")
    else:
        # Check if TrainingImage folder exists, if not create it
        if not os.path.exists("TrainingImage"):
            try:
                os.makedirs("TrainingImage")
                print("TrainingImage folder created!")
            except Exception as e:
                Notification.configure(text=f"Error creating folder: {str(e)}", bg="red", fg="white")
                return

        # Initialize the webcam
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            Notification.configure(text="Camera Error! Cannot access camera.", bg="red", fg="white")
            print("Error: Camera could not be opened.")
            return

        # Initialize face detector
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        sampleNum = 0

        while True:
            ret, img = cam.read()
            if not ret:
                Notification.configure(text="Error capturing image!", bg="red", fg="white")
                print("Error: Failed to capture image.")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            # Print number of faces detected
            print(f"Faces detected: {len(faces)}")

            # If no faces are detected, give the user a prompt to adjust their position
            if len(faces) == 0:
                Notification.configure(text="No faces detected. Adjust your position.", bg="red", fg="white")
                print("No faces detected.")
            else:
                for (x, y, w, h) in faces:
                    sampleNum += 1
                    print(f"Face {sampleNum} detected: x={x}, y={y}, w={w}, h={h}")

                    # Create filename for the image
                    filename = f"TrainingImage/{name}_{enrollment}_{sampleNum}.jpg"
                    print(f"Saving Image: {filename}")

                    # Save only the first face detected and stop the loop
                    cv2.imwrite(filename, gray[y:y+h, x:x+w])
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # After saving the first image, stop the loop
                    sampleNum = 1
                    break

            cv2.imshow('Capturing Images', img)

            # Stop after saving one image
            if sampleNum == 1:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Update the notification text after the camera operation is done
        if sampleNum == 1:
            Notification.configure(text=f"Image Saved for Enrollment: {enrollment}, Name: {name}", bg="green", fg="white")
        else:
            Notification.configure(text="No faces captured! Try again.", bg="red", fg="white")
