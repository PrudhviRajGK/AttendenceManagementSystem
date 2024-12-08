import tkinter as tk
import cv2
import os
import numpy as np
from PIL import Image
import csv
from tkinter import messagebox
from datetime import datetime

# Initialize the recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Main window setup
window = tk.Tk()
window.title("Automatic Attendance System")
window.geometry('600x400')
window.configure(background='lightgrey')

# Notification label
Notification = tk.Label(window, text="", bg="lightgrey", fg="black", width=40, height=2, font=('times', 12))
Notification.place(x=50, y=250)

# Function to capture images and save them for training
def take_img():
    enrollment = txt.get()
    name = txt2.get()
    if enrollment == '' or name == '':
        Notification.configure(text="Enrollment & Name required!", bg="red", fg="white")
    else:
        if not os.path.exists("TrainingImage"):
            os.makedirs("TrainingImage")
            print("TrainingImage folder created!")

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            Notification.configure(text="Camera Error! Cannot access camera.", bg="red", fg="white")
            return

        sampleNum = 0
        while True:
            ret, img = cam.read()
            if not ret:
                Notification.configure(text="Error capturing image!", bg="red", fg="white")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                Notification.configure(text="No faces detected. Adjust your position.", bg="red", fg="white")
                print("No faces detected.")
            else:
                for (x, y, w, h) in faces:
                    sampleNum += 1
                    filename = f"TrainingImage/{name}_{enrollment}_{sampleNum}.jpg"
                    cv2.imwrite(filename, gray[y:y+h, x:x+w])
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    break

            cv2.imshow('Capturing Images', img)

            if sampleNum == 1:
                break

        cam.release()
        cv2.destroyAllWindows()

        if sampleNum == 1:
            Notification.configure(text=f"Image Saved for Enrollment: {enrollment}, Name: {name}", bg="green", fg="white")
        else:
            Notification.configure(text="No faces captured! Try again.", bg="red", fg="white")

# Function to train the recognizer
def train_recognizer():
    faces, Ids = getImagesAndLabels('TrainingImage')
    if faces and Ids:
        recognizer.train(faces, np.array(Ids))
        recognizer.save('TrainingImageLabel/trainner.yml')
        Notification.configure(text="Training successful!", bg="green", fg="white")
    else:
        Notification.configure(text="No training data found! Please add images.", bg="red", fg="white")

# Function to get images and labels for training
def getImagesAndLabels(path):
    if not os.path.exists(path):
        print(f"Error: Folder '{path}' does not exist!")
        return [], []

    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    if not imagePaths:
        print(f"No images found in the folder '{path}'!")
        return [], []

    faceSamples = []
    Ids = []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split("_")[1])  # Adjusted for file format: name_enrollment_1.jpg
        faces = detector.detectMultiScale(imageNp)

        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)

    return faceSamples, Ids

# Function to mark attendance automatically
def take_attendance():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        Notification.configure(text="Camera Error! Cannot access camera.", bg="red", fg="white")
        return

    while True:
        ret, img = cam.read()
        if not ret:
            Notification.configure(text="Error capturing image!", bg="red", fg="white")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

                if conf < 50:
                    mark_attendance(Id)
                    Notification.configure(text=f"Attendance marked for ID: {Id}", bg="green", fg="white")
                else:
                    Notification.configure(text="Unknown face detected.", bg="red", fg="white")
                break
        else:
            Notification.configure(text="No faces detected.", bg="red", fg="white")

        cv2.imshow('Attendance System', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Function to save attendance
def mark_attendance(id):
    name = txt2.get()  # Get the name of the student (this can be modified to get the name from a database)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save attendance to a CSV file
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([id, name, current_time])

# Function to view attendance records
def view_attendance():
    attendance_file = 'attendance.csv'

    # Check if the file exists
    try:
        with open(attendance_file, 'r') as file:
            reader = csv.reader(file)
            records = list(reader)

        if len(records) == 0:
            Notification.configure(text="No attendance records found.", bg="red", fg="white")
        else:
            # Display records in a new window
            records_window = tk.Toplevel(window)
            records_window.title("Attendance Records")

            text_area = tk.Text(records_window, width=50, height=15)
            text_area.pack()

            # Adding header
            text_area.insert(tk.END, "ID\tName\tTime\n")
            
            for record in records:
                text_area.insert(tk.END, "\t".join(record) + "\n")

            text_area.config(state=tk.DISABLED)  # Make it non-editable
            
            Notification.configure(text="Attendance records displayed.", bg="green", fg="white")

    except FileNotFoundError:
        messagebox.showerror("Error", "Attendance file not found!")

# GUI Setup
lbl = tk.Label(window, text="Enter Enrollment: ", width=20, height=2, fg="black", bg="lightgrey", font=('times', 12, 'bold'))
lbl.place(x=50, y=100)

txt = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 12))
txt.place(x=250, y=115)

lbl2 = tk.Label(window, text="Enter Name: ", width=20, fg="black", bg="lightgrey", height=2, font=('times', 12, 'bold'))
lbl2.place(x=50, y=150)

txt2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 12))
txt2.place(x=250, y=165)

# Create buttons
takeImg = tk.Button(window, text="Capture Images", command=take_img, fg="black", bg="lightblue", width=15, height=2, font=('times', 12, 'bold'))
takeImg.place(x=50, y=320)

trainButton = tk.Button(window, text="Train Model", command=train_recognizer, fg="black", bg="lightblue", width=15, height=2, font=('times', 12, 'bold'))
trainButton.place(x=250, y=320)

takeAttendanceButton = tk.Button(window, text="Automatic Attendance", command=take_attendance, fg="black", bg="lightblue", width=15, height=2, font=('times', 12, 'bold'))
takeAttendanceButton.place(x=450, y=320)

checkSheetsButton = tk.Button(window, text="Check Sheets", command=view_attendance, fg="black", bg="lightblue", width=15, height=2, font=('times', 12, 'bold'))
checkSheetsButton.place(x=50, y=370)

window.mainloop()
