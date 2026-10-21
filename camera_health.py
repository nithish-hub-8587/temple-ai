import cv2
import time
import json
import os
import numpy as np


# ==========================================
# CONFIGURATION
# ==========================================

CAMERA_ID = 0

HEALTH_FILE = "health_status.json"

RECONNECT_DELAY = 2
MAX_RECONNECT_ATTEMPTS = 5
FREEZE_TIMEOUT = 3

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ==========================================
# FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Face detector could not be loaded!")
    exit()

print("Face detector loaded successfully.")


# ==========================================
# VARIABLES
# ==========================================

camera = None

fps = 0
frame_count = 0
fps_start_time = time.time()

previous_gray = None
frozen_start_time = None

camera_status = "OFFLINE"
camera_faces = 0

total_reconnects = 0

last_frame_time = "Never"


# ==========================================
# WRITE HEALTH STATUS
# ==========================================

def write_health_status():

    data = {
        "status": camera_status,
        "camera_id": CAMERA_ID,
        "fps": round(fps, 1),
        "faces": camera_faces,
        "reconnects": total_reconnects,
        "last_frame": last_frame_time,
        "last_update": time.time()
    }

    try:

        temp_file = HEALTH_FILE + ".tmp"

        with open(temp_file, "w") as file:
            json.dump(data, file, indent=4)

        os.replace(temp_file, HEALTH_FILE)

    except Exception as error:

        print("Health file error:", error)


# ==========================================
# OPEN CAMERA
# ==========================================

def open_camera():

    global camera

    print("Opening camera...")

    if camera is not None:
        camera.release()

    camera = cv2.VideoCapture(CAMERA_ID)

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    if camera.isOpened():

        print("🟢 CAMERA ONLINE")

        return True

    print("🔴 CAMERA OFFLINE")

    return False


# ==========================================
# START CAMERA
# ==========================================

open_camera()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # ------------------------------------------
    # CAMERA NOT OPEN
    # ------------------------------------------

    if camera is None or not camera.isOpened():

        camera_status = "OFFLINE"

        camera_faces = 0
        fps = 0

        write_health_status()

        print("Trying to reconnect...")

        time.sleep(RECONNECT_DELAY)

        if open_camera():

            total_reconnects += 1

        continue


    # ------------------------------------------
    # READ FRAME
    # ------------------------------------------

    ret, frame = camera.read()


    # ------------------------------------------
    # FRAME FAILURE
    # ------------------------------------------

    if not ret:

        print("Could not read frame.")

        camera_status = "OFFLINE"

        camera_faces = 0
        fps = 0

        write_health_status()

        camera.release()

        camera = None

        continue


    # ------------------------------------------
    # CAMERA ONLINE
    # ------------------------------------------

    camera_status = "ONLINE"

    frame_count += 1

    last_frame_time = time.strftime("%H:%M:%S")


    # ------------------------------------------
    # FPS
    # ------------------------------------------

    current_time = time.time()

    elapsed = current_time - fps_start_time

    if elapsed >= 1:

        fps = frame_count / elapsed

        frame_count = 0

        fps_start_time = current_time


    # ------------------------------------------
    # GRAYSCALE
    # ------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ------------------------------------------
    # FROZEN FRAME DETECTION
    # ------------------------------------------

    if previous_gray is not None:

        difference = cv2.absdiff(
            previous_gray,
            gray
        )

        mean_difference = np.mean(
            difference
        )

        if mean_difference < 0.5:

            if frozen_start_time is None:

                frozen_start_time = time.time()

            freeze_duration = (
                time.time() -
                frozen_start_time
            )

            if freeze_duration >= FREEZE_TIMEOUT:

                camera_status = "FROZEN"

        else:

            frozen_start_time = None

            if camera_status == "FROZEN":

                camera_status = "ONLINE"


    previous_gray = gray.copy()


    # ------------------------------------------
    # FACE DETECTION
    # ------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    camera_faces = len(faces)


    # ------------------------------------------
    # DRAW FACES
    # ------------------------------------------

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ------------------------------------------
    # CAMERA STATUS
    # ------------------------------------------

    cv2.putText(
        frame,
        f"Camera: {camera_status}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ------------------------------------------
    # FPS
    # ------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ------------------------------------------
    # FACE COUNT
    # ------------------------------------------

    cv2.putText(
        frame,
        f"Faces: {camera_faces}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ------------------------------------------
    # RECONNECTION COUNT
    # ------------------------------------------

    cv2.putText(
        frame,
        f"Reconnects: {total_reconnects}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ------------------------------------------
    # SAVE HEALTH DATA
    # ------------------------------------------

    write_health_status()


    # ------------------------------------------
    # SHOW CAMERA
    # ------------------------------------------

    cv2.imshow(
        "Temple AI - Camera Health",
        frame
    )


    # ------------------------------------------
    # QUIT
    # ------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera_status = "OFFLINE"
camera_faces = 0
fps = 0

write_health_status()


if camera is not None:
    camera.release()

cv2.destroyAllWindows()

print("🛑 Temple AI camera monitoring stopped.")