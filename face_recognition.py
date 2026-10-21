import cv2
from deepface import DeepFace
import os
import sqlite3
from datetime import datetime
import time
import json
import threading
from playsound import playsound


# =========================================================
# CONFIGURATION
# =========================================================

DATABASE = "missing_persons.db"
FACE_FOLDER = "known_faces"

CAMERA_ID = 0

HEALTH_FILE = "health_status.json"
FRAME_FILE = "latest_frame.jpg"

ALERT_COOLDOWN = 30

# Number of seconds before a camera is considered frozen
FROZEN_TIME = 3


# =========================================================
# GLOBAL VARIABLES
# =========================================================

last_alert_person = ""
last_alert_time = 0

reconnect_count = 0


# =========================================================
# UPDATE CAMERA HEALTH
# =========================================================

def update_health(status, fps, faces):

    data = {
        "status": status,
        "camera_id": CAMERA_ID,
        "fps": round(fps, 2),
        "faces": faces,
        "reconnects": reconnect_count,
        "last_frame": datetime.now().strftime("%H:%M:%S"),
        "last_update": time.time()
    }

    try:

        # Temporary file prevents Flask from reading
        # incomplete JSON data.

        temp_file = HEALTH_FILE + ".tmp"

        with open(temp_file, "w") as file:

            json.dump(data, file)

        os.replace(
            temp_file,
            HEALTH_FILE
        )

    except Exception as e:

        print(
            "Health update error:",
            e
        )


# =========================================================
# SAVE ALERT TO DATABASE
# =========================================================

def save_alert(person_name):

    try:

        connection = sqlite3.connect(
            DATABASE
        )

        cursor = connection.cursor()

        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        cursor.execute(
            """
            INSERT INTO alerts
            (person_name, camera_name, match_time, status)
            VALUES (?, ?, ?, ?)
            """,
            (
                person_name,
                "Camera 1",
                current_time,
                "Pending Verification"
            )
        )

        connection.commit()

        connection.close()

        print(
            "🚨 ALERT SAVED TO DATABASE"
        )

        print(
            "Person:",
            person_name
        )

        print(
            "Time:",
            current_time
        )

    except Exception as e:

        print(
            "Database alert error:",
            e
        )


# =========================================================
# PLAY SOUND ALERT
# =========================================================

def play_alert_sound():

    try:

        if os.path.exists("alert.mp3"):

            # Play sound in background so that
            # camera processing does not stop.

            threading.Thread(
                target=playsound,
                args=("alert.mp3",),
                daemon=True
            ).start()

            print(
                "🔊 SOUND ALERT PLAYED"
            )

        else:

            print(
                "⚠ alert.mp3 not found"
            )

    except Exception as e:

        print(
            "Sound error:",
            e
        )


# =========================================================
# FACE DETECTOR
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


print(
    "Face detector loaded:",
    not face_cascade.empty()
)


# =========================================================
# CHECK FACE DATABASE
# =========================================================

if not os.path.exists(FACE_FOLDER):

    print(
        "❌ known_faces folder not found"
    )

    exit()


print(
    "Face database:",
    FACE_FOLDER
)


# =========================================================
# OPEN CAMERA
# =========================================================

camera = cv2.VideoCapture(
    CAMERA_ID
)


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# =========================================================
# CHECK CAMERA
# =========================================================

if not camera.isOpened():

    print(
        "❌ Camera could not be opened"
    )

    update_health(
        "OFFLINE",
        0,
        0
    )

    exit()


# =========================================================
# START MESSAGE
# =========================================================

print(
    "========================================"
)

print(
    "Temple AI Face Recognition Started"
)

print(
    "========================================"
)

print(
    "Camera ID:",
    CAMERA_ID
)

print(
    "Live frame:",
    FRAME_FILE
)

print(
    "Health file:",
    HEALTH_FILE
)

print(
    "Press Q to quit"
)

print(
    "========================================"
)


# =========================================================
# FPS VARIABLES
# =========================================================

previous_time = time.time()

fps = 0


# =========================================================
# FROZEN FRAME VARIABLES
# =========================================================

previous_gray = None

frozen_start = None


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    # =====================================================
    # READ CAMERA FRAME
    # =====================================================

    ret, frame = camera.read()


    # =====================================================
    # CAMERA READ FAILURE
    # =====================================================

    if not ret:

        print(
            "❌ Could not read camera frame"
        )

        update_health(
            "RECONNECTING",
            0,
            0
        )

        camera.release()

        time.sleep(2)

        reconnect_count += 1

        print(
            "🔄 Attempting camera reconnect..."
        )

        camera = cv2.VideoCapture(
            CAMERA_ID
        )

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        continue


    # =====================================================
    # FPS CALCULATION
    # =====================================================

    current_time = time.time()

    time_difference = (
        current_time - previous_time
    )

    if time_difference > 0:

        fps = 1 / time_difference

    previous_time = current_time


    # =====================================================
    # CONVERT TO GRAYSCALE
    # =====================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # =====================================================
    # FACE DETECTION
    # =====================================================

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(50, 50)

    )


    # =====================================================
    # FROZEN FRAME DETECTION
    # =====================================================

    camera_frozen = False


    if previous_gray is not None:

        difference = cv2.absdiff(
            previous_gray,
            gray
        )

        movement = difference.mean()


        if movement < 0.5:

            if frozen_start is None:

                frozen_start = time.time()

            elif (
                time.time()
                - frozen_start
                > FROZEN_TIME
            ):

                camera_frozen = True

        else:

            frozen_start = None


    previous_gray = gray.copy()


    # =====================================================
    # PROCESS EACH FACE
    # =====================================================

    for (x, y, w, h) in faces:

        face = frame[
            y:y+h,
            x:x+w
        ]


        try:

            # =================================================
            # DEEPFACE RECOGNITION
            # =================================================

            results = DeepFace.find(

                img_path=face,

                db_path=FACE_FOLDER,

                enforce_detection=False,

                silent=True

            )


            # =================================================
            # MATCH FOUND
            # =================================================

            if (
                len(results) > 0
                and len(results[0]) > 0
            ):

                result = results[0]


                # ---------------------------------------------
                # GET MATCHED IMAGE
                # ---------------------------------------------

                matched_file = result.iloc[0][
                    "identity"
                ]


                # ---------------------------------------------
                # GET PERSON NAME
                # ---------------------------------------------

                relative_path = os.path.relpath(

                    matched_file,

                    FACE_FOLDER

                )


                parts = os.path.normpath(
                    relative_path
                ).split(os.sep)


                if len(parts) > 1:

                    name = parts[0]

                else:

                    name = os.path.splitext(
                        parts[0]
                    )[0]


                # =================================================
                # DRAW GREEN FACE BOX
                # =================================================

                cv2.rectangle(

                    frame,

                    (x, y),

                    (x + w, y + h),

                    (0, 255, 0),

                    2

                )


                # =================================================
                # DISPLAY MATCH FOUND
                # =================================================

                cv2.putText(

                    frame,

                    "MATCH FOUND",

                    (x, y - 40),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 255, 0),

                    2

                )


                # =================================================
                # DISPLAY NAME
                # =================================================

                cv2.putText(

                    frame,

                    "Name: " + name,

                    (x, y - 15),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 0),

                    2

                )


                # =================================================
                # VERIFICATION MESSAGE
                # =================================================

                cv2.putText(

                    frame,

                    "Verification Required",

                    (x, y + h + 25),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (0, 255, 255),

                    2

                )


                print(
                    "Potential Match:",
                    name
                )


                # =================================================
                # ALERT CONTROL
                # =================================================

                alert_time = time.time()


                if (

                    name != last_alert_person

                    or

                    alert_time
                    - last_alert_time
                    > ALERT_COOLDOWN

                ):

                    # ---------------------------------------------
                    # SAVE ALERT
                    # ---------------------------------------------

                    save_alert(
                        name
                    )


                    # ---------------------------------------------
                    # SOUND ALERT
                    # ---------------------------------------------

                    play_alert_sound()


                    # ---------------------------------------------
                    # UPDATE ALERT TRACKING
                    # ---------------------------------------------

                    last_alert_person = name

                    last_alert_time = alert_time


            # =================================================
            # NO MATCH
            # =================================================

            else:

                cv2.rectangle(

                    frame,

                    (x, y),

                    (x + w, y + h),

                    (255, 0, 0),

                    2

                )


                cv2.putText(

                    frame,

                    "No Match",

                    (x, y - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (255, 0, 0),

                    2

                )


        except Exception as e:

            print(
                "Recognition error:",
                e
            )


    # =====================================================
    # CAMERA FROZEN WARNING
    # =====================================================

    if camera_frozen:

        cv2.putText(

            frame,

            "CAMERA FROZEN",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 0, 255),

            2

        )

        update_health(

            "FROZEN",

            fps,

            len(faces)

        )

    else:

        update_health(

            "ONLINE",

            fps,

            len(faces)

        )


    # =====================================================
    # DISPLAY FPS
    # =====================================================

    cv2.putText(

        frame,

        f"FPS: {fps:.1f}",

        (10, 470),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    # =====================================================
    # DISPLAY FACE COUNT
    # =====================================================

    cv2.putText(

        frame,

        f"Faces: {len(faces)}",

        (10, 445),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    # =====================================================
    # SAVE LIVE FRAME FOR FLASK
    # =====================================================

    try:

        success, encoded_frame = cv2.imencode(

            ".jpg",

            frame,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                80
            ]

        )


        if success:

            temp_frame = (
                FRAME_FILE
                + ".tmp"
            )


            with open(
                temp_frame,
                "wb"
            ) as file:

                file.write(
                    encoded_frame.tobytes()
                )


            # Atomic replacement
            # Flask will always get a complete JPEG.

            os.replace(

                temp_frame,

                FRAME_FILE

            )


    except Exception as e:

        print(
            "Live frame save error:",
            e
        )


    # =====================================================
    # SHOW LOCAL CAMERA WINDOW
    # =====================================================

    cv2.imshow(

        "Temple AI - Missing Person Detection",

        frame

    )


    # =====================================================
    # QUIT WITH Q
    # =====================================================

    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):

        break


# =========================================================
# CLEANUP
# =========================================================

camera.release()

cv2.destroyAllWindows()


# =========================================================
# SET CAMERA OFFLINE
# =========================================================

update_health(

    "OFFLINE",

    0,

    0

)


print(
    "Temple AI stopped."
)