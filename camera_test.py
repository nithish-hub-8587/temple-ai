import cv2
import time
import numpy as np


# ==========================================
# CONFIGURATION
# ==========================================

CAMERA_ID = 0

RECONNECT_DELAY = 2
MAX_RECONNECT_ATTEMPTS = 5

FREEZE_TIMEOUT = 3


# ==========================================
# FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

print("Face detector loaded:", not face_cascade.empty())


# ==========================================
# CAMERA FUNCTION
# ==========================================

def open_camera():

    print("Opening camera...")

    camera = cv2.VideoCapture(CAMERA_ID)

    if camera.isOpened():

        print("🟢 CAMERA ONLINE")

        # Optional camera settings
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return camera

    print("🔴 CAMERA OPEN FAILED")

    camera.release()

    return None


# ==========================================
# INITIAL CAMERA
# ==========================================

camera = open_camera()

if camera is None:

    print("Could not start camera.")

    exit()


# ==========================================
# HEALTH VARIABLES
# ==========================================

fps = 0

frame_count = 0

fps_start_time = time.time()

previous_gray = None

frozen_start_time = None

camera_status = "CAMERA ONLINE"

reconnect_attempts = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # --------------------------------------
    # READ FRAME
    # --------------------------------------

    ret, frame = camera.read()

    current_time = time.time()


    # ======================================
    # CAMERA FAILURE
    # ======================================

    if not ret or frame is None:

        camera_status = "CAMERA OFFLINE"

        print("🔴 Camera frame lost!")

        camera.release()

        camera = None


        # ==================================
        # AUTOMATIC RECONNECTION
        # ==================================

        while camera is None:

            reconnect_attempts += 1

            print(
                f"🔄 Reconnecting camera... "
                f"Attempt {reconnect_attempts}"
            )

            time.sleep(RECONNECT_DELAY)

            camera = open_camera()


            # ------------------------------
            # MAX ATTEMPTS
            # ------------------------------

            if camera is None:

                if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:

                    print(
                        "⚠️ Maximum reconnect attempts reached."
                    )

                    reconnect_attempts = 0

                    break


        # ----------------------------------
        # CAMERA SUCCESSFULLY RECONNECTED
        # ----------------------------------

        if camera is not None:

            camera_status = "CAMERA ONLINE"

            reconnect_attempts = 0

            previous_gray = None

            frozen_start_time = None

            print("✅ Camera successfully recovered!")

            continue


        # ----------------------------------
        # SHOW OFFLINE SCREEN
        # ----------------------------------

        offline_frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

        cv2.putText(
            offline_frame,
            "CAMERA OFFLINE",
            (150, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

        cv2.putText(
            offline_frame,
            "Attempting recovery...",
            (150, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Temple AI - Camera Health Monitoring",
            offline_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue


    # ======================================
    # CAMERA RECOVERED
    # ======================================

    reconnect_attempts = 0

    camera_status = "CAMERA ONLINE"


    # ======================================
    # FPS CALCULATION
    # ======================================

    frame_count += 1

    elapsed = current_time - fps_start_time

    if elapsed >= 1:

        fps = frame_count / elapsed

        frame_count = 0

        fps_start_time = current_time


    # ======================================
    # GRAYSCALE
    # ======================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ======================================
    # FROZEN FRAME DETECTION
    # ======================================

    if previous_gray is not None:

        difference = cv2.absdiff(
            previous_gray,
            gray
        )

        average_difference = difference.mean()


        if average_difference < 0.5:

            if frozen_start_time is None:

                frozen_start_time = current_time

            elif (
                current_time - frozen_start_time
                >= FREEZE_TIMEOUT
            ):

                camera_status = "CAMERA FROZEN"


        else:

            frozen_start_time = None

            camera_status = "CAMERA ONLINE"


    previous_gray = gray.copy()


    # ======================================
    # FACE DETECTION
    # ======================================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )


    # ======================================
    # DRAW FACES
    # ======================================

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


    # ======================================
    # HEALTH INFORMATION
    # ======================================

    status_color = (
        (0, 255, 0)
        if camera_status == "CAMERA ONLINE"
        else (0, 0, 255)
    )


    cv2.putText(
        frame,
        f"Camera: {camera_status}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "Temple AI - Camera Health Monitoring",
        frame
    )


    # ======================================
    # EXIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

if camera is not None:

    camera.release()

cv2.destroyAllWindows()

print("🛑 Temple AI camera monitoring stopped.")