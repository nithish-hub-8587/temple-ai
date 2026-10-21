import os
import json
import time
import sqlite3
import subprocess
import sys
import atexit

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    jsonify,
    send_from_directory,
    Response
)

from werkzeug.utils import secure_filename


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE = os.path.join(
    BASE_DIR,
    "missing_persons.db"
)

FACE_FOLDER = os.path.join(
    BASE_DIR,
    "known_faces"
)

HEALTH_FILE = os.path.join(
    BASE_DIR,
    "health_status.json"
)

FRAME_FILE = os.path.join(
    BASE_DIR,
    "latest_frame.jpg"
)

CAMERA_SCRIPT = os.path.join(
    BASE_DIR,
    "face_recognition.py"
)


# =========================================================
# CREATE FACE FOLDER
# =========================================================

os.makedirs(
    FACE_FOLDER,
    exist_ok=True
)


# =========================================================
# CAMERA PROCESS
# =========================================================

camera_process = None


# =========================================================
# START CAMERA
# =========================================================

def start_camera():

    global camera_process

    # -----------------------------------------
    # Check whether already running
    # -----------------------------------------

    if camera_process is not None:

        if camera_process.poll() is None:

            return False, "Camera monitoring is already running"


    # -----------------------------------------
    # Check face recognition file
    # -----------------------------------------

    if not os.path.exists(CAMERA_SCRIPT):

        return False, (
            "face_recognition.py not found"
        )


    try:

        print(
            "========================================"
        )

        print(
            "Starting Temple AI Camera..."
        )

        print(
            "Python:",
            sys.executable
        )

        print(
            "Script:",
            CAMERA_SCRIPT
        )

        print(
            "========================================"
        )


        # -----------------------------------------
        # Start AI process
        # -----------------------------------------

        camera_process = subprocess.Popen(

            [
                sys.executable,
                CAMERA_SCRIPT
            ],

            cwd=BASE_DIR,

            stdout=None,

            stderr=None

        )


        # -----------------------------------------
        # Give process time to start
        # -----------------------------------------

        time.sleep(2)


        # -----------------------------------------
        # Check whether process crashed
        # -----------------------------------------

        if camera_process.poll() is not None:

            exit_code = camera_process.returncode

            camera_process = None

            return False, (
                "Camera program stopped immediately. "
                f"Exit code: {exit_code}"
            )


        print(
            "🟢 Camera process started successfully"
        )


        return True, (
            "Camera monitoring started"
        )


    except Exception as e:

        print(
            "❌ Camera start error:",
            e
        )

        camera_process = None

        return False, str(e)


# =========================================================
# STOP CAMERA
# =========================================================

def stop_camera():

    global camera_process


    # -----------------------------------------
    # No process
    # -----------------------------------------

    if camera_process is None:

        return False, (
            "Camera monitoring is not running"
        )


    try:

        # -----------------------------------------
        # Stop process
        # -----------------------------------------

        if camera_process.poll() is None:

            print(
                "🔴 Stopping camera..."
            )

            camera_process.terminate()


            try:

                camera_process.wait(
                    timeout=5
                )

            except subprocess.TimeoutExpired:

                print(
                    "⚠ Force stopping camera..."
                )

                camera_process.kill()

                camera_process.wait()


        camera_process = None


        # -----------------------------------------
        # Update health status
        # -----------------------------------------

        health_data = {

            "status": "OFFLINE",

            "camera_id": 0,

            "fps": 0,

            "faces": 0,

            "reconnects": 0,

            "last_frame":
                "Stopped from dashboard",

            "last_update": 0

        }


        try:

            temp_file = (
                HEALTH_FILE +
                ".tmp"
            )


            with open(
                temp_file,
                "w"
            ) as file:

                json.dump(
                    health_data,
                    file
                )


            os.replace(
                temp_file,
                HEALTH_FILE
            )


        except Exception as e:

            print(
                "Health reset error:",
                e
            )


        print(
            "🔴 Camera monitoring stopped"
        )


        return True, (
            "Camera monitoring stopped"
        )


    except Exception as e:

        print(
            "❌ Camera stop error:",
            e
        )

        camera_process = None

        return False, str(e)


# =========================================================
# CAMERA CONTROL API
# =========================================================

@app.route(
    "/api/camera-control",
    methods=["POST"]
)
def camera_control():

    try:

        data = request.get_json()

        if not data:

            return jsonify({

                "success": False,

                "message":
                    "No action received"

            }), 400


        action = data.get(
            "action"
        )


        # =====================================
        # START
        # =====================================

        if action == "start":

            success, message = (
                start_camera()
            )


            return jsonify({

                "success": success,

                "message": message

            })


        # =====================================
        # STOP
        # =====================================

        elif action == "stop":

            success, message = (
                stop_camera()
            )


            return jsonify({

                "success": success,

                "message": message

            })


        # =====================================
        # INVALID
        # =====================================

        return jsonify({

            "success": False,

            "message":
                "Invalid camera action"

        }), 400


    except Exception as e:

        print(
            "Camera control error:",
            e
        )

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500


# =========================================================
# CAMERA PROCESS STATUS
# =========================================================

@app.route(
    "/api/camera-process"
)
def camera_process_status():

    global camera_process


    running = False


    if camera_process is not None:

        if camera_process.poll() is None:

            running = True

        else:

            camera_process = None


    return jsonify({

        "running": running

    })


# =========================================================
# CLEANUP CAMERA WHEN FLASK CLOSES
# =========================================================

@atexit.register
def cleanup_camera():

    global camera_process


    if camera_process is not None:

        try:

            if camera_process.poll() is None:

                print(
                    "Cleaning up camera process..."
                )

                camera_process.terminate()

                camera_process.wait(
                    timeout=3
                )

        except Exception:

            pass

        camera_process = None


# =========================================================
# CREATE DATABASE
# =========================================================

def create_database():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    # -----------------------------------------
    # Missing persons
    # -----------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS missing_persons (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            age INTEGER,

            description TEXT,

            photo_path TEXT NOT NULL

        )
    """)


    # -----------------------------------------
    # Alerts
    # -----------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            person_name TEXT NOT NULL,

            camera_name TEXT,

            match_time TEXT,

            status TEXT

        )
    """)


    connection.commit()

    connection.close()


    print(
        "Database and tables are ready!"
    )


# =========================================================
# MAIN DASHBOARD
# =========================================================

@app.route("/")
def home():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    # -----------------------------------------
    # Registered persons
    # -----------------------------------------

    cursor.execute("""
        SELECT *
        FROM missing_persons
        ORDER BY id DESC
    """)

    persons = cursor.fetchall()


    # -----------------------------------------
    # Alerts
    # -----------------------------------------

    cursor.execute("""
        SELECT *
        FROM alerts
        ORDER BY id DESC
    """)

    alerts = cursor.fetchall()


    registered_count = len(
        persons
    )

    alert_count = len(
        alerts
    )


    connection.close()


    return render_template(

        "index.html",

        persons=persons,

        alerts=alerts,

        registered_count=
            registered_count,

        alert_count=
            alert_count

    )


# =========================================================
# ALERT API
# =========================================================

@app.route("/api/alerts")
def get_alerts():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    cursor.execute("""
        SELECT
            id,
            person_name,
            camera_name,
            match_time,
            status
        FROM alerts
        ORDER BY id DESC
    """)


    alerts = cursor.fetchall()

    connection.close()


    alert_list = []


    for alert in alerts:

        alert_list.append({

            "id": alert[0],

            "person_name": alert[1],

            "camera_name": alert[2],

            "match_time": alert[3],

            "status": alert[4]

        })


    return jsonify(
        alert_list
    )


# =========================================================
# CAMERA HEALTH API
# =========================================================

@app.route(
    "/api/camera-health"
)
def camera_health():

    default_data = {

        "status": "OFFLINE",

        "camera_id": 0,

        "fps": 0,

        "faces": 0,

        "reconnects": 0,

        "last_frame": "Never",

        "last_update": 0

    }


    # -----------------------------------------
    # No health file
    # -----------------------------------------

    if not os.path.exists(
        HEALTH_FILE
    ):

        return jsonify(
            default_data
        )


    try:

        with open(
            HEALTH_FILE,
            "r"
        ) as file:

            data = json.load(
                file
            )


        # -----------------------------------------
        # Fill missing fields
        # -----------------------------------------

        for key in default_data:

            if key not in data:

                data[key] = (
                    default_data[key]
                )


        # -----------------------------------------
        # Check stale health data
        # -----------------------------------------

        current_time = time.time()

        last_update = data.get(
            "last_update",
            0
        )


        if (
            last_update == 0
            or
            current_time -
            last_update > 5
        ):

            data["status"] = "OFFLINE"

            data["fps"] = 0

            data["faces"] = 0


        return jsonify(
            data
        )


    except Exception as error:

        print(
            "Health file error:",
            error
        )

        return jsonify(
            default_data
        )


# =========================================================
# ACKNOWLEDGE ALERT
# =========================================================

@app.route(
    "/api/alerts/<int:alert_id>/acknowledge",
    methods=["POST"]
)
def acknowledge_alert(
    alert_id
):

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    cursor.execute("""
        UPDATE alerts

        SET status = ?

        WHERE id = ?
    """, (
        "Acknowledged",
        alert_id
    ))


    connection.commit()


    if cursor.rowcount == 0:

        connection.close()


        return jsonify({

            "success": False,

            "message":
                "Alert not found"

        }), 404


    connection.close()


    return jsonify({

        "success": True,

        "message":
            "Alert acknowledged"

    })


# =========================================================
# PERSON DETAILS
# =========================================================

@app.route(
    "/api/person/<name>"
)
def get_person(name):

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    cursor.execute("""
        SELECT
            id,
            name,
            age,
            description,
            photo_path

        FROM missing_persons

        WHERE name = ?

        ORDER BY id DESC

        LIMIT 1
    """, (name,))


    person = cursor.fetchone()

    connection.close()


    if not person:

        return jsonify({

            "error":
                "Person not found"

        }), 404


    return jsonify({

        "id": person[0],

        "name": person[1],

        "age": person[2],

        "description": person[3],

        "photo_path": person[4]

    })


# =========================================================
# PERSON PHOTO
# =========================================================

@app.route(
    "/person_photo/<person_id>"
)
def person_photo(person_id):

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    cursor.execute("""
        SELECT photo_path

        FROM missing_persons

        WHERE id = ?
    """, (person_id,))


    result = cursor.fetchone()

    connection.close()


    if not result:

        return (
            "Person not found",
            404
        )


    photo_path = result[0]


    if not os.path.exists(
        photo_path
    ):

        return (
            "Photo not found",
            404
        )


    folder = os.path.dirname(
        photo_path
    )

    filename = os.path.basename(
        photo_path
    )


    return send_from_directory(

        folder,

        filename

    )


# =========================================================
# REGISTER MISSING PERSON
# =========================================================

@app.route(
    "/register",
    methods=["POST"]
)
def register():

    name = request.form[
        "name"
    ].strip()

    age = request.form[
        "age"
    ]

    description = request.form[
        "description"
    ].strip()

    photo = request.files[
        "photo"
    ]


    if not name or not photo:

        return (
            "Name and photo are required!"
        )


    # -----------------------------------------
    # Person folder
    # -----------------------------------------

    folder_name = secure_filename(
        name
    )


    person_folder = os.path.join(

        FACE_FOLDER,

        folder_name

    )


    os.makedirs(

        person_folder,

        exist_ok=True

    )


    # -----------------------------------------
    # Photo
    # -----------------------------------------

    filename = secure_filename(

        photo.filename

    )


    photo_path = os.path.join(

        person_folder,

        filename

    )


    photo.save(
        photo_path
    )


    # -----------------------------------------
    # Database
    # -----------------------------------------

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    cursor.execute("""
        INSERT INTO missing_persons
        (
            name,
            age,
            description,
            photo_path
        )

        VALUES (?, ?, ?, ?)
    """, (

        name,

        age,

        description,

        photo_path

    ))


    connection.commit()

    connection.close()


    return redirect("/")


# =========================================================
# LIVE VIDEO STREAM
# =========================================================

@app.route(
    "/video_feed"
)
def video_feed():

    def generate():

        while True:

            try:

                # -----------------------------------------
                # Check frame exists
                # -----------------------------------------

                if os.path.exists(
                    FRAME_FILE
                ):

                    with open(
                        FRAME_FILE,
                        "rb"
                    ) as file:

                        frame = file.read()


                    if frame:

                        yield (

                            b"--frame\r\n"

                            b"Content-Type: "
                            b"image/jpeg\r\n"

                            b"Content-Length: "
                            + str(
                                len(frame)
                            ).encode()

                            + b"\r\n\r\n"

                            + frame

                            + b"\r\n"

                        )


            except Exception as e:

                print(
                    "Video stream error:",
                    e
                )


            time.sleep(
                0.08
            )


    return Response(

        generate(),

        mimetype=
            "multipart/x-mixed-replace; "
            "boundary=frame",

        headers={

            "Cache-Control":
                "no-cache",

            "Pragma":
                "no-cache"

        }

    )


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    create_database()


    print(
        "========================================"
    )

    print(
        "       TEMPLE AI SERVER"
    )

    print(
        "========================================"
    )

    print(
        "Dashboard:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "========================================"
    )


    app.run(

        debug=True,

        use_reloader=False,

        host="0.0.0.0",

        port=5000

    )