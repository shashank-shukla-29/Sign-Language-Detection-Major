
# ============================================================
# COMBINED ISL RECOGNITION SYSTEM
# WORDS (8 classes) + ALPHABETS (35 classes)
# Windows-compatible version
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import math
import time
import threading
import queue
from tensorflow import keras
from cvzone.HandTrackingModule import HandDetector


# ============================================================
# 1. CONFIGURATION
# ============================================================

CUSTOM_MODEL_PATH = "Model/keras_model_fixed.h5"
CUSTOM_LABELS_PATH = "Model/labels.txt"

KAGGLE_MODEL_PATH = "Model/kaggle_model_fixed.h5"
KAGGLE_LABELS_PATH = "Model/kaggle_labels.txt"

CAMERA_INDEX = 0
OFFSET = 20
IMG_SIZE = 300

CONFIDENCE_THRESHOLD = 0.60
DEBOUNCE_FRAMES = 15
COOLDOWN_SECONDS = 3.0

WORDS = "WORDS"
ALPHABETS = "ALPHABETS"


# ============================================================
# 2. COMPATIBILITY FOR FIXED H5 MODELS
# ============================================================

class CompatibleInputLayer(keras.layers.InputLayer):

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop("optional", None)

        if "batch_shape" in config:
            config["batch_input_shape"] = config.pop("batch_shape")

        return cls(**config)


CUSTOM_OBJECTS = {
    "InputLayer": CompatibleInputLayer,
    "CompatibleInputLayer": CompatibleInputLayer,
    "keras.layers.InputLayer": CompatibleInputLayer,
}


# ============================================================
# 3. LOAD LABELS AND MODELS
# ============================================================

def load_labels(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Labels file not found: {filepath}"
        )

    labels = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(line)

    if not labels:
        raise ValueError(f"No labels found in {filepath}")

    return labels


def load_h5_model(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    print(f"\nLoading model: {model_path}")

    model = keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=CUSTOM_OBJECTS
    )

    print("Model loaded.")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    return model


print("\n" + "=" * 55)
print("INITIALIZING COMBINED ISL RECOGNITION")
print("=" * 55)

print("\nLoading custom word model...")
model_custom = load_h5_model(CUSTOM_MODEL_PATH)
labels_custom = load_labels(CUSTOM_LABELS_PATH)

print("\nLoading Kaggle alphabet model...")
model_kaggle = load_h5_model(KAGGLE_MODEL_PATH)
labels_kaggle = load_labels(KAGGLE_LABELS_PATH)

if model_custom.output_shape[-1] != len(labels_custom):
    raise ValueError(
        f"Word model has {model_custom.output_shape[-1]} outputs "
        f"but labels.txt contains {len(labels_custom)} labels."
    )

if model_kaggle.output_shape[-1] != len(labels_kaggle):
    raise ValueError(
        f"Kaggle model has {model_kaggle.output_shape[-1]} outputs "
        f"but kaggle_labels.txt contains {len(labels_kaggle)} labels."
    )

print("\nBoth models and labels loaded successfully!")


# ============================================================
# 4. HAND DETECTORS
# ============================================================

detector_1hand = HandDetector(
    maxHands=1,
    detectionCon=0.7,
    minTrackCon=0.5
)

detector_2hands = HandDetector(
    maxHands=2,
    detectionCon=0.7,
    minTrackCon=0.5
)


# ============================================================
# 5. WINDOWS TEXT-TO-SPEECH
# ============================================================

speech_queue = queue.Queue()
tts_available = False
tts_engine = None

try:
    import pyttsx3

    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 150)

    tts_available = True
    print("Text-to-speech is ready.")

except Exception as error:
    print("Text-to-speech unavailable.")
    print("Recognized labels will print instead.")
    print("Details:", error)


def speech_worker():
    while True:
        text_to_speak = speech_queue.get()

        try:
            if text_to_speak is None:
                return

            if tts_available and tts_engine is not None:
                tts_engine.say(text_to_speak)
                tts_engine.runAndWait()
            else:
                print("Detected:", text_to_speak)

        except Exception as error:
            print("Speech error:", error)

        finally:
            speech_queue.task_done()


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

speech_thread.start()


def speak(text_to_speak):
    if text_to_speak:
        speech_queue.put(text_to_speak)


# ============================================================
# 6. PREPROCESSING
# ============================================================

def preprocess_words(img_crop):

    if img_crop is None or img_crop.size == 0:
        return None, None

    h, w = img_crop.shape[:2]

    if h == 0 or w == 0:
        return None, None

    img_white = np.ones(
        (IMG_SIZE, IMG_SIZE, 3),
        dtype=np.uint8
    ) * 255

    aspect_ratio = h / w

    if aspect_ratio > 1:

        scale = IMG_SIZE / h

        width_calculated = max(
            1, int(math.ceil(scale * w))
        )

        width_calculated = min(
            width_calculated, IMG_SIZE
        )

        img_resize = cv2.resize(
            img_crop,
            (width_calculated, IMG_SIZE)
        )

        width_gap = (IMG_SIZE - width_calculated) // 2

        img_white[
            :, width_gap:width_gap + width_calculated
        ] = img_resize

    else:

        scale = IMG_SIZE / w

        height_calculated = max(
            1, int(math.ceil(scale * h))
        )

        height_calculated = min(
            height_calculated, IMG_SIZE
        )

        img_resize = cv2.resize(
            img_crop,
            (IMG_SIZE, height_calculated)
        )

        height_gap = (IMG_SIZE - height_calculated) // 2

        img_white[
            height_gap:height_gap + height_calculated, :
        ] = img_resize

    input_h = model_custom.input_shape[1]
    input_w = model_custom.input_shape[2]

    resized = cv2.resize(
        img_white,
        (input_w, input_h)
    )

    normalized = resized.astype(np.float32) / 255.0

    return np.expand_dims(normalized, axis=0), img_white


def preprocess_alphabets(img_crop):

    if img_crop is None or img_crop.size == 0:
        return None

    input_h = model_kaggle.input_shape[1]
    input_w = model_kaggle.input_shape[2]

    resized = cv2.resize(
        img_crop,
        (input_w, input_h)
    )

    normalized = resized.astype(np.float32) / 255.0

    return np.expand_dims(normalized, axis=0)


# ============================================================
# 7. PREDICTION
# ============================================================

def predict(model, img_batch, labels):

    if img_batch is None:
        return "", 0.0

    try:
        predictions = model.predict(
            img_batch,
            verbose=0
        )[0]

        class_index = int(np.argmax(predictions))
        confidence = float(predictions[class_index])

        if class_index >= len(labels):
            return "", 0.0

        return labels[class_index], confidence

    except Exception as error:
        print("Prediction error:", error)
        return "", 0.0


# ============================================================
# 8. START CAMERA
# ============================================================

print("\nOpening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    cap.release()
    raise RuntimeError(
        "Unable to open webcam. "
        "Close other camera applications."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Webcam opened. Waiting for frames...")

success = False
img = None

# Allow the webcam to initialize and retry frame capture.
for attempt in range(30):

    success, img = cap.read()

    if success and img is not None:
        break

    time.sleep(0.2)

if not success or img is None:
    cap.release()
    raise RuntimeError(
        "Webcam opened but no frames received. "
        "Close other camera applications and try again."
    )

print("Webcam frame captured successfully!")

current_mode = WORDS
counter = 0
candidate_label = ""
last_spoken_label = ""
last_speech_time = 0.0


# ============================================================
# 9. MAIN LOOP
# ============================================================

print("\n" + "=" * 55)
print("COMBINED ISL RECOGNITION STARTED")
print("SPACE : Switch between WORDS and ALPHABETS")
print("Q     : Quit")
print("=" * 55)

failed_frames = 0

try:

    while True:

        success, img = cap.read()

        if not success or img is None:

            failed_frames += 1

            if failed_frames >= 30:
                print(
                    "Camera failed to deliver 30 consecutive frames."
                )
                break

            time.sleep(0.05)
            continue

        failed_frames = 0

        img = cv2.flip(img, 1)
        img_output = img.copy()

        detected_label = ""
        confidence = 0.0

        # ----------------------------------------------------
        # WORDS MODE
        # ----------------------------------------------------

        if current_mode == WORDS:

            hands, _ = detector_1hand.findHands(
                img,
                draw=False
            )

            if hands:

                x, y, w, h = hands[0]["bbox"]

                y1 = max(0, y - OFFSET)
                y2 = min(img.shape[0], y + h + OFFSET)

                x1 = max(0, x - OFFSET)
                x2 = min(img.shape[1], x + w + OFFSET)

                img_crop = img[y1:y2, x1:x2]

                img_batch, img_white = preprocess_words(
                    img_crop
                )

                if img_batch is not None:

                    detected_label, confidence = predict(
                        model_custom,
                        img_batch,
                        labels_custom
                    )

                    cv2.rectangle(
                        img_output,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 255),
                        3
                    )

                    cv2.putText(
                        img_output,
                        f"{detected_label} {confidence * 100:.1f}%",
                        (x1, max(30, y1 - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    cv2.imshow(
                        "Model Input - Words",
                        img_white
                    )

        # ----------------------------------------------------
        # ALPHABETS MODE
        # ----------------------------------------------------

        else:

            hands, _ = detector_2hands.findHands(
                img,
                draw=False
            )

            if hands:

                x_min = min(
                    hand["bbox"][0]
                    for hand in hands
                )

                y_min = min(
                    hand["bbox"][1]
                    for hand in hands
                )

                x_max = max(
                    hand["bbox"][0] + hand["bbox"][2]
                    for hand in hands
                )

                y_max = max(
                    hand["bbox"][1] + hand["bbox"][3]
                    for hand in hands
                )

                x1 = max(0, x_min - OFFSET)
                y1 = max(0, y_min - OFFSET)

                x2 = min(
                    img.shape[1],
                    x_max + OFFSET
                )

                y2 = min(
                    img.shape[0],
                    y_max + OFFSET
                )

                img_crop = img[y1:y2, x1:x2]

                img_batch = preprocess_alphabets(
                    img_crop
                )

                if img_batch is not None:

                    detected_label, confidence = predict(
                        model_kaggle,
                        img_batch,
                        labels_kaggle
                    )

                    cv2.rectangle(
                        img_output,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        img_output,
                        f"{detected_label} {confidence * 100:.1f}%",
                        (x1, max(30, y1 - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    cv2.imshow(
                        "Model Input - Alphabets",
                        img_crop
                    )

        # ----------------------------------------------------
        # STABLE RECOGNITION AND SPEECH
        # ----------------------------------------------------

        now = time.time()

        if detected_label and confidence >= CONFIDENCE_THRESHOLD:

            if detected_label == candidate_label:
                counter += 1
            else:
                candidate_label = detected_label
                counter = 1

            if (
                counter >= DEBOUNCE_FRAMES
                and detected_label != last_spoken_label
                and now - last_speech_time >= COOLDOWN_SECONDS
            ):

                print(
                    f"Recognized: {detected_label} "
                    f"({confidence * 100:.1f}%)"
                )

                speak(detected_label)

                last_spoken_label = detected_label
                last_speech_time = now
                counter = 0

        else:
            counter = 0
            candidate_label = ""

        # ----------------------------------------------------
        # DISPLAY MODE AND INSTRUCTIONS
        # ----------------------------------------------------

        mode_color = (
            (255, 0, 255)
            if current_mode == WORDS
            else (0, 255, 0)
        )

        cv2.rectangle(
            img_output,
            (0, 0),
            (img_output.shape[1], 50),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            img_output,
            f"MODE: {current_mode}",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            mode_color,
            2
        )

        cv2.putText(
            img_output,
            "SPACE: Switch Mode | Q: Quit",
            (260, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "Combined ISL Recognition",
            img_output
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key == ord(" "):

            current_mode = (
                ALPHABETS
                if current_mode == WORDS
                else WORDS
            )

            counter = 0
            candidate_label = ""
            last_spoken_label = ""

            print(f"\nSwitched to {current_mode} mode")

except KeyboardInterrupt:
    print("\nRecognition interrupted by user.")

except Exception as error:
    print("\nUnexpected error:", error)

finally:

    cap.release()
    cv2.destroyAllWindows()

    speech_queue.put(None)
    speech_queue.join()

    speech_thread.join(timeout=5)

    print("\nISL Recognition stopped.")