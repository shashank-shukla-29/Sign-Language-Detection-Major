import cv2
import time

backends = [
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
    ("Default", cv2.CAP_ANY),
]

for name, backend in backends:
    print(f"\nTesting {name}...")

    cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        print("Camera failed to open.")
        cap.release()
        continue

    print("Camera opened. Waiting for frames...")

    success_count = 0

    for i in range(30):
        ret, frame = cap.read()

        if ret and frame is not None:
            success_count += 1
            cv2.imshow("Webcam Test", frame)

        cv2.waitKey(30)

    print(f"Successful frames: {success_count}/30")

    cap.release()
    cv2.destroyAllWindows()

    if success_count > 0:
        print(f"\nWORKING BACKEND: {name}")
        break

print("\nCamera backend testing completed.")