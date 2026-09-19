import cv2
import easyocr
import requests


# ==============================
# BASERA OCR
# ==============================

BACKEND_URL = "http://127.0.0.1:8001/ocr"


def send_to_backend(text):
    """Send OCR text to BASERA backend."""

    try:
        response = requests.post(
            BACKEND_URL,
            json={
                "text": text,
                "student_type": "blind",
                "subject": "general"
            },
            timeout=60
        )

        if response.status_code != 200:
            print("Backend error:", response.status_code)
            print(response.text)
            return

        data = response.json()

        print("\n==============================")
        print("BASERA RESPONSE")
        print("==============================")
        print(data.get("response", "No response"))
        print("==============================\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to BASERA backend.")
        print("Make sure backend is running on port 8001.\n")

    except Exception as e:
        print("\n❌ Error:", e)


def main():

    print("================================")
    print("       BASERA OCR MODULE")
    print("================================")

    print("\nLoading EasyOCR...")
    
    reader = easyocr.Reader(
        ["ar", "en"],
        gpu=False
    )

    print("EasyOCR ready.")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    print("\nCamera ready.")
    print("Press SPACE to scan text.")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("❌ Could not read camera frame.")
            break

        cv2.imshow("BASERA - OCR", frame)

        key = cv2.waitKey(1) & 0xFF

        # ==============================
        # SPACE → OCR
        # ==============================

        if key == 32:

            print("\nReading text...")

            results = reader.readtext(frame)

            if not results:
                print("❌ No text detected.")
                continue

            extracted_text = []

            for bbox, text, confidence in results:

                if confidence >= 0.30:
                    extracted_text.append(text)

            final_text = " ".join(extracted_text).strip()

            if not final_text:
                print("❌ No clear text detected.")
                continue

            print("\n==============================")
            print("DETECTED TEXT")
            print("==============================")
            print(final_text)
            print("==============================")

            # Send OCR result to backend
            send_to_backend(final_text)

        # ==============================
        # Q → EXIT
        # ==============================

        elif key == ord("q"):

            print("Closing OCR...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()