import cv2
import torch
import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import pytesseract

# ---------------- SPEAK FUNCTION ---------------- #
def speak(text):
    print(f"🗣 Speaking: {text}")
    tts = gTTS(text=text, lang='en')
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

# ---------------- OBJECT DETECTION (single run) ---------------- #
def identify_object():
    cap = cv2.VideoCapture(0)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Capture several frames for better detection
    for _ in range(5):
        ret, frame = cap.read()
    cap.release()

    if not ret:
        speak("Sorry, I could not access the camera.")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)

    df = results.pandas().xyxy[0]
    if not df.empty:
        labels = df['name'].tolist()
        unique_objects = list(set(labels))
        objects_str = ", ".join(unique_objects)
        speak(f"I can see: {objects_str}")
        # Optional: Show frame with detections for debugging
        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow("Detections", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    else:
        speak("I did not detect any objects.")

# ---------------- TEXT READING (OCR) ---------------- #
def read_text():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        speak("Sorry, I could not access the camera.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    if text.strip():
        speak(f"The text says: {text}")
    else:
        speak("No readable text found.")

# ---------------- NAVIGATION (simple obstacle detection) ---------------- #
def navigate():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        speak("Sorry, I could not access the camera.")
        return

    # Use YOLOv5 to detect obstacles (person, chair, etc.)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    df = results.pandas().xyxy[0]

    obstacles = df[df['name'].isin(['person', 'chair', 'bench', 'car', 'bicycle'])]
    if not obstacles.empty:
        speak("Obstacle detected ahead. Please be careful.")
    else:
        speak("Path seems clear.")

# ---------------- SPEECH RECOGNITION LOOP ---------------- #
def listen_for_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    speak("SightMate++ is ready. Say read, identify, navigate, or exit.")

    while True:
        with mic as source:
            print("🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"✅ Recognized: {command}")

            if "identify" in command:
                identify_object()

            elif "exit" in command or "quit" in command:
                speak("Goodbye from SightMate++.")
                break

            elif "read" in command:
                read_text()

            elif "navigate" in command:
                navigate()

            else:
                speak("Sorry, I did not understand. Please say read, identify, navigate, or exit.")

        except sr.UnknownValueError:
            print("❌ Could not understand")
        except sr.RequestError:
            print("⚠️ Could not request results, check internet connection")

# ---------------- MAIN ---------------- #
def main():
    listen_for_commands()

if __name__ == "__main__":
    main()
