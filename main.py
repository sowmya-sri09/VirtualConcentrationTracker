import cv2
import time
from focus_tracker import FocusTracker
from voice_alert import speak
from utils import log_focus
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
tracker = FocusTracker()

focus_count = 0
distract_count = 0
distracted_start = None
status_history = []
start_time = time.time()

spoken_status = None  # track last spoken state

print("✅ Webcam running. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    result = tracker.detect_all(frame)
    status = result["status"]

    now = time.time()
    elapsed_min = (now - start_time) / 60

    if status == "focused":
        cv2.putText(frame, "FOCUSED ✅", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        focus_count += 1
        log_focus("Focused")
        distracted_start = None
        if spoken_status != "focused":
            speak("You are focused")
            spoken_status = "focused"

    elif status == "distracted":
        cv2.putText(frame, "DISTRACTED ❌", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        distract_count += 1
        log_focus("Distracted")
        if distracted_start is None:
            distracted_start = time.time()
        elif now - distracted_start > 3 and spoken_status != "distracted":
            speak("You are distracted")
            spoken_status = "distracted"

    else:
        cv2.putText(frame, "NO FACE ❌", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        distract_count += 1
        log_focus("No Face")
        spoken_status = None

    # Log focus % over time
    total = focus_count + distract_count
    focus_pct = int((focus_count / total) * 100) if total else 100
    status_history.append((elapsed_min, focus_pct))

    # Live Focus Meter
    cv2.rectangle(frame, (10, 450), (10 + focus_pct * 5, 470), (0, 255, 0), -1)
    cv2.putText(frame, f"Focus: {focus_pct}%", (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Concentration Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 📈 Graph
x_vals, y_vals = zip(*status_history)
plt.plot(x_vals, y_vals, color='blue')
plt.xlabel('Time (min)')
plt.ylabel('Focus %')
plt.title('Focus vs Time')
plt.grid(True)
plt.savefig("focus_graph.png")
plt.show()

# 🎙️ Say final motivational quote
speak("Session complete. Here's a quote for you.")
speak("Believe you can, and you're halfway there.")


