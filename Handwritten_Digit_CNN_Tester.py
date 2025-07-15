import cv2
import numpy as np
from tensorflow.keras.models import load_model #type: ignore

model = load_model("Models and Datasets/Handwritten_Digit_CNN.keras")
camera = cv2.VideoCapture(0)
print("Webcam started! Press 'Q' anytime to exit")

while True:
    r, frame = camera.read()
    if not r:
        break

    x, y, w, h = 200, 100, 200, 200
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
    roi = frame[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # Converting to grayscale
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA) # Resizing the image
    roi = roi.astype('float32') / 255.0 # Normalizing the image
    roi = roi.reshape(1, 28, 28, 1) # Reshaping for the model input

    result = model.predict(roi, verbose=0)
    digit = np.argmax(result)
    confidence = np.max(result)

    cv2.putText(frame, f"Predicted Digit: {digit} (Confidence: {confidence*100:.2f}%)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Handwritten Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break
camera.release()
cv2.destroyAllWindows()