import serial
import joblib
import numpy as np

# Load the trained model
model = joblib.load("nerve_signal_model.pkl")

# Connect to Arduino
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)

normal_signals = []

print("Listening for EMG data...")

while True:
    try:
        data = arduino.readline().decode('utf-8').strip()
        if data:
            value = int(data)
            prediction = model.predict(np.array([[value]]))  # Predict normal/abnormal
            
            if prediction[0] == 0:
                normal_signals.append(value)
            else:
                closest_normal = min(normal_signals, key=lambda x: abs(x - value)) if normal_signals else value
                print(f"Abnormal: {value}, Closest Normal: {closest_normal}")
                arduino.write(f"{closest_normal}\n".encode())

    except Exception as e:
        print("Error:", e)
