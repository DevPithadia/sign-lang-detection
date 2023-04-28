import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model(r'C:\Users\devku\Downloads\RealTimeObjectDetection-main\RealTimeObjectDetection-main\RealTimeObjectDetection-main\Tensorflow\workspace\models\my_ssd_mobnet')

# Define the class names for the sign language gestures
class_names = ['thankyou','hello','yes']

# Define the function to perform real-time sign language detection
def sign_language_detection():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    # Set the width and height of the video capture
    cap.set(3, 640)
    cap.set(4, 480)

    # Start the video capture
    while True:
        # Read the frame from the video capture
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale image to 28x28 pixels
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Reshape the image to be a single row of pixels
        input_image = np.reshape(resized, (1, 784))

        # Normalize the pixel values to be between 0 and 1
        input_image = input_image / 255.0

        # Use the pre-trained model to make a prediction
        predictions = model.predict(input_image)

        # Find the class name with the highest predicted probability
        class_index = np.argmax(predictions[0])
        class_name = class_names[class_index]

        # Display the class name on the video frame
        cv2.putText(frame, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the video frame with the class name
        cv2.imshow('Sign Language Detection', frame)

        # Check for the 'q' key to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

# Create a Streamlit app
st.title('Real-time Sign Language Detection')
st.write('Press the button to start detecting sign language gestures.')

# Define the button to start the sign language detection
if st.button('Start Detection'):
    sign_language_detection()
