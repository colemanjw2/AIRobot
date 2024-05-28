import os
import time
import pygame
import serial
import tempfile
import subprocess
import threading
import speech_recognition as sr
from google.cloud import speech
from google.oauth2 import service_account
from openai import OpenAI
import random
import re
import signal
import sys
import cv2
import sounddevice
from contextlib import contextmanager
from flask import Flask, Response, render_template
from google.cloud import vision
from google.cloud.vision_v1 import types
from object_detection_module import ObjectDetector
import json


# Constants
MIC_NAME = 'HD Pro Webcam C920'
MIC_INDEX = None
GOOGLE_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SERIAL_PORTS = ['/usr/local/arduinoeyes', '/usr/local/arduinomouth', '/usr/local/arduinocylon']
SERIAL_BAUD_RATE = 115200
ROBOTIC_SOUNDS_DIR = '/Users/jcoleman/Documents/ai_robot/makerspacepi5/robot_sounds'  # Update this path as needed

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH

# Conversation history
conversation_history = []

# Flask app setup
app = Flask(__name__)

# Initialize speech recognizer and microphone index
r = sr.Recognizer()
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Microphone {index}: {name}")  # Debug information to list all microphones
    if MIC_NAME in name:
        MIC_INDEX = index
        break
if MIC_INDEX is None:
    raise Exception(f"Microphone named {MIC_NAME} not found.")
else:
    print(f"Using microphone '{MIC_NAME}' at index {MIC_INDEX}")

# Setup serial connections
ser = []
for port in SERIAL_PORTS:
    try:
        ser.append(serial.Serial(port, SERIAL_BAUD_RATE, timeout=1))
    except serial.SerialException as e:
        print(f"Error connecting to {port}: {e}")
time.sleep(2)

# Initialize shared camera
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    raise Exception("Could not open video device")

# Set a lower resolution and frame rate for the camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

# Initialize Object Detector with the same VideoCapture object
object_detector = ObjectDetector(cap, camera_index)

# Define global variables
speaking_event = threading.Event()
isAwake = False
human_detected_counter = 0
human_not_detected_counter = 0

def clean_up():
    cap.release()
    object_detector.release()

def signal_handler(signal, frame):
    print("Exiting...")
    clean_up()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Google service account credentials
credentials = service_account.Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH)
speech_client = speech.SpeechClient(credentials=credentials)
vision_client = vision.ImageAnnotatorClient(credentials=credentials)
client = OpenAI(api_key=OPENAI_API_KEY)

@contextmanager
def safe_microphone(device_index=None):
    mic = sr.Microphone(device_index=device_index)
    recognizer = sr.Recognizer()
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)  # Increase duration for better calibration
            print("Adjusted to ambient noise.")
            yield source, recognizer
    except Exception as e:
        print(f"Error initializing microphone: {str(e)}")
        yield None, None

def play_robotic_sound(directory_path):
    def run():
        print("Thread started")  # Debug print
        files = [f for f in os.listdir(directory_path) if f.endswith('.mp3')]
        if not files:
            print("No MP3 files found in the directory.")
            return
        audio_file_path = os.path.join(directory_path, random.choice(files))
        print(f"Playing sound from: {audio_file_path}")  # Debug print
        
        # Initialize pygame mixer
        pygame.mixer.init()
        print("Pygame mixer initialized")  # Debug print
        
        # Load and play the sound
        pygame.mixer.music.load(audio_file_path)
        print("Music loaded")  # Debug print
        pygame.mixer.music.play()
        print("Music playing")  # Debug print
        
        # Wait for the music to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        print("Music finished")  # Debug print

    threading.Thread(target=run).start()

def play_audio_ffplay(filename, interruptible=False):
    global speaking_event
    process = subprocess.Popen(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    speaking_event.clear()

    def wait_for_process():
        process.wait()
        speaking_event.set()
        print("Speaking event set")

    threading.Thread(target=wait_for_process).start()

    if interruptible:
        while process.poll() is None:
            time.sleep(0.1)
    else:
        process.wait()

def speak_text_openai(text, interruptible=False):
    text = re.sub(r'[#&*()_]', '', text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="onyx",
        input=text,
        response_format="opus",
    ) as response:
        response.stream_to_file(temp_file.name)

    for s in ser:
        s.write(b'WAKE\n')
    ser[0].write(b'CONVERSE\n')
    play_audio_ffplay(temp_file.name, interruptible)
    os.unlink(temp_file.name)

    for s in ser:
        s.write(b'SLEEP\n')

def recognize_speech(source, recognizer, speech_client, stop_listening_event, timeout=5):
    global isAwake
    try:
        for s in ser:
            s.write(b'SLEEP\n')
        speaking_event.wait()
        print("Listening for input...")
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
        print("Audio captured.")

        if stop_listening_event.is_set():
            print("Listening stopped by user.")
            return "", False

        audio_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
        audio_content = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        )
        response = speech_client.recognize(config=config, audio=audio_content)

        if response.results and response.results[0].alternatives:
            transcript = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            print(f"Transcript: '{transcript}' with confidence {confidence:.2f}")
            return transcript, confidence > 0.5
        else:
            print("No speech detected. Triggering timeout.")
            isAwake = False
            return "", False
    except sr.WaitTimeoutError:
        print("Listening timed out.")
        isAwake = False
        return "", False
    except Exception as e:
        print(f"Error during recognition: {str(e)}")
        isAwake = False
        return "", False

def generate_motion_message():
    try:
        global conversation_history
        prompt = "You are a robot and someone has walked into your field of view. Be very dude-like and/or funny in your greeting and ask how you can help."
        system_message = {"role": "system", "content": prompt}
        response = client.chat.completions.create(model="gpt-4o", messages=[system_message])
        motion_message = response.choices[0].message.content
        return {"role": "assistant", "content": motion_message}
    except Exception as e:
        print(f"Error generating motion message: {str(e)}")
        return {"role": "assistant", "content": "I see you! If you need help, just say 'hello computer'."}

def detect_human_position():
    detected_objects, frame = object_detector.detect_objects(size_threshold=0.70)
    human_position = None
    human_confidence_threshold = 0.7  # Adjust as needed

    for obj in detected_objects:
        if obj[0].lower() == 'person' and obj[1] > human_confidence_threshold:
            human_bbox = obj[2]
            # Calculate the center of the bounding box
            x_center = sum([point[0] for point in human_bbox]) / len(human_bbox)
            y_center = sum([point[1] for point in human_bbox]) / len(human_bbox)
            human_position = (x_center, y_center)

            # Draw bounding box for humans
            cv2.rectangle(frame, (int(human_bbox[0][0] * frame.shape[1]), int(human_bbox[0][1] * frame.shape[0])), 
                          (int(human_bbox[2][0] * frame.shape[1]), int(human_bbox[2][1] * frame.shape[0])), (0, 255, 0), 2)

        else:
            # Draw bounding box for other objects
            bounding_poly = obj[2]
            cv2.rectangle(frame, (int(bounding_poly[0][0] * frame.shape[1]), int(bounding_poly[0][1] * frame.shape[0])), 
                          (int(bounding_poly[2][0] * frame.shape[1]), int(bounding_poly[2][1] * frame.shape[0])), (255, 0, 0), 2)

    return human_position, frame

def map_to_dac(value, in_min, in_max, out_min, out_max):
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def get_dac_values(x_center, y_center):
    dac_x = map_to_dac(x_center, 0, 1, 0, 4095)  # 12-bit DAC range 0-4095
    dac_y = map_to_dac(y_center, 0, 1, 0, 4095)
    return dac_x, dac_y

def send_coordinates_to_arduino(dac_x, dac_y):
    command = f"MOVE {dac_x},{dac_y}\n"
    ser[0].write(command.encode())  # Assuming the Arduino for eye control is on ser[0]

def listen_and_respond(source, recognizer, speech_client, stop_listening_event):
    global isAwake, conversation_history
    while isAwake:
        transcript, success = recognize_speech(source, recognizer, speech_client, stop_listening_event)
        if success:
            print(f"You said: {transcript}")
            for s in ser:
                s.write(b'THINK\n')
            current_message = {"role": "user", "content": transcript}
            conversation_history.append(current_message)
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

            special_response = handle_specific_query(transcript)
            if special_response:
                speak_text_openai(special_response, interruptible=True)
                conversation_history.append({"role": "assistant", "content": special_response})
            elif any(keyword in transcript.lower() for keyword in ["identify object", "what is this", "can you see", "what am i holding"]):
                handle_object_detection()
            else:
                handle_regular_query()

            while True:
                for s in ser:
                    s.write(b'WAKE\n')
                speak_text_openai("Is there anything else I can help you with?", interruptible=True)
                speaking_event.wait()
                for s in ser:
                    s.write(b'WAKE\n')

                transcript, success = recognize_speech(source, recognizer, speech_client, stop_listening_event, timeout=10)
                if success:
                    print(f"You said: {transcript}")
                    for s in ser:
                        s.write(b'THINK\n')
                    current_message = {"role": "user", "content": transcript}
                    conversation_history.append(current_message)
                    if len(conversation_history) > 20:
                        conversation_history = conversation_history[-20:]

                    special_response = handle_specific_query(transcript)
                    if special_response:
                        speak_text_openai(special_response, interruptible=True)
                        conversation_history.append({"role": "assistant", "content": special_response})
                    elif any(keyword in transcript.lower() for keyword in ["identify object", "what is this", "can you see", "what am i holding"]):
                        handle_object_detection()
                    else:
                        handle_regular_query()
                else:
                    speak_text_openai("Alright, guess I'm off to bed then. Just approach if you need anything else.", interruptible=True)
                    speaking_event.wait()
                    isAwake = False
                    for s in ser:
                        s.write(b'CONVERSE\n')
                    break
        else:
            if transcript:
                print(f"Detected speech with low confidence: {transcript}")
            else:
                print("No understandable audio detected, trying again...")
                isAwake = False



# Function to handle specific queries based on predefined makerspace query
import re

def handle_specific_query(query):
    print(f"Debug: Original Query - {query}")
    query = query.lower()  # Normalize the case
    print(f"Debug: Normalized Query - {query}")

    # Simplified pattern to increase flexibility in activation phrases
    pattern = r"(middle\s*school.*maker\s*space|maker\s*space.*middle\s*school)"
    print(f"Debug: Regex pattern used - {pattern}")

    if re.search(pattern, query):
        try:
            with open('makerspace_facts.txt', 'r') as file:
                facts = file.read()
            prompt = f"Here are some facts about the Middle School Makerspace at the Singapore American School: {facts}"
            print(f"Debug: Using Makerspace text prompt for OpenAI")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}]
            )
            formatted_response = response.choices[0].message.content
            for s in ser:
                s.write(b'THOUGHT\n')
                s.write(b'WAKE\n')
            return formatted_response
        except Exception as e:
            print(f"Error accessing or processing the facts file: {str(e)}")
            return "I am currently unable to retrieve information about the Makerspace. Please try again later."
    else:
        print("Debug: No specific query detected based on keywords.")
        return None

def handle_object_detection():
    print("Identifying object...")
    try:
        detected_objects, frame = object_detector.detect_objects(size_threshold=0.1)
        object_confidence_threshold = 0.5  # Adjust as needed

        # Filter out non-person objects with sufficient confidence
        nearby_objects = [obj for obj in detected_objects if obj[0].lower() != 'person' and obj[1] > object_confidence_threshold]

        if nearby_objects:
            object_names = ", ".join([obj[0] for obj in nearby_objects if obj[0].lower() not in ["clothing", "computer monitor", "cabinet"]])
            object_description = f"I see the following objects: {object_names}."
            print(object_description)

            prompt = f"You are presented an object and you need to list off these items: {object_names}. Say 'I see' the object names.  If it's clothing, computer monitors, or cabinets, do not mention it in the list."
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            detailed_description = response.choices[0].message.content

            # Add to conversation history for context retention
            conversation_history.append({"role": "system", "content": object_description})
            conversation_history.append({"role": "assistant", "content": detailed_description})

            for s in ser:
                s.write(b'THOUGHT\n')
                s.write(b'WAKE\n')

            print(detailed_description)
            speak_text_openai(detailed_description, interruptible=True)
        else:
            speak_text_openai("I'm sorry, I couldn't identify any objects.", interruptible=True)
            for s in ser:
                s.write(b'THOUGHT\n')
                s.write(b'WAKE\n')
    except Exception as e:
        print(f"Error identifying object: {e}")


def handle_regular_query():
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history
        )
        response_text = response.choices[0].message.content
        print(response_text)
        conversation_history.append({"role": "assistant", "content": response_text})
        for s in ser:
            s.write(b'THINK\n')  # Ensure THINK command is sent during processing
            s.write(b'THOUGHT\n')
        for s in ser:
            s.write(b'WAKE\n')
        speak_text_openai(response_text, interruptible=True)
        speaking_event.wait()  # Wait until speaking is done
    except Exception as e:
        print(f"Error during chat completion: {str(e)}")
        time.sleep(4)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames(source, recognizer, speech_client, stop_listening_event):
    global isAwake, human_detected_counter, human_not_detected_counter
    while True:
        human_position, frame = detect_human_position()
        if human_position is not None:
            human_detected_counter += 1
            human_not_detected_counter = 0  # Reset the counter when a human is detected
            x_center, y_center = human_position
            if human_detected_counter > 2:  # Ensure human presence over multiple frames
                if not isAwake:
                    print(f"Human detected at: x={x_center}, y={y_center}")
                    dac_x, dac_y = get_dac_values(x_center, y_center)
                    send_coordinates_to_arduino(dac_x, dac_y)
                    motion_message = generate_motion_message()
                    conversation_history.append(motion_message)
                    speak_text_openai(motion_message['content'], interruptible=False)
                    speaking_event.wait()  # Wait until speaking is done
                    isAwake = True

                while isAwake:
                    human_position, frame = detect_human_position()
                    if human_position is not None:
                        human_detected_counter += 1
                        human_not_detected_counter = 0
                        x_center, y_center = human_position
                        dac_x, dac_y = get_dac_values(x_center, y_center)
                        send_coordinates_to_arduino(dac_x, dac_y)
                        listen_and_respond(source, recognizer, speech_client, stop_listening_event)
                    else:
                        human_not_detected_counter += 1
                        if human_not_detected_counter > 2:  # Ensure human absence over multiple frames
                            isAwake = False  # Reset isAwake if no human is detected
                            break
        else:
            human_not_detected_counter += 1
            if human_not_detected_counter > 2:
                if isAwake:
                    isAwake = False  # Reset isAwake if no human is detected
                time.sleep(1)  # Wait a moment before checking again

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    global source, recognizer, speech_client, stop_listening_event
    return Response(generate_frames(source, recognizer, speech_client, stop_listening_event), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    for s in ser:
        s.write(b'SLEEP\n')
    speaking_event.set()  # Set the event initially to allow the main loop to start
    stop_listening_event = threading.Event()
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, stop_listening_event))
    print("Starting the voice assistant...")
    
    with safe_microphone(device_index=MIC_INDEX) as (source, recognizer):
        if source is None or recognizer is None:
            print("Error initializing microphone")
        else:
            app.run(host='0.0.0.0', port=5000)
