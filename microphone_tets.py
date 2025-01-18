import speech_recognition as sr

def test_microphone():
    recognizer = sr.Recognizer()

    try:
        # Use the default microphone
        with sr.Microphone() as source:
            print("Adjusting for ambient noise. Please wait...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Listening... Speak something into the microphone.")
            
            # Listen for user input
            audio = recognizer.listen(source)
            print("Processing your input...")
            
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            print(f"Recognized Speech: {text}")
    
    except sr.UnknownValueError:
        print("Speech was not understood. Try speaking more clearly.")
    except sr.RequestError as e:
        print(f"Could not request results from the speech recognition service; {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_microphone()