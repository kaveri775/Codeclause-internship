# Import required libraries
import speech_recognition as sr       # For speech recognition
import pyttsx3                        # For text-to-speech

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize recognizer
recognizer = sr.Recognizer()

# Use the microphone as source for input
with sr.Microphone() as source:
    print("Say something...")
    
    # Listen for the user's input
    audio = recognizer.listen(source)

    try:
        # Convert speech to text using Google's API
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        
        # Speak back what the user said
        engine.say("You said " + text)
        engine.runAndWait()

        # Command-based logic
        if "hello" in text.lower():
            print("Hi there!")
            engine.say("Hi there!")
        elif "open" in text.lower():
            print("Opening something...")
            engine.say("Opening something")
        elif "stop" in text.lower():
            print("Stopping the operation...")
            engine.say("Stopping the operation")
        else:
            print("Command not recognized.")
            engine.say("Command not recognized.")

        # Run the TTS engine
        engine.runAndWait()

    # Handle case where speech was not understood
    except sr.UnknownValueError:
        print("Sorry, I could not understand your speech.")
        engine.say("Sorry, I could not understand your speech.")
        engine.runAndWait()

    # Handle case where Google API fails
    except sr.RequestError:
        print("Could not request results, check your internet.")
        engine.say("Could not request results. Check your internet connection.")
        engine.runAndWait()
