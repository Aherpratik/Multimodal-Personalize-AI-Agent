import random
import webbrowser
import pyttsx3
import speech_recognition as sr
from llama_cpp import Llama
import os
import json
from typing import Optional, Dict, Any
from datetime import datetime

class Config:
    @staticmethod
    def load_config() -> Dict[str, str]:
        """Load configuration from JSON file."""
        try:
            with open("config.json") as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            raise Exception("Configuration file 'config.json' not found.")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON in configuration file.")

class SpeechHandler:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()

    def speak(self, text: str) -> None:
        """Convert text to speech."""
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self) -> str:
        """Convert speech to text."""
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = self.recognizer.listen(source)
                command = self.recognizer.recognize_google(audio)
                print(f"You said: {command}")
                return command.lower()
            except sr.UnknownValueError:
                print("Sorry, I didn't understand that.")
                return ""
            except sr.RequestError as e:
                print(f"Error with the speech recognition service: {e}")
                return ""

class LlamaHandler:
    def __init__(self, model_path: str):
        """Initialize Llama model."""
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,  # Context window
                n_threads=4   # Number of CPU threads to use
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Llama model: {e}")

    def generate_response(self, prompt: str, system_message: str = "You are a helpful assistant.") -> str:
        """Generate response using Llama."""
        try:
            # Format the prompt similar to ChatML format
            formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant"""
            
            response = self.llm(
                formatted_prompt,
                max_tokens=512,
                stop=["<|im_end|>"],
                echo=False
            )
            
            # Extract the generated text
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your request."

    def summarize_email(self, email_content: str) -> str:
        """Summarize email content."""
        prompt = f"Summarize the following email concisely: {email_content}"
        return self.generate_response(prompt)

class CalendarHandler:
    @staticmethod
    def add_event(title: str, start_time: str, end_time: str) -> bool:
        """Add event to calendar (dummy implementation)."""
        try:
            datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
            datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
            print(f"Event '{title}' scheduled from {start_time} to {end_time}.")
            return True
        except ValueError:
            print("Invalid datetime format")
            return False

class WebBrowserHandler:
    @staticmethod
    def open_website(command: str) -> str:
        """Handle website opening commands."""
        websites = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com"
        }
        
        for site, url in websites.items():
            if site in command:
                webbrowser.open(url)
                return f"Opening {site.capitalize()}."
        return "I'm not sure which website to open."

class ConversationalAI:
    def __init__(self, model_path: str):
        self.speech_handler = SpeechHandler()
        self.llama_handler = LlamaHandler(model_path)
        self.calendar_handler = CalendarHandler()
        self.web_handler = WebBrowserHandler()
        self.conversation_context = {}

    def handle_schedule_event(self) -> None:
        """Handle event scheduling conversation flow."""
        self.speech_handler.speak("What is the event title?")
        title = self.speech_handler.listen()
        self.speech_handler.speak("What is the start time? (Format: YYYY-MM-DDThh:mm:ss)")
        start_time = self.speech_handler.listen()
        self.speech_handler.speak("What is the end time? (Format: YYYY-MM-DDThh:mm:ss)")
        end_time = self.speech_handler.listen()
        
        if self.calendar_handler.add_event(title, start_time, end_time):
            self.speech_handler.speak(f"Event '{title}' successfully scheduled.")
        else:
            self.speech_handler.speak("Failed to schedule event. Please check the time format.")

    def run(self):
        """Main conversation loop."""
        self.speech_handler.speak("Hello! How can I assist you today?")
        
        while True:
            command = self.speech_handler.listen()
            
            if not command:
                continue

            if "exit" in command or "bye" in command:
                response = self.llama_handler.generate_response("User said goodbye.")
                self.speech_handler.speak(response)
                break

            # Command handling
            if "hello" in command or "hi" in command:
                response = self.llama_handler.generate_response("User said hello.")
                self.speech_handler.speak(response)

            elif "summarize" in command and "email" in command:
                email_content = "This is an example email content for summarization."
                summary = self.llama_handler.summarize_email(email_content)
                self.speech_handler.speak(f"The summary is: {summary}")

            elif "schedule" in command and "event" in command:
                self.handle_schedule_event()

            elif "open" in command:
                result = self.web_handler.open_website(command)
                self.speech_handler.speak(result)

            else:
                response = self.llama_handler.generate_response(f"User said: {command}")
                self.speech_handler.speak(response)

def main():
    try:
        config = Config.load_config()
        model_path = config.get("LLAMA_MODEL_PATH")
        if not model_path:
            raise Exception("LLAMA_MODEL_PATH not found in config file")
            
        ai_agent = ConversationalAI(model_path)
        ai_agent.run()
    except Exception as e:
        print(f"Error initializing the AI agent: {e}")

if __name__ == "__main__":
    main()