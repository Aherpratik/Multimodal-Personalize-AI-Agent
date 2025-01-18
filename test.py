import random
import webbrowser
import pyttsx3
import speech_recognition as sr
from transformers import pipeline
import os
import json
from typing import Optional, Dict, Any
from datetime import datetime
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

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

class HuggingFaceHandler:
    def __init__(self):
        """Initialize the Hugging Face text generation pipeline."""
        try:
            self.generator = pipeline("text2text-generation", model="google/flan-t5-large")
            self.conversation_context = []  # Maintain conversation history
        except Exception as e:
            raise Exception(f"Failed to initialize Hugging Face model: {e}")

    def generate_response(self, user_input: str) -> str:
        """Generate conversational response with history."""
        try:
            # Add the user's input to the context
            self.conversation_context.append(f"User: {user_input}")

            # Build the prompt with context
            context = "\n".join(self.conversation_context[-5:])  # Use the last 5 exchanges for brevity
            prompt = f"The following is a conversation between a helpful assistant and a user:\n{context}\nAssistant:"

            # Generate a response
            response = self.generator(prompt, max_length=200, num_return_sequences=1)

            # Extract the generated text
            bot_response = response[0]["generated_text"].strip()

            # Add the assistant's response to the context
            self.conversation_context.append(f"Assistant: {bot_response}")

            return bot_response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your request."

    def summarize_email(self, email_content: str, sender: str) -> str:
        """Summarize email content and include sender information."""
        prompt = f"Summarize the following email from {sender} concisely: {email_content}"
        response = self.generator(prompt, max_length=100, num_return_sequences=1)
        return response[0]["generated_text"].strip()

class GmailHandler:
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

    def __init__(self):
        """Initialize Gmail API."""
        self.service = self.authenticate_gmail()

    def authenticate_gmail(self):
        """Authenticate Gmail API."""
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)

            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        return build('gmail', 'v1', credentials=creds)

    def fetch_last_email(self):
        """Fetch the latest email."""
        try:
            results = self.service.users().messages().list(userId='me', maxResults=1).execute()
            messages = results.get('messages', [])
            if not messages:
                return None, "No emails found."

            message = self.service.users().messages().get(userId='me', id=messages[0]['id']).execute()
            headers = message.get('payload', {}).get('headers', [])
            sender = next((header['value'] for header in headers if header['name'] == 'From'), "Unknown sender")
            snippet = message.get('snippet', '')
            return sender, snippet
        except Exception as e:
            print(f"Error fetching the last email: {e}")
            return None, "Error fetching the last email."

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
            "google": "https://www.google.com",
            "facebook": "https://www.facebook.com",
            "twitter": "https://www.twitter.com",
            "instagram": "https://www.instagram.com",
            "linkedin": "https://www.linkedin.com",
            "amazon": "https://www.amazon.com",
            "reddit": "https://www.reddit.com",
            "github": "https://www.github.com",
            "stackoverflow": "https://stackoverflow.com"
        }
        
        for site, url in websites.items():
            if site in command:
                webbrowser.open(url)
                return f"Opening {site.capitalize()}."
        return "I'm not sure which website to open."

class ConversationalAI:
    def __init__(self):
        self.speech_handler = SpeechHandler()
        self.huggingface_handler = HuggingFaceHandler()
        self.calendar_handler = CalendarHandler()
        self.web_handler = WebBrowserHandler()
        self.gmail_handler = GmailHandler()

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
                response = self.huggingface_handler.generate_response("Goodbye!")
                self.speech_handler.speak(response)
                break

            if "summarize my last email" in command:
                self.speech_handler.speak("Fetching your last email...")
                sender, last_email = self.gmail_handler.fetch_last_email()
                if last_email and "Error" not in last_email:
                    summary = self.huggingface_handler.summarize_email(last_email, sender)
                    self.speech_handler.speak(f"The summary of your last email from {sender} is: {summary}")
                else:
                    self.speech_handler.speak(last_email)
                continue

            # Handle other user commands dynamically
            response = self.huggingface_handler.generate_response(command)
            self.speech_handler.speak(response)

def main():
    try:
        ai_agent = ConversationalAI()
        ai_agent.run()
    except Exception as e:
        print(f"Error initializing the AI agent: {e}")

if __name__ == "__main__":
    main()
