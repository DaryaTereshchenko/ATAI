import time
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from speakeasypy import Chatroom, EventType, Speakeasy
from src.main.orchestrator import Orchestrator
import dotenv

dotenv.load_dotenv()
DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'

class Agent:
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()

        # Initialize the orchestrator
        self.orchestrator = Orchestrator()

        # Register callbacks for different events
        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

    def listen(self):
        """Start listening for events."""
        self.speakeasy.start_listening()

    def on_new_message(self, message: str, room: Chatroom):
        """Callback function to handle new messages."""
        print(f"\n{'='*80}")
        print(f"New message in room {room.room_id}")
        print(f"Message: {message}")
        print(f"{'='*80}\n")
        
        # Process the message through the orchestrator with workflow
        try:
            response = self.orchestrator.process_query(message)
            
            # Post the response
            room.post_messages(response)
            
            print(f"\n{'='*80}")
            print(f"Response sent to room {room.room_id}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            error_msg = f"❌ **Error**\n\nSorry, I encountered an error processing your request.\n\nDetails: {str(e)}"
            print(f"Error processing message: {e}")
            room.post_messages(error_msg)

    def on_new_reaction(self, reaction: str, message_ordinal: int, room: Chatroom): 
        """Callback function to handle new reactions."""
        print(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")
        room.post_messages(f"Thanks for your reaction: '{reaction}'")

if __name__ == '__main__':
    # Use your bot's credentials
    username = dotenv.get("SPEAKEASY_USERNAME")
    password = dotenv.get("SPEAKEASY_PASSWORD")
    my_bot = Agent(username, password)
    my_bot.listen()