import time
from speakeasypy import Chatroom, EventType, Speakeasy
from orchestrator import Orchestrator
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
        print(f"New message in room {room.room_id}: {message}")
        
        # Process the message through the orchestrator
        try:
            response = self.orchestrator.process_query(message)
            room.post_messages(response)
        except Exception as e:
            print(f"Error processing message: {e}")
            room.post_messages("Sorry, I encountered an error processing your request.")

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