import time
from speakeasypy import Chatroom, EventType, Speakeasy

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'

class Agent:
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()

        # Register callbacks for different events
        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

    def listen(self):
        """Start listening for events."""
        self.speakeasy.start_listening()

    def on_new_message(self, message: str, room: Chatroom):
        """Callback function to handle new messages."""
        print(f"New message in room {room.room_id}: {message}")
        # Implement your bot logic here
        room.post_messages(f"You said: '{message}'. How can I assist further?")

    def on_new_reaction(self, reaction: str, message_ordinal: int, room: Chatroom): 
        """Callback function to handle new reactions."""
        print(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")
        room.post_messages(f"Thanks for your reaction: '{reaction}'")

if __name__ == '__main__':
    # Use your bot's credentials
    my_bot = Agent("BeigeCrackingEgg", "Vb2Xv8iY")
    my_bot.listen()