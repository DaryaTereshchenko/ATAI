import time
from speakeasypy import Chatroom, EventType, Speakeasy
from sparql_handler import SPARQLHandler
import config
import logging

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAgent:
    """Test agent for the 1st intermediate evaluation - SPARQL query handling."""
    
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()

        # Initialize the SPARQL handler (loads graph from config)
        self.sparql_handler = SPARQLHandler(graph_file_path="data/graph.nt")

        # Register callbacks for different events
        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

    def listen(self):
        """Start listening for events."""
        self.speakeasy.start_listening()

    def on_new_message(self, message: str, room: Chatroom):
        """Callback function to handle new messages - SPARQL queries only."""
        logger.info(f"New message in room {room.room_id}: {message[:100]}...")

        # Check if the message contains a SPARQL query
        if self.sparql_handler.is_sparql_query(message):
            try:
                response = self.sparql_handler.process_sparql_input(message)
                logger.info(f"Response: {response}")
                room.post_messages(response)
            except Exception as e:
                error_msg = f"Error processing SPARQL query: {str(e)}"
                logger.error(error_msg)
                room.post_messages(error_msg)
        else:
            room.post_messages("Please provide a valid SPARQL query. This bot only processes SPARQL queries for the 1st intermediate evaluation.")

    def on_new_reaction(self, reaction: str, message_ordinal: int, room: Chatroom): 
        """Callback function to handle new reactions."""
        logger.info(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")

if __name__ == '__main__':
    # Use your bot's credentials
    username = config.SPEAKEASY_USERNAME
    password = config.SPEAKEASY_PASSWORD
    test_bot = TestAgent(
        username=username,
        password=password
    )
    test_bot.listen()
    
