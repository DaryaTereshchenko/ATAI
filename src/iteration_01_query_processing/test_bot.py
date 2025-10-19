import time
from main.sparql_handler import SPARQLHandler
from main.result_formatter import ResultFormatter
import logging
import os
import sys 
import json


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from speakeasypy import Chatroom, EventType, Speakeasy
from config import BOT_USERNAME, BOT_PASSWORD, SPEAKEASY_HOST, GRAPH_FILE_PATH


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAgent:
    """Test agent for the 1st intermediate evaluation - SPARQL query handling."""
    
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login
        self.speakeasy = Speakeasy(host=SPEAKEASY_HOST, username=username, password=password)
        self.speakeasy.login()

        # Initialize the SPARQL handler (loads graph from config)
        self.sparql_handler = SPARQLHandler(graph_file_path=GRAPH_FILE_PATH)
        

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
        validation = self.sparql_handler.validate_query(message)
        if validation['valid']:
            try:
                logger.info("Executing SPARQL query...")
                response = self.sparql_handler.execute_query(message)
                if response['success']:
                    answer = response['data']
                    logger.info(f"Query result: {answer}")
                    
                    # Post the answer directly (it's already a formatted string)
                    room.post_messages(answer)
                else:
                    logger.error(f"Error: {response['error']}")
                    room.post_messages(f"Query execution failed: {response['error']}")
            except Exception as e:
                error_msg = f"Error processing SPARQL query: {str(e)}"
                logger.error(error_msg)
                room.post_messages(error_msg)
        else:
            room.post_messages(f"Invalid SPARQL query: {validation['message']}\n\nPlease provide a valid SPARQL query. This bot only processes SPARQL queries for the 1st intermediate evaluation.")

    def on_new_reaction(self, reaction: str, message_ordinal: int, room: Chatroom): 
        """Callback function to handle new reactions."""
        logger.info(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")

if __name__ == '__main__':
    # Use your bot's credentials
    username = BOT_USERNAME
    password = BOT_PASSWORD
    test_bot = TestAgent(
        username=username,
        password=password
    )
    test_bot.listen()


