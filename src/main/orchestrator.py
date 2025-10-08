from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from sparql_handler import SPARQLHandler
from src.main.config import LLM_MODEL, LLM_TEMPERATURE

class QuestionType(str, Enum):
    FACTUAL = "factual"
    EMBEDDING = "embedding"
    MULTIMEDIA = "multimedia"
    RECOMMENDATION = "recommendation"

class QueryClassification(BaseModel):
    """Classification of a user query into one of four types."""
    question_type: QuestionType = Field(
        description="The type of question: factual, embedding, multimedia, or recommendation"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )

class Orchestrator:
    """Routes user queries to appropriate processing nodes based on question type."""
    
    def __init__(self, llm: Optional[ChatOllama] = None):
        """Initialize the orchestrator with a language model."""
        self.llm = llm or ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        self.parser = PydanticOutputParser(pydantic_object=QueryClassification)
        self.sparql_handler = SPARQLHandler()
        self._setup_classifier()
    
    def _setup_classifier(self):
        """Set up the classification prompt and chain."""
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for a movie information system.
            Classify user queries into one of four types:

            1. FACTUAL: Questions about specific facts in the knowledge graph (directors, actors, release dates, etc.)
            Examples: "Who directed Good Will Hunting?", "Who is the director of Star Wars?"

            2. EMBEDDING: Questions that require similarity search or semantic understanding
            Examples: "Who is the screenwriter of...", "What is the MPAA rating of...", "What is the genre of..."

            3. MULTIMEDIA: Questions asking to show, display, or view images/media of people
            Examples: "Show me a picture of...", "What does X look like?", "Let me see...", "Display...", "Let me know what X looks like."

            4. RECOMMENDATION: Questions asking for movie suggestions or recommendations
            Examples: "Recommend movies like...", "Can you suggest movies similar to...", "Given that I like..."

            {format_instructions}"""),
            ("user", "{query}")
        ])
        
        self.classification_chain = classification_prompt | self.llm | self.parser
    
    def classify_query(self, query: str) -> QueryClassification:
        """Classify a user query into one of the four types."""
        result = self.classification_chain.invoke({
            "query": query,
            "format_instructions": self.parser.get_format_instructions()
        })
        return result
    
    def process_query(self, query: str) -> str:
        """Process a query by routing it to the appropriate node."""
        classification = self.classify_query(query)
        
        print(f"Query classified as: {classification.question_type.value}")
        print(f"Confidence: {classification.confidence}")
        print(f"Reasoning: {classification.reasoning}")
        
        # Route to appropriate node
        if classification.question_type == QuestionType.FACTUAL:
            return self._handle_factual(query)
        elif classification.question_type == QuestionType.EMBEDDING:
            return self._handle_embedding(query)
        elif classification.question_type == QuestionType.MULTIMEDIA:
            return self._handle_multimedia(query)
        elif classification.question_type == QuestionType.RECOMMENDATION:
            return self._handle_recommendation(query)
        else:
            return "I'm not sure how to handle that question."
    
    def _handle_factual(self, query: str) -> str:
        """Handle factual questions using knowledge graph."""
        # Check if it's a direct SPARQL query
        if self.sparql_handler.is_sparql_query(query):
            return self.sparql_handler.process_sparql_input(query)
        
        # TODO: Implement natural language to SPARQL conversion
        return f"[FACTUAL NODE - Natural language processing not yet implemented] Processing: {query}"
    
    def _handle_embedding(self, query: str) -> str:
        """Handle embedding-based questions (placeholder)."""
        # TODO: Implement embedding search logic
        return f"[EMBEDDING NODE - Not yet implemented] Processing: {query}"
    
    def _handle_multimedia(self, query: str) -> str:
        """Handle multimedia questions (placeholder)."""
        # TODO: Implement multimedia retrieval logic
        return f"[MULTIMEDIA NODE - Not yet implemented] Processing: {query}"
    
    def _handle_recommendation(self, query: str) -> str:
        """Handle recommendation questions (placeholder)."""
        # TODO: Implement recommendation logic
        return f"[RECOMMENDATION NODE - Not yet implemented] Processing: {query}"
