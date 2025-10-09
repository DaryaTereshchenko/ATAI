import logging
import json
from typing import Dict, List, Any, Optional
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultFormatter:
    """Formats SPARQL query results into human-friendly messages using llama.cpp."""
    
    def __init__(self, model_path: str, n_ctx: int = 512, n_threads: int = 4):
        """
        Initialize the result formatter with llama.cpp.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (default: 512, suitable for TinyLlama)
            n_threads: Number of CPU threads to use (default: 4)
        """
        self.model_path = model_path
        logger.info(f"Loading model from {model_path}...")
        
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=0,  # CPU only
                verbose=False
            )
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.llm = None
        
    def format_results(self, results: str, query: Optional[str] = None) -> str:
        """
        Format results into a human-friendly message.
        
        Args:
            results: String result from SPARQL query
            query: Optional original SPARQL query for context
            
        Returns:
            Human-friendly formatted string
        """
        if not results or results == "No answer found in the database.":
            logger.info("Query returned no results")
            return "No answer found in the database."
        
        logger.info(f"Formatting result: {results[:100]}...")
        
        # If model failed to load, use fallback
        if self.llm is None:
            logger.warning("Model not loaded, returning result as-is")
            return results
        
        # Prepare the prompt for the LLM
        prompt = self._build_prompt(results, query)
        logger.info(f"Prompt for LLM:\n{prompt}")
        
        try:
            # Generate response using llama.cpp
            response = self._generate_response(prompt)
            logger.info("Response from LLM: {}".format(response))
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback to returning the result as-is
            return results
    
    def _build_prompt(self, results: str, query: Optional[str] = None) -> str:
        """Build the prompt for the LLM using TinyLlama chat format."""
        
        prompt = f"""<|system|>
You are a helpful chatbot assistant that presents database query results in a clear, conversational, and user-friendly format.</s>
<|user|>
Present this database query result as a natural chatbot response to the user:

Query Result:
{results}

Guidelines:
- Use conversational language like "The query returned:", "Based on the database:", or "Here's what I found:"
- Be concise and friendly (maximum 2 sentences)
- Keep all factual information from the result unchanged
- Make it sound natural for a chatbot conversation
- Do not mention that you are an AI or language model</s>
<|assistant|>
"""
        return prompt
    
    def _generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Generate a response using llama.cpp with TinyLlama-optimized parameters.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate (reduced for TinyLlama)
            
        Returns:
            Generated text response
        """
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            echo=False,
            stop=["</s>", "<|user|>", "<|system|>"]
        )
        
        response_text = output['choices'][0]['text'].strip()
        return response_text
    
    def _fallback_format(self, results: str) -> str:
        """
        Fallback formatting method if model is unavailable.
        
        Args:
            results: String result
            
        Returns:
            Simple formatted string
        """
        return f"Query executed successfully!\n\nResult: {results}"
    
    def _fallback_format(self, results: List[Dict[str, Any]]) -> str:
        """
        Fallback formatting method if model is unavailable.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Simple formatted string
        """
        formatted_lines = ["Query executed successfully!\n\nResults:"]
        
        for result in results:
            values = list(result.values())
            if len(values) == 1:
                formatted_lines.append(f"- {values[0]}")
            else:
                formatted_lines.append(f"- {', '.join(str(v) for v in values)}")
        
        return '\n'.join(formatted_lines)
