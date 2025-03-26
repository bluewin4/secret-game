"""Anthropic-specific implementation of the conversation batch service.

This service uses concurrent requests to process large numbers
of game interactions using conversation-level batching.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from .conversation_batch_service import ConversationBatchService
from ..utils.errors import BatchError

logger = logging.getLogger(__name__)


class AnthropicConversationBatchService(ConversationBatchService):
    """Anthropic-specific implementation of the conversation batch service."""
    
    def __init__(
        self, 
        game_service,
        api_key: Optional[str] = None,
        output_dir: str = "results/batch_jobs",
        max_concurrent_tasks: int = 5,
        batch_size: int = 50
    ):
        """Initialize the Anthropic conversation batch service.
        
        Args:
            game_service: GameService instance for running game interactions
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            output_dir: Directory where results will be saved
            max_concurrent_tasks: Maximum number of tasks to process concurrently
            batch_size: Number of tasks to include in each API batch request
        """
        super().__init__(
            game_service=game_service,
            output_dir=output_dir,
            max_concurrent_tasks=max_concurrent_tasks,
            batch_size=batch_size
        )
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise BatchError("Anthropic API key is required")
    
    async def _execute_batch_requests(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a batch of requests concurrently using Anthropic's API.
        
        Args:
            batch_requests: List of prepared batch requests
            
        Returns:
            List of responses for each request
        """
        try:
            import anthropic
        except ImportError:
            raise BatchError("The anthropic package is required to use AnthropicConversationBatchService")
        
        logger.info(f"Processing batch request with {len(batch_requests)} items using Anthropic's API")
        
        try:
            # Create a semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
            
            # Process each request concurrently
            tasks = []
            for request in batch_requests:
                task = asyncio.create_task(
                    self._process_single_request(request, semaphore)
                )
                tasks.append(task)
            
            # Wait for all requests to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses, handling any exceptions
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Error processing request {batch_requests[i]['task_id']}: {str(response)}")
                    processed_responses.append({
                        "task_id": batch_requests[i]["task_id"],
                        "message_index": batch_requests[i]["message_index"],
                        "agent_id": batch_requests[i]["agent_id"],
                        "message": f"Error: {str(response)}",
                        "error": str(response)
                    })
                else:
                    processed_responses.append(response)
            
            return processed_responses
            
        except Exception as e:
            logger.error(f"Error executing Anthropic batch request: {str(e)}")
            logger.info("Error occurred during batch processing")
            
            # Return error responses for all requests
            error_responses = []
            for request in batch_requests:
                error_responses.append({
                    "task_id": request["task_id"],
                    "message_index": request["message_index"],
                    "agent_id": request["agent_id"],
                    "message": f"Error: {str(e)}",
                    "error": str(e)
                })
            
            return error_responses
    
    async def _process_single_request(self, request: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process a single request using Anthropic's API.
        
        Args:
            request: The request to process
            semaphore: Semaphore to limit concurrent requests
            
        Returns:
            The response for the request
        """
        task_id = request["task_id"]
        message_index = request["message_index"]
        agent_id = request["agent_id"]
        context = request["context"]
        
        try:
            import anthropic
            
            async with semaphore:
                # Create a client
                client = anthropic.Anthropic(api_key=self.api_key)
                
                # Create system prompt
                system_content = self._create_system_prompt(context)
                
                # Format messages for Anthropic
                messages = [{"role": "system", "content": system_content}]
                
                # Add conversation history
                current_conversation = context.get("current_conversation", [])
                if current_conversation:
                    for msg in current_conversation:
                        role = "user" if msg.get("role") == "user" else "assistant"
                        messages.append({"role": role, "content": msg.get("content", "")})
                else:
                    # If this is the first message, add a generic starter that's not biasing
                    messages.append({"role": "user", "content": "Hello."})
                
                # Determine the model to use based on the context or default to Claude 3 Sonnet
                model_name = "claude-3-sonnet-20240229"
                if hasattr(self.game_service.agent_service, "model_name"):
                    model_name = self.game_service.agent_service.model_name
                
                # Log the request
                logger.debug(f"Sending request to Anthropic API for task {task_id}, message {message_index}")
                
                # Call the API with retry logic
                max_retries = 3
                retry_delay = 2  # seconds
                
                for retry in range(max_retries):
                    try:
                        # Call the API
                        response = client.messages.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=1000,
                            temperature=0.7
                        )
                        
                        # Extract the message
                        message = response.content[0].text
                        
                        logger.debug(f"Received response for task {task_id}, message {message_index}")
                        
                        return {
                            "task_id": task_id,
                            "message_index": message_index,
                            "agent_id": agent_id,
                            "message": message
                        }
                    except Exception as e:
                        if retry < max_retries - 1:
                            logger.warning(f"Retry {retry+1}/{max_retries} for task {task_id}: {str(e)}")
                            await asyncio.sleep(retry_delay * (retry + 1))  # Exponential backoff
                        else:
                            # Last retry failed, propagate the error
                            raise
                
                # This should not be reached due to the raise in the loop
                return {
                    "task_id": task_id,
                    "message_index": message_index,
                    "agent_id": agent_id,
                    "message": "Error: Maximum retries exceeded",
                    "error": "Maximum retries exceeded"
                }
                
        except Exception as e:
            logger.error(f"Error getting response for task {task_id}, message {message_index}: {str(e)}")
            return {
                "task_id": task_id,
                "message_index": message_index,
                "agent_id": agent_id,
                "message": f"Error: {str(e)}",
                "error": str(e)
            }
    
    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Create a system prompt based on game context.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the system prompt
        """
        prompt_sections = []
        
        # Introduction - always included
        prompt_sections.append(f"You are an AI agent playing a secret trading game.")
        
        # Add secret if included in context
        if "secret" in context:
            prompt_sections.append(f"Your secret is: \"{context['secret']}\"")
        
        # Add rules if included in context
        if "rules" in context:
            rules = context["rules"]
            mode = rules.get("mode", "standard")
            
            rules_section = [
                "Game rules:",
                "- You are chatting with another AI agent who also has a secret",
                "- Your goal is to learn their secret while deciding whether to reveal your own"
            ]
            
            # Add mode-specific scoring information
            if mode == "standard":
                rules_section.append(
                    "- Scoring: 0 points if no secrets revealed, 1 point each if both revealed, "
                    "3 points if you get their secret without revealing yours"
                )
            elif mode == "retained":
                rules_section.append(
                    "- Secrets are retained when shared, rarer secrets are worth more points"
                )
                rules_section.append(
                    "- Final scores are based on the rarity of secrets you collect"
                )
            elif mode == "diversity":
                rules_section.append("- Each unique secret you collect gives you points")
                rules_section.append(
                    f"- Points per unique secret: {rules['scoring'].get('points_per_unique_secret', 2)}"
                )
            elif mode == "targeted":
                rules_section.append("- One specific secret is worth extra points")
                rules_section.append(
                    f"- Regular secrets worth {rules['scoring'].get('standard_secret_points', 1)} point(s)"
                )
                rules_section.append(
                    f"- The targeted secret is worth {rules['scoring'].get('targeted_secret_points', 5)} points"
                )
            
            prompt_sections.append("\n".join(rules_section))
        
        # Add collected secrets if included in context
        if "collected_secrets" in context and context["collected_secrets"]:
            prompt_sections.append(
                f"Secrets you've already collected: {', '.join(context['collected_secrets'])}"
            )
        
        # Join all sections with double newlines
        return "\n\n".join(prompt_sections) 