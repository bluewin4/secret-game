"""Anthropic-specific implementation of the batch service.

This service uses Anthropic's Message Batches API to efficiently process 
large numbers of game interactions asynchronously.
"""

import os
import json
import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

from .batch_service import BatchService
from ..utils.errors import BatchError

logger = logging.getLogger(__name__)


class AnthropicBatchService(BatchService):
    """Anthropic-specific implementation of the batch service using Claude Message Batches API."""
    
    def __init__(
        self, 
        game_service,
        api_key: Optional[str] = None,
        output_dir: str = "results/batch_jobs",
        max_concurrent_tasks: int = 5,
        batch_size: int = 50,  # Anthropic supports up to 10,000 per batch
        model: str = "claude-3-haiku-20240307"
    ):
        """Initialize the Anthropic batch service.
        
        Args:
            game_service: GameService instance for running game interactions
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            output_dir: Directory where results will be saved
            max_concurrent_tasks: Maximum number of tasks to process concurrently
            batch_size: Number of tasks to include in each API batch request
            model: Anthropic model to use for batch processing
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
        self.model = model
    
    async def _execute_batch_requests(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a batch of requests using Anthropic's Message Batches API.
        
        Args:
            batch_requests: List of prepared batch requests
            
        Returns:
            List of responses for each request
        """
        try:
            import anthropic
        except ImportError:
            raise BatchError("The anthropic package is required to use AnthropicBatchService")
        
        logger.info(f"Processing batch request with {len(batch_requests)} items using Anthropic's Batch API")
        
        # Create a client
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # Prepare the requests for Anthropic's batch API
        batch_requests_formatted = []
        for request in batch_requests:
            context = request["context"]
            
            # Format the conversation history for Claude
            messages = []
            
            # Add conversation history
            for msg in context.get("current_conversation", []):
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("content", "")})
            
            # If no messages or last message was from assistant, add a short prompt
            if not messages or messages[-1]["role"] == "assistant":
                messages.append({"role": "user", "content": "Your turn in our conversation."})
            
            # Create a system prompt from the context
            system = self._create_system_prompt(context)
            
            # Format for Anthropic batch API
            batch_requests_formatted.append({
                "custom_id": f"{request['task_id']}_{request['message_index']}",
                "params": {
                    "model": self.model,
                    "max_tokens": 1000,
                    "system": system,
                    "messages": messages
                }
            })
        
        # Create a batch directory
        batch_dir = os.path.join(self.output_dir, "temp")
        os.makedirs(batch_dir, exist_ok=True)
        
        try:
            # Create the batch job
            batch_job = client.beta.messages.batches.create(
                requests=batch_requests_formatted
            )
            
            batch_id = batch_job.id
            logger.info(f"Created Anthropic batch job: {batch_id}")
            
            # Poll until the batch job is completed
            while True:
                batch_job_status = client.beta.messages.batches.retrieve(batch_id)
                logger.info(f"Batch job status: {batch_job_status.processing_status}")
                
                if batch_job_status.processing_status == "ended":
                    break
                elif batch_job_status.processing_status == "errored":
                    raise BatchError(f"Batch job {batch_id} ended with status: {batch_job_status.processing_status}")
                
                # Wait before polling again
                await asyncio.sleep(30)  # Check every 30 seconds
            
            # Retrieve the results
            results = await self._download_batch_results(client, batch_job_status)
            
            # Save the raw results (for debugging)
            raw_results_file = os.path.join(batch_dir, f"results_{int(time.time())}.json")
            with open(raw_results_file, 'w') as f:
                json.dump(results, f)
            
            # Format the responses
            responses = []
            for request in batch_requests:
                custom_id = f"{request['task_id']}_{request['message_index']}"
                result = next((r for r in results if r.get("custom_id") == custom_id), None)
                
                if result and "message" in result:
                    # Extract the message content
                    message_content = result["message"]["content"][0]["text"]
                    
                    responses.append({
                        "task_id": request["task_id"],
                        "message_index": request["message_index"],
                        "agent_id": request["agent_id"],
                        "message": message_content
                    })
                else:
                    error_message = "Result not found or missing message" if result else "No result found"
                    logger.error(f"{error_message} for request {custom_id}")
                    
                    responses.append({
                        "task_id": request["task_id"],
                        "message_index": request["message_index"],
                        "agent_id": request["agent_id"],
                        "message": f"Error: {error_message}",
                        "error": error_message
                    })
            
            return responses
            
        except Exception as e:
            logger.error(f"Error executing Anthropic batch request: {str(e)}")
            # Fall back to sequential processing
            return await super()._execute_batch_requests(batch_requests)
    
    async def _download_batch_results(self, client, batch_job) -> List[Dict[str, Any]]:
        """Download and parse the batch results.
        
        Args:
            client: Anthropic client
            batch_job: Batch job object
            
        Returns:
            List of parsed results
        """
        results = []
        
        # Stream the results to avoid loading everything into memory
        async for chunk in client.beta.messages.batches.results(batch_job.id):
            # Each chunk is a line of JSON
            results.append(json.loads(chunk))
        
        return results
    
    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Create a system prompt based on game context for Claude.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the system prompt
        """
        prompt_sections = []
        
        # Introduction
        prompt_sections.append("You are an AI agent playing a secret trading game. Your goal is to discover the other agent's secret while strategically deciding whether to reveal your own.")
        
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