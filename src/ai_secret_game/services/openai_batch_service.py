"""OpenAI-specific implementation of the batch service.

This service uses OpenAI's batch API to efficiently process large numbers
of game interactions asynchronously.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import time

from .batch_service import BatchService, BatchTask
from ..utils.errors import BatchError

logger = logging.getLogger(__name__)


class OpenAIBatchService(BatchService):
    """OpenAI-specific implementation of the batch service."""
    
    def __init__(
        self, 
        game_service,
        api_key: Optional[str] = None,
        output_dir: str = "results/batch_jobs",
        max_concurrent_tasks: int = 5,
        batch_size: int = 20
    ):
        """Initialize the OpenAI batch service.
        
        Args:
            game_service: GameService instance for running game interactions
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise BatchError("OpenAI API key is required")
    
    async def _execute_batch_requests(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a batch of requests using OpenAI's batch API.
        
        Args:
            batch_requests: List of prepared batch requests
            
        Returns:
            List of responses for each request
        """
        try:
            import openai
        except ImportError:
            raise BatchError("The openai package is required to use OpenAIBatchService")
        
        logger.info(f"Processing batch request with {len(batch_requests)} items using OpenAI's batch API")
        
        # Create a temporary JSONL file with the batch requests
        batch_dir = os.path.join(self.output_dir, "temp")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Generate a unique filename for this batch
        batch_file = os.path.join(batch_dir, f"batch_{int(time.time())}.jsonl")
        
        # Write the requests to a JSONL file
        with open(batch_file, 'w') as f:
            for i, request in enumerate(batch_requests):
                # Transform our internal request format to OpenAI's batch format
                context = request["context"]
                
                # Create a system prompt from the context
                system_content = self._create_system_prompt(context)
                
                # Format the conversation history for the model
                messages = []
                
                # Add system message
                messages.append({
                    "role": "system", 
                    "content": system_content
                })
                
                # Add conversation history
                for msg in context.get("current_conversation", []):
                    role = "user" if msg.get("role") == "user" else "assistant"
                    messages.append({"role": role, "content": msg.get("content", "")})
                
                # If no messages or last message was from assistant, add a short prompt
                if not messages or messages[-1]["role"] == "assistant":
                    messages.append({"role": "user", "content": "Your turn in our conversation."})
                
                # Format for OpenAI batch API
                batch_item = {
                    "custom_id": f"{request['task_id']}_{request['message_index']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-3.5-turbo",  # Or use other model as needed
                        "messages": messages,
                        "max_tokens": 1000
                    }
                }
                
                f.write(json.dumps(batch_item) + '\n')
        
        # Create a client
        client = openai.OpenAI(api_key=self.api_key)
        
        try:
            # Upload the file for batch processing
            batch_file_obj = client.files.create(
                file=open(batch_file, "rb"),
                purpose="batch"
            )
            
            # Create the batch job
            batch_job = client.batches.create(
                input_file_id=batch_file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            logger.info(f"Created OpenAI batch job: {batch_job.id}")
            
            # Poll until the batch job is completed
            while True:
                batch_job_status = client.batches.retrieve(batch_job.id)
                logger.info(f"Batch job status: {batch_job_status.status}")
                
                if batch_job_status.status == "completed":
                    break
                elif batch_job_status.status in ["failed", "cancelled", "expired"]:
                    raise BatchError(f"Batch job {batch_job.id} ended with status: {batch_job_status.status}")
                
                # Wait before polling again
                await asyncio.sleep(30)  # Check every 30 seconds
            
            # Download the results
            result_file_id = batch_job_status.output_file_id
            result = client.files.content(result_file_id).content
            
            # Save the raw results
            raw_results_file = os.path.join(batch_dir, f"results_{int(time.time())}.jsonl")
            with open(raw_results_file, 'wb') as f:
                f.write(result)
            
            # Parse the results
            responses = []
            results_mapping = {}
            with open(raw_results_file, 'r') as f:
                for line in f:
                    result_obj = json.loads(line.strip())
                    custom_id = result_obj.get("custom_id")
                    if custom_id:
                        task_id, message_index = custom_id.split('_')
                        message = result_obj["response"]["body"]["choices"][0]["message"]["content"]
                        results_mapping[custom_id] = {
                            "task_id": task_id,
                            "message_index": int(message_index),
                            "message": message
                        }
            
            # Match results to requests
            for request in batch_requests:
                custom_id = f"{request['task_id']}_{request['message_index']}"
                result = results_mapping.get(custom_id)
                
                if result:
                    responses.append({
                        "task_id": request["task_id"],
                        "message_index": request["message_index"],
                        "agent_id": request["agent_id"],
                        "message": result["message"]
                    })
                else:
                    logger.error(f"No result found for request {custom_id}")
                    # Add an error response
                    responses.append({
                        "task_id": request["task_id"],
                        "message_index": request["message_index"],
                        "agent_id": request["agent_id"],
                        "message": "Error: No response received from batch processing",
                        "error": "Missing batch result"
                    })
            
            return responses
            
        except Exception as e:
            logger.error(f"Error executing OpenAI batch request: {str(e)}")
            # Fall back to sequential processing
            return await super()._execute_batch_requests(batch_requests)
        finally:
            # Clean up temporary files
            try:
                os.remove(batch_file)
            except:
                pass
    
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