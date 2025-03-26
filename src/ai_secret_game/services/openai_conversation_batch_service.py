"""OpenAI-specific implementation of the conversation batch service.

This service uses OpenAI's batch API to efficiently process large numbers
of game interactions using conversation-level batching.
"""

import os
import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from .conversation_batch_service import ConversationBatchService
from ..utils.errors import BatchError

logger = logging.getLogger(__name__)


class OpenAIConversationBatchService(ConversationBatchService):
    """OpenAI-specific implementation of the conversation batch service."""
    
    def __init__(
        self, 
        game_service,
        api_key: Optional[str] = None,
        output_dir: str = "results/batch_jobs",
        max_concurrent_tasks: int = 5,
        batch_size: int = 50
    ):
        """Initialize the OpenAI conversation batch service.
        
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
            raise BatchError("The openai package is required to use OpenAIConversationBatchService")
        
        # Skip batch processing if there's just one request - go directly to sequential
        if len(batch_requests) == 1:
            logger.info("Only one request, skipping batch API and processing sequentially")
            return await self._execute_sequential_requests(batch_requests)
        
        logger.info(f"Processing batch request with {len(batch_requests)} items using OpenAI's API")
        
        try:
            # Create a temporary JSONL file with the batch requests
            batch_dir = os.path.join(self.output_dir, "temp")
            os.makedirs(batch_dir, exist_ok=True)
            
            # Generate a unique filename for this batch
            batch_file = os.path.join(batch_dir, f"batch_{int(time.time())}.jsonl")
            
            # Write the requests to a JSONL file
            with open(batch_file, 'w') as f:
                for request in batch_requests:
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
                    
                    # Add conversation history without adding a default message
                    current_conversation = context.get("current_conversation", [])
                    if current_conversation:
                        for msg in current_conversation:
                            role = "user" if msg.get("role") == "user" else "assistant"
                            messages.append({"role": role, "content": msg.get("content", "")})
                    else:
                        # If this is the first message, add a generic starter that's not biasing
                        messages.append({"role": "user", "content": "Hello."})
                    
                    # Determine the model to use based on the context or default to GPT-3.5
                    model_name = "gpt-3.5-turbo"
                    if hasattr(self.game_service.agent_service, "model_name"):
                        model_name = self.game_service.agent_service.model_name
                    
                    # Format for OpenAI batch API
                    batch_item = {
                        "custom_id": f"{request['task_id']}_{request['message_index']}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_name,
                            "messages": messages,
                            "max_tokens": 1000,
                            "temperature": 0.7
                        }
                    }
                    
                    f.write(json.dumps(batch_item) + '\n')
            
            # Create a client
            client = openai.OpenAI(api_key=self.api_key)
            
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
                    logger.error(f"Batch job {batch_job.id} ended with status: {batch_job_status.status}")
                    logger.info("Falling back to sequential processing")
                    return await self._execute_sequential_requests(batch_requests)
                
                # Wait before polling again
                await asyncio.sleep(5)  # Check every 5 seconds to speed up development (previously 30)
            
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
            logger.info("Falling back to sequential processing")
            return await self._execute_sequential_requests(batch_requests)
        finally:
            # Clean up temporary files
            try:
                if 'batch_file' in locals():
                    os.remove(batch_file)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary files: {cleanup_error}")
    
    async def _execute_sequential_requests(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process requests sequentially as a fallback.
        
        Args:
            batch_requests: List of prepared batch requests
            
        Returns:
            List of responses
        """
        import openai
        
        logger.info(f"Processing {len(batch_requests)} requests sequentially")
        client = openai.OpenAI(api_key=self.api_key)
        responses = []
        
        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        tasks = []
        
        # Create tasks for all requests
        for request in batch_requests:
            tasks.append(self._process_single_sequential_request(
                client, request, semaphore
            ))
        
        # Execute all tasks concurrently (with concurrency limit)
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process any exceptions
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
    
    async def _process_single_sequential_request(self, client, request: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process a single request sequentially with concurrency control.
        
        Args:
            client: OpenAI client
            request: Request to process
            semaphore: Semaphore for concurrency control
            
        Returns:
            Response for the request
        """
        async with semaphore:
            task_id = request["task_id"]
            message_index = request["message_index"]
            agent_id = request["agent_id"]
            context = request["context"]
            
            try:
                # Create system prompt
                system_content = self._create_system_prompt(context)
                
                # Format messages
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
                
                # Determine the model to use
                model_name = "gpt-3.5-turbo"
                if hasattr(self.game_service.agent_service, "model_name"):
                    model_name = self.game_service.agent_service.model_name
                
                # Call the API
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7
                )
                
                # Extract the message
                message = response.choices[0].message.content
                
                return {
                    "task_id": task_id,
                    "message_index": message_index,
                    "agent_id": agent_id,
                    "message": message
                }
            except Exception as e:
                logger.error(f"Error getting response for task {task_id}, message {message_index}: {str(e)}")
                # Add an error response
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