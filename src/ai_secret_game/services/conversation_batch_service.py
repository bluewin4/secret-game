"""Conversation-level batch service for more efficient batch processing.

This implementation batches requests at the conversation level, sending all first messages,
then all second messages, etc., which is more efficient than completing entire conversations
sequentially.
"""

import os
import uuid
import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

from .batch_service import BatchService, BatchTask, BatchJob
from ..models.game import Game, GameMode
from ..models.agent import Agent
from ..services.agent_service import AgentService
from ..services.game_service import GameService
from ..utils.errors import BatchError

logger = logging.getLogger(__name__)


class ConversationBatchService(ABC):
    """Base class for services that implement conversation-level batching.
    
    Unlike the standard BatchService, which processes each conversation sequentially,
    this service processes messages at the same position across conversations in batches.
    This approach significantly improves throughput when running many parallel conversations.
    
    For example, with 100 conversations of 5 messages each, the standard approach would
    process 100 sequential conversations. The conversation-level approach processes:
    1. First message of all 100 conversations in a batch
    2. Second message of all 100 conversations in a batch
    And so on, requiring only 5 batches instead of 100 sequential conversations.
    """
    
    def __init__(
        self,
        game_service,
        output_dir: str = "results/batch_jobs",
        max_concurrent_tasks: int = 5,
        batch_size: int = 50
    ):
        """Initialize the conversation batch service.
        
        Args:
            game_service: GameService instance for running game interactions
            output_dir: Directory where results will be saved
            max_concurrent_tasks: Maximum number of tasks to process concurrently
            batch_size: Maximum number of requests to include in each batch
        """
        self.game_service = game_service
        self.output_dir = output_dir
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def run_batch_jobs(self, batch_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run multiple batch jobs using true conversation-level batching.
        
        This method takes multiple batch jobs and processes them together in waves,
        processing all first messages across all jobs, then all second messages, etc.
        
        Args:
            batch_jobs: List of batch job configurations
            
        Returns:
            List of job results
        """
        # Create asyncio event loop if not in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an event loop, we need to ensure the task is awaited
                # before returning results
                task = asyncio.create_task(self._run_batch_jobs(batch_jobs))
                # Since we can't use await directly here, we need to return something 
                # to indicate this is async. Let's fall back to sequential processing.
                logger.warning("Running in an existing event loop but can't await the task. "
                              "Consider using the async version of this method directly.")
                # Process jobs sequentially as a fallback
                results = []
                for job in batch_jobs:
                    result = self.run_batch_job(job)
                    results.append(result)
                return results
            else:
                # We're not in an event loop, use run_until_complete
                return loop.run_until_complete(self._run_batch_jobs(batch_jobs))
        except RuntimeError:
            # If there's no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._run_batch_jobs(batch_jobs))
            finally:
                # Clean up the loop when we're done with it
                loop.close()
    
    async def _run_batch_jobs(self, batch_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple batch jobs asynchronously using conversation-level batching.
        
        Args:
            batch_jobs: List of batch job configurations
            
        Returns:
            List of job results
        """
        logger.info(f"Processing {len(batch_jobs)} batch jobs together using conversation-level batching")
        
        # Initialize results for all batch jobs
        results = []
        for batch_job in batch_jobs:
            result = self._initialize_result(batch_job)
            results.append(result)
        
        # Initialize agent memories for all batch jobs
        agent_memories = {}
        for i, batch_job in enumerate(batch_jobs):
            batch_id = batch_job["id"]
            model1 = batch_job["model1"]
            model2 = batch_job["model2"]
            secret1 = batch_job["secret1"]
            secret2 = batch_job["secret2"]
            settings = batch_job["settings"]
            
            # Prepare agent memories for this batch job
            agent_memories[batch_id] = {
                "agent1": {
                    "id": f"{batch_id}_agent1",
                    "name": model1,
                    "secret": secret1,
                    "collected_secrets": [],
                    "conversation": []
                },
                "agent2": {
                    "id": f"{batch_id}_agent2",
                    "name": model2,
                    "secret": secret2,
                    "collected_secrets": [],
                    "conversation": []
                }
            }
        
        # Track the maximum number of messages across all jobs
        max_messages = max(job.get("messages_per_interaction", 5) for job in batch_jobs)
        
        # Process the batch jobs in conversation-level waves
        for message_index in range(max_messages):
            # Collect requests for this message index across all conversations
            requests = []
            
            for i, batch_job in enumerate(batch_jobs):
                batch_id = batch_job["id"]
                messages_per_interaction = batch_job["messages_per_interaction"]
                
                # Skip if this job has fewer messages than the current index
                if message_index >= messages_per_interaction:
                    continue
                
                # Determine which agent's turn it is
                agent_key = "agent1" if message_index % 2 == 0 else "agent2"
                other_agent_key = "agent2" if agent_key == "agent1" else "agent1"
                
                # Get agent memory
                agent_memory = agent_memories[batch_id][agent_key]
                other_agent_memory = agent_memories[batch_id][other_agent_key]
                
                # Create the context for this request
                context = {
                    "secret": agent_memory["secret"],
                    "rules": batch_job["settings"],
                    "collected_secrets": agent_memory["collected_secrets"],
                    "current_conversation": agent_memory["conversation"]
                }
                
                # Add the request to the list
                requests.append({
                    "task_id": batch_id,
                    "message_index": message_index,
                    "agent_id": agent_memory["id"],
                    "context": context
                })
            
            # Skip if no requests for this message index
            if not requests:
                continue
            
            agent_role = "Agent 1 (first messages)" if message_index % 2 == 0 else "Agent 2 (responses)"
            logger.info(f"Processing wave {message_index+1} with {len(requests)} requests for {agent_role}")
            
            # Log which conversations are being processed in this wave
            conversation_ids = [req["task_id"] for req in requests]
            logger.info(f"Wave {message_index+1} conversations: {', '.join(conversation_ids)}")
            
            # Process all requests for this message index in batches
            wave_start = time.time()
            responses = await self._process_request_wave(requests)
            wave_duration = time.time() - wave_start
            
            logger.info(f"Wave {message_index+1} completed in {wave_duration:.2f} seconds, avg {wave_duration/len(responses):.2f} sec per message")
            
            # Update agent memories with the responses
            for response in responses:
                batch_id = response["task_id"]
                message_index = response["message_index"]
                agent_id = response["agent_id"]
                message = response["message"]
                
                # Find the corresponding agent memory
                for agent_key in ["agent1", "agent2"]:
                    if agent_memories[batch_id][agent_key]["id"] == agent_id:
                        agent_memory = agent_memories[batch_id][agent_key]
                        other_agent_key = "agent2" if agent_key == "agent1" else "agent1"
                        other_agent_memory = agent_memories[batch_id][other_agent_key]
                        break
                
                # Add message to conversation
                role = "agent1" if agent_key == "agent1" else "agent2"
                agent_memory["conversation"].append({
                    "role": "user" if role == "agent2" else "assistant",
                    "content": message
                })
                
                # Add to other agent's conversation
                other_agent_memory["conversation"].append({
                    "role": "assistant" if role == "agent2" else "user",
                    "content": message
                })
                
                # Add message to the result
                for i, result in enumerate(results):
                    if result["id"] == batch_id:
                        result["conversation"].append({
                            "role": role,
                            "agent": agent_memory["name"],
                            "content": message,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Check for shared secrets
                        for secret_phrase in [agent_memory["secret"], other_agent_memory["secret"]]:
                            if secret_phrase.lower() in message.lower():
                                if secret_phrase == agent_memory["secret"]:
                                    # Agent revealed their own secret
                                    result["revealed_secrets"].append({
                                        "agent": agent_memory["name"],
                                        "secret": secret_phrase,
                                        "message_index": message_index
                                    })
                                else:
                                    # Agent obtained the other agent's secret
                                    if secret_phrase not in agent_memory["collected_secrets"]:
                                        agent_memory["collected_secrets"].append(secret_phrase)
                                        
                                    result["obtained_secrets"].append({
                                        "agent": agent_memory["name"],
                                        "secret": secret_phrase,
                                        "message_index": message_index
                                    })
                        break
            
            logger.info(f"Completed processing wave {message_index+1}")
        
        # Save results for each job
        for i, result in enumerate(results):
            batch_id = batch_jobs[i]["id"]
            result_path = os.path.join(self.output_dir, "results", f"{batch_id}.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Saved result to {result_path}")
            
            # Mark this result as completed
            result["status"] = "completed"
            result["timestamp"] = datetime.now().isoformat()
        
        return results
    
    async def _process_request_wave(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a wave of requests across multiple conversations.
        
        Args:
            requests: List of requests to process
            
        Returns:
            List of responses
        """
        # Process requests in chunks to manage memory and API rate limits
        chunk_size = min(self.batch_size, 50)  # Cap at 50 requests per batch to be safe
        responses = []
        
        for i in range(0, len(requests), chunk_size):
            chunk = requests[i:i+chunk_size]
            chunk_responses = await self._execute_batch_requests(chunk)
            responses.extend(chunk_responses)
        
        return responses
    
    async def _execute_batch_requests(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a batch of requests using the appropriate API.
        
        This method should be implemented by provider-specific subclasses.
        
        Args:
            batch_requests: List of prepared batch requests
            
        Returns:
            List of responses for each request
        """
        raise NotImplementedError("Subclasses must implement _execute_batch_requests")
    
    def _initialize_result(self, batch_job: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize a result object for a batch job.
        
        Args:
            batch_job: Configuration for the batch job
            
        Returns:
            Initialized result object
        """
        return {
            "id": batch_job["id"],
            "model1": batch_job["model1"],
            "model2": batch_job["model2"],
            "secret1": batch_job["secret1"],
            "secret2": batch_job["secret2"],
            "settings": batch_job["settings"],
            "conversation": [],
            "revealed_secrets": [],
            "obtained_secrets": [],
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }
    
    def run_batch_job(self, batch_job: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single batch job.
        
        This is a convenience method to run a single batch job. For true conversation-level
        batching with multiple jobs, use run_batch_jobs.
        
        Args:
            batch_job: Configuration for the batch job
            
        Returns:
            The job result
        """
        return self.run_batch_jobs([batch_job])[0]

    def _save_result(self, job_id: str, result: Dict[str, Any]) -> None:
        """Save the result of a batch job to a file.
        
        Args:
            job_id: ID of the batch job
            result: Result to save
        """
        # Create results directory if it doesn't exist
        results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the result
        result_file = os.path.join(results_dir, f"{job_id}.json")
        
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Saved result to {result_file}")

    async def run_batch_job(self, batch_id: str) -> BatchJob:
        """Run a batch job using conversation-level batching.
        
        Args:
            batch_id: ID of the batch job to run
            
        Returns:
            Updated BatchJob with results
            
        Raises:
            BatchError: If the batch job doesn't exist
        """
        if batch_id not in self.jobs:
            raise BatchError(f"Batch job {batch_id} not found")
        
        batch_job = self.jobs[batch_id]
        batch_job.status = "processing"
        
        logger.info(f"Starting batch job {batch_id} with {len(batch_job.tasks)} tasks using conversation-level batching")
        
        try:
            # Initialize all agent memories for all tasks
            for task in batch_job.tasks:
                interaction_id = task.id
                self._initialize_agent_memories(task.agent1, task.agent2, interaction_id)
            
            # Determine the maximum number of message exchanges
            max_messages = max(task.messages_per_round for task in batch_job.tasks)
            
            # Track completed tasks
            completed_tasks: Set[str] = set()
            
            # Process each message exchange level across all conversations
            for message_index in range(max_messages * 2):
                # Prepare batch requests for this message exchange level
                batch_requests = []
                
                for task in batch_job.tasks:
                    # Skip tasks that are already completed
                    if task.id in completed_tasks:
                        continue
                    
                    # Skip if we've exceeded the messages for this task
                    if message_index >= task.messages_per_round * 2:
                        completed_tasks.add(task.id)
                        task.result = self._build_interaction_result(task)
                        task.status = "completed"
                        continue
                    
                    # Determine which agent speaks at this exchange level
                    current_agent = task.agent1 if message_index % 2 == 0 else task.agent2
                    other_agent = task.agent2 if message_index % 2 == 0 else task.agent1
                    
                    # Get the agent context
                    context = current_agent.get_context(task.game.rules, task.id)
                    
                    # Add the current conversation history
                    context["current_conversation"] = self.game_service.agent_service._extract_conversation(
                        current_agent, other_agent, task.id
                    )
                    
                    # Add this request to the batch
                    batch_requests.append({
                        "task_id": task.id,
                        "message_index": message_index,
                        "agent_id": current_agent.id,
                        "context": context
                    })
                
                # Skip if there are no requests at this level
                if not batch_requests:
                    continue
                
                # Process batch requests in chunks (max 50 per batch)
                logger.info(f"Processing {len(batch_requests)} requests for message exchange level {message_index}")
                
                # Split into chunks of batch_size
                request_batches = [
                    batch_requests[i:i + self.batch_size] 
                    for i in range(0, len(batch_requests), self.batch_size)
                ]
                
                # Process each batch concurrently
                semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
                batch_tasks = []
                
                for i, request_batch in enumerate(request_batches):
                    batch_tasks.append(self._process_request_batch(
                        semaphore, request_batch, i, len(request_batches)
                    ))
                
                # Wait for all batches to complete
                batch_responses = await asyncio.gather(*batch_tasks)
                
                # Flatten responses
                all_responses = [response for batch in batch_responses for response in batch]
                
                # Process the responses and update agent memories
                for response in all_responses:
                    task_id = response["task_id"]
                    message_index = response["message_index"]
                    agent_id = response["agent_id"]
                    message = response["message"]
                    
                    # Find the task
                    task = next(t for t in batch_job.tasks if t.id == task_id)
                    
                    # Determine which agents are involved in this message
                    current_agent = task.agent1 if task.agent1.id == agent_id else task.agent2
                    other_agent = task.agent2 if task.agent1.id == agent_id else task.agent1
                    
                    # Update agent memories with this message
                    self._update_agent_memories(current_agent, other_agent, message, task_id)
                    
                    # Check if this is the last message for this task
                    if message_index == (task.messages_per_round * 2) - 1:
                        completed_tasks.add(task_id)
                        task.result = self._build_interaction_result(task)
                        task.status = "completed"
            
            # Mark any remaining tasks as failed
            for task in batch_job.tasks:
                if task.status != "completed":
                    task.status = "failed"
                    task.error = "Task processing incomplete"
            
            # Update batch job statistics
            batch_job.total_completed = sum(1 for task in batch_job.tasks if task.status == "completed")
            batch_job.total_failed = sum(1 for task in batch_job.tasks if task.status == "failed")
            
            # Update batch job status
            batch_job.status = "completed" if batch_job.total_failed == 0 else "partially_completed"
            batch_job.completed_at = datetime.now()
            
            # Save results
            results_path = self._save_batch_results(batch_job)
            batch_job.results_path = results_path
            
            logger.info(f"Completed batch job {batch_id}: "
                       f"{batch_job.total_completed} completed, "
                       f"{batch_job.total_failed} failed")
            
            return batch_job
            
        except Exception as e:
            batch_job.status = "failed"
            logger.error(f"Error processing batch job {batch_id}: {str(e)}")
            raise BatchError(f"Failed to process batch job: {str(e)}")
    
    async def _process_request_batch(
        self,
        semaphore: asyncio.Semaphore,
        batch_requests: List[Dict[str, Any]],
        batch_idx: int,
        total_batches: int
    ) -> List[Dict[str, Any]]:
        """Process a batch of requests using the batch API.
        
        Args:
            semaphore: Semaphore to limit concurrency
            batch_requests: List of requests in this batch
            batch_idx: Index of this batch
            total_batches: Total number of batches
            
        Returns:
            List of responses
        """
        async with semaphore:
            try:
                logger.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(batch_requests)} requests")
                return await self._execute_batch_requests(batch_requests)
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
                # Return error responses
                return [{
                    "task_id": request["task_id"],
                    "message_index": request["message_index"],
                    "agent_id": request["agent_id"],
                    "message": f"Error: {str(e)}",
                    "error": str(e)
                } for request in batch_requests] 