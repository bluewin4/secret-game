"""Batch service for processing multiple game interactions asynchronously.

This service enables batch processing of game interactions to reduce API costs
and increase throughput when collecting statistical data about agent performance.
"""

import os
import uuid
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..models.game import Game, GameMode
from ..models.agent import Agent
from ..services.agent_service import AgentService
from ..services.game_service import GameService
from ..utils.errors import BatchError

logger = logging.getLogger(__name__)


@dataclass
class BatchTask:
    """Represents a single task within a batch processing job.
    
    Attributes:
        id: Unique identifier for this task
        game: Game instance for this task
        agent1: First agent in the interaction
        agent2: Second agent in the interaction
        messages_per_round: Number of messages each agent can send
        status: Current status of this task
        result: Result of the task once completed
        error: Error message if the task failed
    """
    id: str
    game: Game
    agent1: Agent
    agent2: Agent
    messages_per_round: int
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class BatchJob:
    """Represents a batch processing job containing multiple tasks.
    
    Attributes:
        id: Unique identifier for this batch job
        tasks: List of tasks in this batch
        status: Current status of the batch job
        created_at: When the batch job was created
        completed_at: When the batch job was completed
        results_path: Path to the saved results
        total_completed: Number of completed tasks
        total_failed: Number of failed tasks
    """
    id: str
    tasks: List[BatchTask] = field(default_factory=list)
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results_path: Optional[str] = None
    total_completed: int = 0
    total_failed: int = 0


class BatchService:
    """Service for handling batch processing of game interactions.
    
    This service enables efficient processing of multiple game interactions
    using batching capabilities of AI service providers.
    """
    
    def __init__(
        self, 
        game_service: GameService,
        output_dir: str = "results/batch_jobs",
        max_concurrent_tasks: int = 5,
        batch_size: int = 20
    ):
        """Initialize the batch service.
        
        Args:
            game_service: GameService instance for running game interactions
            output_dir: Directory where results will be saved
            max_concurrent_tasks: Maximum number of tasks to process concurrently
            batch_size: Number of tasks to include in each API batch request
        """
        self.game_service = game_service
        self.output_dir = output_dir
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        self.jobs: Dict[str, BatchJob] = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def create_batch_job(
        self, 
        agents: List[Agent],
        game_mode: GameMode = GameMode.STANDARD,
        num_interactions: int = 100,
        messages_per_interaction: int = 5,
        max_rounds: int = 1
    ) -> BatchJob:
        """Create a new batch job for processing multiple game interactions.
        
        Args:
            agents: List of agents to participate in the games
            game_mode: Game mode to use
            num_interactions: Number of interactions to process
            messages_per_interaction: Number of messages per interaction
            max_rounds: Maximum number of rounds to play
            
        Returns:
            Newly created BatchJob
        """
        batch_id = str(uuid.uuid4())
        batch_job = BatchJob(id=batch_id)
        
        # Create a game with the specified parameters
        game = self.game_service.create_game(
            agents=agents,
            mode=game_mode,
            max_rounds=max_rounds,
            messages_per_round=messages_per_interaction
        )
        
        # Create tasks for each interaction
        for _ in range(num_interactions):
            # For each interaction, randomly select two agents
            pairings = self.game_service._generate_pairings(game)
            
            for agent1_id, agent2_id in pairings:
                agent1 = next(a for a in game.agents if a.id == agent1_id)
                agent2 = next(a for a in game.agents if a.id == agent2_id)
                
                task = BatchTask(
                    id=str(uuid.uuid4()),
                    game=game,
                    agent1=agent1,
                    agent2=agent2,
                    messages_per_round=messages_per_interaction
                )
                batch_job.tasks.append(task)
        
        self.jobs[batch_id] = batch_job
        logger.info(f"Created batch job {batch_id} with {len(batch_job.tasks)} tasks")
        
        return batch_job
    
    # Additional methods for batch processing will be added in subsequent implementations
    
    async def run_batch_job(self, batch_id: str) -> BatchJob:
        """Run a batch job asynchronously.
        
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
        
        logger.info(f"Starting batch job {batch_id} with {len(batch_job.tasks)} tasks")
        
        # Process tasks in batches to avoid overwhelming the system
        pending_tasks = batch_job.tasks.copy()
        
        try:
            # Process tasks in chunks using a semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
            
            # Create a list to hold all the task coroutines
            task_coroutines = []
            
            # Group tasks into batches for more efficient API calls
            task_batches = [pending_tasks[i:i + self.batch_size] 
                           for i in range(0, len(pending_tasks), self.batch_size)]
            
            # For each batch, create a coroutine to process it
            for batch_idx, task_batch in enumerate(task_batches):
                task_coroutines.append(
                    self._process_task_batch(semaphore, task_batch, batch_idx, len(task_batches))
                )
            
            # Wait for all tasks to complete
            await asyncio.gather(*task_coroutines)
            
            # Update batch job status
            batch_job.status = "completed"
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
    
    async def _process_task_batch(
        self, 
        semaphore: asyncio.Semaphore,
        tasks: List[BatchTask],
        batch_idx: int,
        total_batches: int
    ) -> None:
        """Process a batch of tasks using the appropriate batch API.
        
        Args:
            semaphore: Semaphore to limit concurrency
            tasks: List of tasks to process in this batch
            batch_idx: Index of this batch
            total_batches: Total number of batches
        """
        async with semaphore:
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(tasks)} tasks")
            
            try:
                # Prepare the batch requests
                batch_requests = []
                for task in tasks:
                    # Mark task as processing
                    task.status = "processing"
                    
                    # Generate a unique interaction ID for this task
                    interaction_id = task.id
                    
                    # Initialize the agent memories for this interaction
                    self._initialize_agent_memories(task.agent1, task.agent2, interaction_id)
                    
                    # Prepare the conversation flow - alternating messages between agents
                    for i in range(task.messages_per_round * 2):
                        current_agent = task.agent1 if i % 2 == 0 else task.agent2
                        other_agent = task.agent2 if i % 2 == 0 else task.agent1
                        
                        # Get the agent context
                        context = current_agent.get_context(task.game.rules, interaction_id)
                        
                        # Add the current conversation
                        context["current_conversation"] = self.game_service.agent_service._extract_conversation(
                            current_agent, other_agent, interaction_id
                        )
                        
                        # Add this request to the batch
                        batch_requests.append({
                            "task_id": task.id,
                            "message_index": i,
                            "agent_id": current_agent.id,
                            "context": context
                        })
                
                # Process the batch requests - this will be implemented differently for different AI providers
                batch_responses = await self._execute_batch_requests(batch_requests)
                
                # Process the responses and update agent memories
                for response in batch_responses:
                    task_id = response["task_id"]
                    message_index = response["message_index"]
                    message = response["message"]
                    
                    # Find the task
                    task = next(t for t in tasks if t.id == task_id)
                    
                    # Determine which agents are involved in this message
                    current_agent = task.agent1 if message_index % 2 == 0 else task.agent2
                    other_agent = task.agent2 if message_index % 2 == 0 else task.agent1
                    
                    # Update agent memories
                    self._update_agent_memories(
                        current_agent, other_agent, message, task.id
                    )
                    
                    # Check if this is the last message in the interaction
                    if message_index == (task.messages_per_round * 2) - 1:
                        # Complete the task and gather results
                        task.result = self._build_interaction_result(task)
                        task.status = "completed"
                
                # Mark failed tasks
                for task in tasks:
                    if task.status == "processing":
                        task.status = "failed"
                        task.error = "Task processing incomplete"
                
                # Update batch job statistics
                self._update_batch_stats(tasks)
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                # Mark all tasks in this batch as failed
                for task in tasks:
                    if task.status != "completed":
                        task.status = "failed"
                        task.error = f"Batch processing error: {str(e)}"
                
                # Update batch job statistics
                self._update_batch_stats(tasks)
    
    async def _execute_batch_requests(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a batch of requests using the appropriate AI provider's batch API.
        
        This is a placeholder that should be overridden by provider-specific implementations.
        
        Args:
            batch_requests: List of prepared batch requests
            
        Returns:
            List of responses for each request
        """
        # This is a placeholder implementation that processes requests sequentially
        # In a real implementation, this would use the batch API of the AI provider
        responses = []
        
        for request in batch_requests:
            task_id = request["task_id"]
            message_index = request["message_index"]
            agent_id = request["agent_id"]
            context = request["context"]
            
            try:
                # Call the AI service to get a response - using the game service's agent service
                message = self.game_service.agent_service._call_ai_service(context)
                
                responses.append({
                    "task_id": task_id,
                    "message_index": message_index,
                    "agent_id": agent_id,
                    "message": message
                })
            except Exception as e:
                logger.error(f"Error getting response for task {task_id}, message {message_index}: {str(e)}")
                # Add an error response
                responses.append({
                    "task_id": task_id,
                    "message_index": message_index,
                    "agent_id": agent_id,
                    "message": f"Error: {str(e)}",
                    "error": str(e)
                })
        
        return responses
    
    def get_batch_job(self, batch_id: str) -> BatchJob:
        """Get a batch job by ID.
        
        Args:
            batch_id: ID of the batch job to get
            
        Returns:
            BatchJob if found
            
        Raises:
            BatchError: If the batch job doesn't exist
        """
        if batch_id not in self.jobs:
            raise BatchError(f"Batch job {batch_id} not found")
        
        return self.jobs[batch_id]
    
    def _initialize_agent_memories(
        self, agent1: Agent, agent2: Agent, interaction_id: str
    ) -> None:
        """Initialize agent memories for a new interaction.
        
        Args:
            agent1: First agent in the interaction
            agent2: Second agent in the interaction
            interaction_id: Unique ID for this interaction
        """
        # Initialize with empty conversation history for this interaction
        agent1.add_to_memory({
            "role": "system",
            "content": f"Beginning conversation with {agent2.name}",
            "from_agent_id": "system",
            "from_agent_name": "System",
            "to_agent_id": agent1.id,
            "to_agent_name": agent1.name,
            "interaction_id": interaction_id
        })
        
        agent2.add_to_memory({
            "role": "system",
            "content": f"Beginning conversation with {agent1.name}",
            "from_agent_id": "system",
            "from_agent_name": "System",
            "to_agent_id": agent2.id,
            "to_agent_name": agent2.name,
            "interaction_id": interaction_id
        })
    
    def _update_agent_memories(
        self, current_agent: Agent, other_agent: Agent, message: str, interaction_id: str
    ) -> None:
        """Update agent memories with a new message.
        
        Args:
            current_agent: Agent sending the message
            other_agent: Agent receiving the message
            message: Content of the message
            interaction_id: Unique ID for this interaction
        """
        # Update agent memory
        current_agent.add_to_memory({
            "role": "assistant",
            "content": message,
            "from_agent_id": current_agent.id,
            "from_agent_name": current_agent.name,
            "to_agent_id": other_agent.id,
            "to_agent_name": other_agent.name,
            "interaction_id": interaction_id
        })
        other_agent.add_to_memory({
            "role": "user",
            "content": message,
            "from_agent_id": current_agent.id,
            "from_agent_name": current_agent.name,
            "to_agent_id": other_agent.id,
            "to_agent_name": other_agent.name,
            "interaction_id": interaction_id
        })
    
    def _build_interaction_result(self, task: BatchTask) -> Dict[str, Any]:
        """Build the result of an interaction.
        
        Args:
            task: The batch task
            
        Returns:
            Dictionary containing interaction results
        """
        # Extract conversation history for this interaction
        conversation = []
        for msg in task.agent1.conversation_memory:
            if msg.get("interaction_id") == task.id:
                if msg.get("role") != "system":  # Skip system messages
                    conversation.append({
                        "agent_id": msg.get("from_agent_id"),
                        "agent_name": msg.get("from_agent_name"),
                        "message": msg.get("content")
                    })
        
        # Check if secrets were revealed
        agent1_revealed_secret = any(
            task.agent1.secret in msg.get("content", "")
            for msg in task.agent1.conversation_memory
            if msg.get("interaction_id") == task.id and msg.get("from_agent_id") == task.agent1.id
        )
        
        agent2_revealed_secret = any(
            task.agent2.secret in msg.get("content", "")
            for msg in task.agent1.conversation_memory
            if msg.get("interaction_id") == task.id and msg.get("from_agent_id") == task.agent2.id
        )
        
        # Build result
        return {
            "interaction_id": task.id,
            "agent1_id": task.agent1.id,
            "agent1_name": task.agent1.name,
            "agent2_id": task.agent2.id,
            "agent2_name": task.agent2.name,
            "messages": conversation,
            "agent1_revealed_secret": agent1_revealed_secret,
            "agent2_revealed_secret": agent2_revealed_secret
        }
    
    def _update_batch_stats(self, tasks: List[BatchTask]) -> None:
        """Update batch job statistics based on task status.
        
        Args:
            tasks: List of tasks to update statistics for
        """
        # Find the batch job for these tasks
        batch_job = next(
            (job for job in self.jobs.values() if any(t.id == task.id for task in tasks for t in job.tasks)),
            None
        )
        
        if batch_job:
            # Count completed and failed tasks
            batch_job.total_completed = sum(1 for t in batch_job.tasks if t.status == "completed")
            batch_job.total_failed = sum(1 for t in batch_job.tasks if t.status == "failed")
    
    def _save_batch_results(self, batch_job: BatchJob) -> str:
        """Save batch job results to a file.
        
        Args:
            batch_job: Batch job to save results for
            
        Returns:
            Path to the saved results file
        """
        # Create a results directory for this batch job
        results_dir = os.path.join(self.output_dir, batch_job.id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare results data
        results_data = {
            "batch_id": batch_job.id,
            "created_at": batch_job.created_at.isoformat(),
            "completed_at": batch_job.completed_at.isoformat() if batch_job.completed_at else None,
            "status": batch_job.status,
            "total_tasks": len(batch_job.tasks),
            "completed_tasks": batch_job.total_completed,
            "failed_tasks": batch_job.total_failed,
            "interactions": [
                task.result for task in batch_job.tasks 
                if task.status == "completed" and task.result is not None
            ]
        }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Saved batch results to {results_file}")
        
        return results_file 