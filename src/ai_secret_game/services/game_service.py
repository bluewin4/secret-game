"""Game service for orchestrating the AI Secret Trading Game."""

import logging
import uuid
import random
from typing import List, Dict, Any, Tuple, Optional

from ..models.game import Game, GameMode
from ..models.agent import Agent
from ..utils.errors import GameError
from ..utils.logging import log_game_state, log_agent_interaction
from .scoring_service import ScoringService
from .agent_service import AgentService

logger = logging.getLogger(__name__)


class GameService:
    """Orchestrates the game flow.
    
    This service is responsible for creating games, running rounds,
    and managing agent interactions.
    """
    
    def __init__(self):
        """Initialize the game service."""
        self.scoring_service = ScoringService()
        self.agent_service = AgentService()
    
    def create_game(
        self, 
        agents: List[Agent], 
        mode: GameMode = GameMode.STANDARD,
        max_rounds: int = 10,
        messages_per_round: int = 3,
        targeted_secret: Optional[str] = None
    ) -> Game:
        """Create a new game with the specified parameters.
        
        Args:
            agents: List of agents participating in the game
            mode: Game mode determining scoring rules
            max_rounds: Maximum number of rounds in the game
            messages_per_round: Number of messages each agent can send per round
            targeted_secret: Secret that's worth extra points in targeted mode
            
        Returns:
            Newly created Game object
        """
        if len(agents) < 2:
            raise GameError("At least 2 agents are required to create a game")
            
        game_id = str(uuid.uuid4())
        game = Game(
            id=game_id,
            mode=mode,
            agents=agents,
            max_rounds=max_rounds,
            messages_per_round=messages_per_round,
            targeted_secret=targeted_secret
        )
        game.initialize_rules()
        
        logger.info(f"Created new game {game_id} with mode {mode.value}")
        log_game_state(game_id, {
            "mode": mode.value,
            "agent_count": len(agents),
            "max_rounds": max_rounds,
            "messages_per_round": messages_per_round
        })
        
        return game
    
    def run_round(self, game: Game) -> Dict[str, Any]:
        """Run a single round of the game.
        
        Args:
            game: The game for which to run a round
            
        Returns:
            Dictionary containing round results
            
        Raises:
            GameError: If the game has already reached maximum rounds
        """
        if game.round >= game.max_rounds:
            raise GameError("Game has already reached maximum rounds")
        
        # Generate pairings for this round
        pairings = self._generate_pairings(game)
        game.pairings_history.append(pairings)
        
        round_results = []
        
        # For each pair, run an interaction
        for agent1_id, agent2_id in pairings:
            agent1 = next(a for a in game.agents if a.id == agent1_id)
            agent2 = next(a for a in game.agents if a.id == agent2_id)
            
            interaction_result = self._run_interaction(
                game, agent1, agent2, game.messages_per_round
            )
            round_results.append(interaction_result)
            
            # Log the interaction
            log_agent_interaction(
                game.id, agent1.id, agent2.id, interaction_result
            )
            
            # Update scores based on interaction results
            self.scoring_service.update_scores(game, agent1, agent2, interaction_result)
        
        game.round += 1
        
        round_summary = {
            "round": game.round,
            "pairings": pairings,
            "results": round_results
        }
        
        log_game_state(game.id, {"round_complete": game.round})
        
        return round_summary
    
    def run_game(self, game: Game) -> Dict[str, Any]:
        """Run a complete game from start to finish.
        
        Args:
            game: The game to run
            
        Returns:
            Dictionary containing final game results
        """
        logger.info(f"Starting game {game.id} with {len(game.agents)} agents")
        
        round_results = []
        for _ in range(game.max_rounds):
            round_result = self.run_round(game)
            round_results.append(round_result)
        
        # Calculate final scores
        final_scores = self.scoring_service.calculate_final_scores(game)
        
        game_results = {
            "game_id": game.id,
            "mode": game.mode.value,
            "rounds": round_results,
            "final_scores": final_scores,
            "winner": self._determine_winner(final_scores)
        }
        
        logger.info(f"Game {game.id} completed. Winner: {game_results['winner']}")
        log_game_state(game.id, {"game_complete": True, "final_scores": final_scores})
        
        return game_results
    
    def _generate_pairings(self, game: Game) -> List[Tuple[str, str]]:
        """Generate pairings for the current round.
        
        Args:
            game: The current game
            
        Returns:
            List of agent ID pairs for this round
        """
        agent_ids = [agent.id for agent in game.agents]
        
        # Shuffle to ensure random pairings
        random.shuffle(agent_ids)
        
        pairings = []
        for i in range(0, len(agent_ids), 2):
            if i + 1 < len(agent_ids):
                pairings.append((agent_ids[i], agent_ids[i + 1]))
            # Handle odd number of agents
            else:
                logger.warning(f"Agent {agent_ids[i]} has no partner this round")
        
        return pairings
    
    def _run_interaction(
        self, game: Game, agent1: Agent, agent2: Agent, messages_per_round: int
    ) -> Dict[str, Any]:
        """Run an interaction between two agents.
        
        Args:
            game: The current game
            agent1: First agent in the interaction
            agent2: Second agent in the interaction
            messages_per_round: Number of messages each agent can send
            
        Returns:
            Dictionary containing interaction results
        """
        # Generate a unique ID for this interaction
        interaction_id = str(uuid.uuid4())
        
        # Initialize the interaction
        interaction = {
            "id": interaction_id,
            "messages": [],
            "agent1_revealed_secret": False,
            "agent2_revealed_secret": False
        }
        
        # Initialize with empty conversation history for this interaction
        # This ensures agents see the correct conversation when generating first message
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
        
        # Alternate messages between agents
        for i in range(messages_per_round * 2):
            current_agent = agent1 if i % 2 == 0 else agent2
            other_agent = agent2 if i % 2 == 0 else agent1
            
            # Get the agent's response
            message = self.agent_service.get_agent_message(
                game, current_agent, other_agent, interaction_id
            )
            
            # Record the message
            message_data = {
                "agent_id": current_agent.id,
                "message": message
            }
            interaction["messages"].append(message_data)
            
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
            
            # Check if secret was revealed
            if current_agent.secret in message:
                if current_agent == agent1:
                    interaction["agent1_revealed_secret"] = True
                else:
                    interaction["agent2_revealed_secret"] = True
        
        return interaction
    
    def _determine_winner(self, final_scores: Dict[str, int]) -> str:
        """Determine the winner of the game.
        
        Args:
            final_scores: Dictionary mapping agent IDs to final scores
            
        Returns:
            ID of the winning agent, or 'Tie' if there's a tie
        """
        if not final_scores:
            return "No participants"
            
        max_score = max(final_scores.values())
        winners = [agent_id for agent_id, score in final_scores.items() if score == max_score]
        
        if len(winners) > 1:
            return "Tie"
        return winners[0] 