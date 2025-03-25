"""Scoring service for the AI Secret Trading Game."""

from typing import Dict, Any, List
import logging

from ..models.game import Game, GameMode
from ..models.agent import Agent
from ..utils.errors import GameError

logger = logging.getLogger(__name__)


class ScoringService:
    """Handles scoring logic for the game.
    
    This service is responsible for calculating and updating scores
    based on the game mode and interaction outcomes.
    """
    
    def update_scores(
        self, game: Game, agent1: Agent, agent2: Agent, interaction_result: Dict[str, Any]
    ) -> None:
        """Update scores based on interaction results and game mode.
        
        Args:
            game: The current game
            agent1: First agent in the interaction
            agent2: Second agent in the interaction
            interaction_result: Dictionary containing interaction results
        """
        if game.mode == GameMode.STANDARD:
            self._standard_scoring(game, agent1, agent2, interaction_result)
        elif game.mode == GameMode.RETAINED:
            self._retained_scoring(game, agent1, agent2, interaction_result)
        elif game.mode == GameMode.DIVERSITY:
            self._diversity_scoring(game, agent1, agent2, interaction_result)
        elif game.mode == GameMode.TARGETED:
            self._targeted_scoring(game, agent1, agent2, interaction_result)
        else:
            raise GameError(f"Unknown game mode: {game.mode}")
        
        logger.info(f"Updated scores: {agent1.name}={agent1.score}, {agent2.name}={agent2.score}")
    
    def _standard_scoring(
        self, game: Game, agent1: Agent, agent2: Agent, interaction_result: Dict[str, Any]
    ) -> None:
        """Standard scoring mode.
        
        Args:
            game: The current game
            agent1: First agent in the interaction
            agent2: Second agent in the interaction
            interaction_result: Dictionary containing interaction results
        """
        agent1_revealed = interaction_result["agent1_revealed_secret"]
        agent2_revealed = interaction_result["agent2_revealed_secret"]
        
        if not agent1_revealed and not agent2_revealed:
            # No secrets revealed
            pass  # 0 points
        elif agent1_revealed and agent2_revealed:
            # Both secrets revealed
            agent1.score += 1
            agent2.score += 1
        elif agent1_revealed and not agent2_revealed:
            # Only agent1 revealed their secret
            agent2.score += 3
        elif not agent1_revealed and agent2_revealed:
            # Only agent2 revealed their secret
            agent1.score += 3
    
    def _retained_scoring(
        self, game: Game, agent1: Agent, agent2: Agent, interaction_result: Dict[str, Any]
    ) -> None:
        """Retained secrets scoring mode based on rarity.
        
        Args:
            game: The current game
            agent1: First agent in the interaction
            agent2: Second agent in the interaction
            interaction_result: Dictionary containing interaction results
        """
        agent1_revealed = interaction_result["agent1_revealed_secret"]
        agent2_revealed = interaction_result["agent2_revealed_secret"]
        
        # Add secrets to collections if revealed
        if agent1_revealed:
            if agent1.secret not in agent2.collected_secrets:
                agent2.collected_secrets.append(agent1.secret)
        
        if agent2_revealed:
            if agent2.secret not in agent1.collected_secrets:
                agent1.collected_secrets.append(agent2.secret)
        
        # In retained mode, we don't update scores immediately
        # Final scores will be calculated at the end of the game
    
    def _diversity_scoring(
        self, game: Game, agent1: Agent, agent2: Agent, interaction_result: Dict[str, Any]
    ) -> None:
        """Diversity scoring mode based on unique secrets collected.
        
        Args:
            game: The current game
            agent1: First agent in the interaction
            agent2: Second agent in the interaction
            interaction_result: Dictionary containing interaction results
        """
        agent1_revealed = interaction_result["agent1_revealed_secret"]
        agent2_revealed = interaction_result["agent2_revealed_secret"]
        
        # Update collected secrets
        if agent1_revealed:
            if agent1.secret not in agent2.collected_secrets:
                agent2.collected_secrets.append(agent1.secret)
                # Give points for new unique secret
                agent2.score += game.rules["scoring"]["points_per_unique_secret"]
        
        if agent2_revealed:
            if agent2.secret not in agent1.collected_secrets:
                agent1.collected_secrets.append(agent2.secret)
                # Give points for new unique secret
                agent1.score += game.rules["scoring"]["points_per_unique_secret"]
    
    def _targeted_scoring(
        self, game: Game, agent1: Agent, agent2: Agent, interaction_result: Dict[str, Any]
    ) -> None:
        """Targeted secret scoring mode.
        
        Args:
            game: The current game
            agent1: First agent in the interaction
            agent2: Second agent in the interaction
            interaction_result: Dictionary containing interaction results
        """
        agent1_revealed = interaction_result["agent1_revealed_secret"]
        agent2_revealed = interaction_result["agent2_revealed_secret"]
        
        # Update collected secrets and check for targeted secret
        if agent1_revealed:
            if agent1.secret not in agent2.collected_secrets:
                agent2.collected_secrets.append(agent1.secret)
                # Standard points
                agent2.score += game.rules["scoring"]["standard_secret_points"]
                
                # Check if this is the targeted secret
                if game.targeted_secret and agent1.secret == game.targeted_secret:
                    agent2.score += game.rules["scoring"]["targeted_secret_points"]
                    logger.info(f"Agent {agent2.name} found the targeted secret!")
        
        if agent2_revealed:
            if agent2.secret not in agent1.collected_secrets:
                agent1.collected_secrets.append(agent2.secret)
                # Standard points
                agent1.score += game.rules["scoring"]["standard_secret_points"]
                
                # Check if this is the targeted secret
                if game.targeted_secret and agent2.secret == game.targeted_secret:
                    agent1.score += game.rules["scoring"]["targeted_secret_points"]
                    logger.info(f"Agent {agent1.name} found the targeted secret!")
    
    def calculate_final_scores(self, game: Game) -> Dict[str, int]:
        """Calculate final scores based on game mode.
        
        Args:
            game: The game for which to calculate final scores
            
        Returns:
            Dictionary mapping agent IDs to final scores
        """
        if game.mode == GameMode.STANDARD:
            return {agent.id: agent.score for agent in game.agents}
        
        elif game.mode == GameMode.RETAINED:
            # Calculate rarity of each secret
            secret_counts = {}
            for agent in game.agents:
                for secret in agent.collected_secrets:
                    secret_counts[secret] = secret_counts.get(secret, 0) + 1
            
            # Calculate rarity scores
            total_agents = len(game.agents)
            scores = {}
            for agent in game.agents:
                score = 0
                for secret in agent.collected_secrets:
                    # Rarity = 1 / (proportion of agents with this secret)
                    rarity = total_agents / secret_counts[secret]
                    score += rarity
                scores[agent.id] = score
            return scores
        
        elif game.mode == GameMode.DIVERSITY:
            # Score based on unique secrets
            return {
                agent.id: len(set(agent.collected_secrets)) * game.rules["scoring"]["points_per_unique_secret"]
                for agent in game.agents
            }
        
        elif game.mode == GameMode.TARGETED:
            # Add bonus for targeted secret
            scores = {agent.id: agent.score for agent in game.agents}
            
            # No need to add additional points here as they're already added during the game
            return scores
        
        return {agent.id: agent.score for agent in game.agents} 