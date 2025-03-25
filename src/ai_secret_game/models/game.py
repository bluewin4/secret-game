"""Game model for AI Secret Trading Game."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum


class GameMode(Enum):
    """Game modes for the AI Secret Trading Game.
    
    Attributes:
        STANDARD: Basic game mode - secrets are one-time use
        RETAINED: Secrets are retained when traded, scoring based on rarity
        DIVERSITY: Scoring based on diversity of collected secrets
        TARGETED: Searching for specific high-value secrets
    """
    STANDARD = "standard"
    RETAINED = "retained"
    DIVERSITY = "diversity"
    TARGETED = "targeted"


@dataclass
class Game:
    """Represents a game session.
    
    Attributes:
        id: Unique identifier for the game
        mode: The game mode that determines scoring rules
        agents: List of agents participating in the game
        round: Current round number
        max_rounds: Maximum number of rounds in the game
        messages_per_round: Number of messages each agent can send per round
        rules: Dictionary containing game rules
        targeted_secret: Secret that's worth extra points in targeted mode
        targeted_secret_points: Points awarded for the targeted secret
        pairings_history: History of agent pairings in each round
    """
    id: str
    mode: GameMode
    agents: List["Agent"]  # Forward reference to avoid circular import
    round: int = 0
    max_rounds: int = 10
    messages_per_round: int = 3
    rules: Dict[str, Any] = field(default_factory=dict)
    targeted_secret: Optional[str] = None
    targeted_secret_points: int = 5
    pairings_history: List[List[Tuple[str, str]]] = field(default_factory=list)
    
    def initialize_rules(self) -> None:
        """Initialize game rules based on the game mode."""
        self.rules = {
            "mode": self.mode.value,
            "scoring": self._get_scoring_rules(),
            "max_rounds": self.max_rounds,
            "current_round": self.round,
            "messages_per_round": self.messages_per_round,
        }
        
        if self.mode == GameMode.TARGETED and self.targeted_secret:
            self.rules["targeted_secret"] = self.targeted_secret
            self.rules["targeted_secret_points"] = self.targeted_secret_points
    
    def _get_scoring_rules(self) -> Dict[str, Any]:
        """Get scoring rules based on game mode.
        
        Returns:
            Dictionary of scoring rules specific to the current game mode
        """
        if self.mode == GameMode.STANDARD:
            return {
                "no_secrets_revealed": 0,
                "both_secrets_revealed": 1,
                "only_opponent_secret_revealed": 3
            }
        elif self.mode == GameMode.RETAINED:
            return {
                "rarity_multiplier": True,
                "base_points_per_secret": 1
            }
        elif self.mode == GameMode.DIVERSITY:
            return {
                "points_per_unique_secret": 2
            }
        elif self.mode == GameMode.TARGETED:
            return {
                "standard_secret_points": 1,
                "targeted_secret_points": self.targeted_secret_points
            }
        return {} 