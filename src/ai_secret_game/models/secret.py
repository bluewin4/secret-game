"""Secret model for AI Secret Trading Game."""

from dataclasses import dataclass


@dataclass
class Secret:
    """Represents a secret in the game.
    
    Attributes:
        id: Unique identifier for the secret
        content: The actual secret content
        owner_id: ID of the agent who originally owns this secret
        rarity: Rarity factor used in RETAINED mode for scoring
        is_targeted: Whether this is a targeted secret in TARGETED mode
    """
    id: str
    content: str
    owner_id: str
    rarity: float = 1.0  # Used in RETAINED mode for scoring
    is_targeted: bool = False 