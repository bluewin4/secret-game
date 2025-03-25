"""Custom exceptions for the AI Secret Trading Game."""


class GameError(Exception):
    """Base exception for game-related errors.
    
    This exception is raised when there's an error related to game logic.
    """
    pass


class AgentError(Exception):
    """Base exception for agent-related errors.
    
    This exception is raised when there's an error related to agent interactions.
    """
    pass


class ConfigError(Exception):
    """Base exception for configuration-related errors.
    
    This exception is raised when there's an error in the application configuration.
    """
    pass 