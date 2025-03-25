"""Unit tests for the scoring service."""

import pytest

from src.ai_secret_game.models.game import GameMode
from src.ai_secret_game.services.scoring_service import ScoringService


def test_standard_scoring_no_reveal(scoring_service, standard_game, test_agents, no_reveal_interaction):
    """Test standard scoring when no secrets are revealed."""
    agent1, agent2 = test_agents[0], test_agents[1]
    
    # Reset scores
    agent1.score = 0
    agent2.score = 0
    
    # Run scoring
    scoring_service._standard_scoring(standard_game, agent1, agent2, no_reveal_interaction)
    
    # Verify scores - both should be 0
    assert agent1.score == 0
    assert agent2.score == 0


def test_standard_scoring_both_reveal(scoring_service, standard_game, test_agents, revealed_interaction):
    """Test standard scoring when both agents reveal secrets."""
    agent1, agent2 = test_agents[0], test_agents[1]
    
    # Reset scores
    agent1.score = 0
    agent2.score = 0
    
    # Run scoring
    scoring_service._standard_scoring(standard_game, agent1, agent2, revealed_interaction)
    
    # Verify scores - both should get 1 point
    assert agent1.score == 1
    assert agent2.score == 1


def test_standard_scoring_one_reveal(scoring_service, standard_game, test_agents, one_sided_interaction):
    """Test standard scoring when only one agent reveals a secret."""
    agent1, agent2 = test_agents[0], test_agents[1]
    
    # Reset scores
    agent1.score = 0
    agent2.score = 0
    
    # Run scoring
    scoring_service._standard_scoring(standard_game, agent1, agent2, one_sided_interaction)
    
    # Verify scores - agent2 should get 3 points for getting agent1's secret
    assert agent1.score == 0
    assert agent2.score == 3


def test_diversity_scoring(scoring_service, diversity_game, test_agents):
    """Test diversity scoring mode."""
    agent1, agent2 = test_agents[0], test_agents[1]
    
    # Reset scores and collected secrets
    agent1.score = 0
    agent2.score = 0
    agent1.collected_secrets = []
    agent2.collected_secrets = []
    
    # Create interaction where agent1 reveals secret
    interaction = {
        "agent1_revealed_secret": True,
        "agent2_revealed_secret": False,
    }
    
    # Run scoring
    scoring_service._diversity_scoring(diversity_game, agent1, agent2, interaction)
    
    # Verify agent2 collected agent1's secret and got points
    assert "Secret 1" in agent2.collected_secrets
    assert agent2.score == diversity_game.rules["scoring"]["points_per_unique_secret"]
    
    # Agent1 didn't collect any secrets
    assert len(agent1.collected_secrets) == 0
    assert agent1.score == 0


def test_targeted_scoring(scoring_service, targeted_game, test_agents):
    """Test targeted scoring mode."""
    agent1, agent2 = test_agents[0], test_agents[1]  # agent2 has the targeted secret
    
    # Reset scores and collected secrets
    agent1.score = 0
    agent2.score = 0
    agent1.collected_secrets = []
    agent2.collected_secrets = []
    
    # Create interaction where agent2 reveals the targeted secret
    interaction = {
        "agent1_revealed_secret": False,
        "agent2_revealed_secret": True,
    }
    
    # Run scoring
    scoring_service._targeted_scoring(targeted_game, agent1, agent2, interaction)
    
    # Verify agent1 collected the targeted secret and got bonus points
    assert "Secret 2" in agent1.collected_secrets
    assert agent1.score == (
        targeted_game.rules["scoring"]["standard_secret_points"] + 
        targeted_game.rules["scoring"]["targeted_secret_points"]
    )
    
    # Agent2 didn't collect any secrets
    assert len(agent2.collected_secrets) == 0
    assert agent2.score == 0


def test_retained_scoring_secret_collection(scoring_service, retained_game, test_agents):
    """Test retained scoring mode secret collection."""
    agent1, agent2 = test_agents[0], test_agents[1]
    
    # Reset collected secrets
    agent1.collected_secrets = []
    agent2.collected_secrets = []
    
    # Create interaction where both reveal secrets
    interaction = {
        "agent1_revealed_secret": True,
        "agent2_revealed_secret": True,
    }
    
    # Run scoring
    scoring_service._retained_scoring(retained_game, agent1, agent2, interaction)
    
    # Verify both agents collected each other's secrets
    assert "Secret 2" in agent1.collected_secrets
    assert "Secret 1" in agent2.collected_secrets


def test_calculate_final_scores_standard(scoring_service, standard_game, test_agents):
    """Test calculating final scores in standard mode."""
    # Set predefined scores
    test_agents[0].score = 3
    test_agents[1].score = 1
    test_agents[2].score = 5
    test_agents[3].score = 0
    
    # Calculate final scores
    final_scores = scoring_service.calculate_final_scores(standard_game)
    
    # Verify scores are returned correctly
    assert final_scores == {
        "agent1": 3,
        "agent2": 1,
        "agent3": 5,
        "agent4": 0
    }


def test_calculate_final_scores_retained(scoring_service, retained_game, test_agents):
    """Test calculating final scores in retained mode."""
    # Set collected secrets to create different rarities
    test_agents[0].collected_secrets = ["Secret 2", "Secret 3"]  # 2 secrets
    test_agents[1].collected_secrets = ["Secret 1"]  # 1 secret
    test_agents[2].collected_secrets = ["Secret 1", "Secret 4"]  # 2 secrets
    test_agents[3].collected_secrets = ["Secret 1", "Secret 2", "Secret 3"]  # 3 secrets
    
    # Calculate final scores
    final_scores = scoring_service.calculate_final_scores(retained_game)
    
    # Secret 1 is held by 3 agents, rarity = 4/3
    # Secret 2 is held by 2 agents, rarity = 4/2 = 2
    # Secret 3 is held by 2 agents, rarity = 4/2 = 2
    # Secret 4 is held by 1 agent, rarity = 4/1 = 4
    
    # Expected scores:
    # agent1: 2 + 2 = 4
    # agent2: 4/3 ≈ 1.33
    # agent3: 4/3 + 4 ≈ 5.33
    # agent4: 4/3 + 2 + 2 ≈ 5.33
    
    # Due to floating point, use approximation
    assert abs(final_scores["agent1"] - 4.0) < 0.01
    assert abs(final_scores["agent2"] - 1.33) < 0.01
    assert abs(final_scores["agent3"] - 5.33) < 0.01
    assert abs(final_scores["agent4"] - 5.33) < 0.01


def test_calculate_final_scores_diversity(scoring_service, diversity_game, test_agents):
    """Test calculating final scores in diversity mode."""
    # Set collected secrets with some duplicates
    test_agents[0].collected_secrets = ["Secret 2", "Secret 3", "Secret 3"]  # 2 unique
    test_agents[1].collected_secrets = ["Secret 1", "Secret 4"]  # 2 unique
    test_agents[2].collected_secrets = ["Secret 1", "Secret 2", "Secret 4"]  # 3 unique
    test_agents[3].collected_secrets = []  # 0 unique
    
    # Calculate final scores
    final_scores = scoring_service.calculate_final_scores(diversity_game)
    
    # Points per unique secret = 2
    # Expected scores:
    # agent1: 2 * 2 = 4
    # agent2: 2 * 2 = 4
    # agent3: 3 * 2 = 6
    # agent4: 0 * 2 = 0
    
    assert final_scores["agent1"] == 4
    assert final_scores["agent2"] == 4
    assert final_scores["agent3"] == 6
    assert final_scores["agent4"] == 0 