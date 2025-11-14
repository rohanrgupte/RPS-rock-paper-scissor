"""
Base strategy class for Rock Paper Scissor
"""

import random
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Base class for all RPS strategies"""
    
    MOVES = ['rock', 'paper', 'scissor']
    BEATS = {'rock': 'scissor', 'paper': 'rock', 'scissor': 'paper'}
    MOVES_TO_IDX = {'rock': 0, 'paper': 1, 'scissor': 2}
    
    def __init__(self, name, description, short_description):
        self.name = name
        self.description = description
        self.short_description = short_description
    
    @abstractmethod
    def predict(self, user_history, ai_history, outcome_history):
        """
        Predict the user's next move and return the counter move
        
        Args:
            user_history: List of user's moves
            ai_history: List of AI's moves
            outcome_history: List of outcomes ('user', 'ai', 'tie')
        
        Returns:
            str: The move to counter predicted user move (rock, paper, or scissor)
        """
        pass
    
    def can_predict(self, user_history):
        """Check if strategy has enough history to make prediction"""
        return len(user_history) >= self.min_history_length()
    
    def min_history_length(self):
        """Minimum history length required for prediction"""
        return 1

