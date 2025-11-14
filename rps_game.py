"""
Rock Paper Scissor Game with Multi-Armed Bandit AI
Uses Random Forest classifier for adaptive opponent
"""

import random
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestClassifier


class MABRPSGame:
    """Rock Paper Scissor game with Multi-Armed Bandit AI opponent"""
    
    MOVES = ['rock', 'paper', 'scissor']
    BEATS = {'rock': 'scissor', 'paper': 'rock', 'scissor': 'paper'}
    
    def __init__(self, history_size=10):
        self.history_size = history_size
        self.user_history = deque(maxlen=history_size)
        self.ai_history = deque(maxlen=history_size)
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.fitted = False
        self.user_score = 0
        self.ai_score = 0
        
    def get_winner(self, user_move, ai_move):
        """Determine round winner"""
        if user_move == ai_move:
            return 'tie'
        return 'user' if self.BEATS[user_move] == ai_move else 'ai'
    
    def update_scores(self, winner):
        """Update game scores"""
        if winner == 'user':
            self.user_score += 1
        elif winner == 'ai':
            self.ai_score += 1
    
    def predict_user_move(self):
        """Predict next user move using pattern recognition"""
        if len(self.user_history) < 3:
            return random.choice(self.MOVES)
        
        # Build features from recent history patterns
        X = []
        y = []
        
        history_list = list(self.user_history)
        ai_history_list = list(self.ai_history)
        
        # Create training examples from history
        for i in range(1, len(history_list)):
            features = []
            # Use last 3 moves as features
            for j in range(min(3, i)):
                idx = i - 1 - j
                if idx >= 0:
                    features.append(self.MOVES.index(history_list[idx]))
                    features.append(self.MOVES.index(ai_history_list[idx]) if idx < len(ai_history_list) else 0)
                else:
                    features.extend([0, 0])
            
            # Pad to 6 features
            while len(features) < 6:
                features.append(0)
            
            X.append(features[:6])
            y.append(self.MOVES.index(history_list[i]))
        
        if len(X) < 1:
            return random.choice(self.MOVES)
        
        try:
            X = np.array(X)
            y = np.array(y)
            
            # Fit or refit model
            self.model.fit(X, y)
            self.fitted = True
            
            # Predict next move using most recent pattern
            last_features = X[-1].reshape(1, -1)
            predicted_idx = self.model.predict(last_features)[0]
            predicted_move = self.MOVES[predicted_idx]
            
            # Counter the predicted move
            return self.BEATS[predicted_move]
        except Exception:
            return random.choice(self.MOVES)
    
    def play_round(self, user_move):
        """Play a single round"""
        ai_move = self.predict_user_move()
        winner = self.get_winner(user_move, ai_move)
        
        # Update history
        self.user_history.append(user_move)
        self.ai_history.append(ai_move)
        
        # Update scores
        self.update_scores(winner)
        
        return ai_move, winner
    
    def reset_game(self):
        """Reset game state"""
        self.user_history.clear()
        self.ai_history.clear()
        self.fitted = False
        self.user_score = 0
        self.ai_score = 0

