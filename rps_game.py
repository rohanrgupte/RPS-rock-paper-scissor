"""
Rock Paper Scissor Game with Multi-Armed Bandit AI
Uses UCB algorithm with Random Forest for adaptive opponent
"""

import random
import numpy as np
from collections import deque, Counter
from sklearn.ensemble import RandomForestClassifier


class MABRPSGame:
    """Rock Paper Scissor game with Multi-Armed Bandit AI opponent"""
    
    MOVES = ['rock', 'paper', 'scissor']
    BEATS = {'rock': 'scissor', 'paper': 'rock', 'scissor': 'paper'}
    
    def __init__(self, history_size=50):
        self.history_size = history_size
        self.user_history = deque(maxlen=history_size)
        self.ai_history = deque(maxlen=history_size)
        self.outcome_history = deque(maxlen=history_size)  # Track wins/losses
        
        # MAB: 3 arms = 3 different strategies
        self.num_arms = 3  # 0: RF Prediction, 1: Counter-Pattern, 2: UCB Direct
        self.arm_counts = np.zeros(self.num_arms)  # How many times each arm used
        self.arm_rewards = np.zeros(self.num_arms)  # Cumulative rewards per arm
        self.arm_values = np.zeros(self.num_arms)  # Average reward per arm
        self.total_plays = 0
        
        # Random Forest for pattern recognition
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        self.rf_fitted = False
        
        # Pattern detection
        self.pattern_window = 3
        
        # Scores
        self.user_score = 0
        self.ai_score = 0
        
    def get_winner(self, user_move, ai_move):
        """Determine round winner"""
        if user_move == ai_move:
            return 'tie'
        return 'user' if self.BEATS[user_move] == ai_move else 'ai'
    
    def get_reward(self, winner):
        """Get reward for MAB update: win=1, tie=0, loss=-1"""
        if winner == 'ai':
            return 1
        elif winner == 'tie':
            return 0
        else:
            return -1
    
    def update_scores(self, winner):
        """Update game scores"""
        if winner == 'user':
            self.user_score += 1
        elif winner == 'ai':
            self.ai_score += 1
    
    def predict_user_move_rf(self):
        """Arm 0: Predict user move using Random Forest"""
        if len(self.user_history) < 4:
            return random.choice(self.MOVES)
        
        try:
            # Build features from history
            X, y = self._build_features()
            
            if len(X) < 2:
                return random.choice(self.MOVES)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train or retrain model
            self.rf_model.fit(X, y)
            self.rf_fitted = True
            
            # Predict next move
            last_features = self._get_last_features().reshape(1, -1)
            predicted_idx = self.rf_model.predict(last_features)[0]
            predicted_move = self.MOVES[predicted_idx]
            
            # Counter the predicted move
            return self.BEATS[predicted_move]
        except Exception:
            return random.choice(self.MOVES)
    
    def predict_user_move_pattern(self):
        """Arm 1: Counter-pattern detection"""
        if len(self.user_history) < self.pattern_window + 1:
            return random.choice(self.MOVES)
        
        try:
            # Look for recent patterns
            recent_moves = list(self.user_history)[-self.pattern_window:]
            
            # Count transitions: what move follows this pattern
            transition_counts = {move: 0 for move in self.MOVES}
            
            # Analyze patterns in history: what move follows what sequence
            history_list = list(self.user_history)
            for i in range(len(history_list) - self.pattern_window):
                pattern = tuple(history_list[i:i+self.pattern_window])
                next_move = history_list[i + self.pattern_window]
                
                # Check if this pattern matches recent pattern
                if pattern == tuple(recent_moves):
                    transition_counts[next_move] += 1
            
            # Find most likely next move based on pattern
            if sum(transition_counts.values()) > 0:
                most_likely = max(transition_counts, key=transition_counts.get)
                return self.BEATS[most_likely]
            else:
                # Fallback: counter most common move in history
                move_counts = Counter(history_list)
                if move_counts:
                    most_common = move_counts.most_common(1)[0][0]
                    return self.BEATS[most_common]
                return random.choice(self.MOVES)
        except Exception:
            return random.choice(self.MOVES)
    
    def predict_user_move_ucb_direct(self):
        """Arm 2: Direct UCB on moves - predicts which move user will play"""
        if len(self.user_history) < 3:
            return random.choice(self.MOVES)
        
        try:
            # Track user's move performance (how often they win with each move)
            move_counts = {move: 1 for move in self.MOVES}  # Laplace smoothing
            move_wins = {move: 0 for move in self.MOVES}
            move_values = {move: 0.0 for move in self.MOVES}
            
            history_list = list(self.user_history)
            outcome_list = list(self.outcome_history)
            
            # Calculate win rate for each move
            for i, move in enumerate(history_list):
                move_counts[move] += 1
                if i < len(outcome_list) and outcome_list[i] == 'user':
                    move_wins[move] += 1
            
            total_count = sum(move_counts.values()) - len(self.MOVES)  # Subtract smoothing
            
            # Calculate UCB: user likely to play move with highest expected win rate
            c = np.sqrt(2 * np.log(total_count + 1))
            ucb_values = {}
            for move in self.MOVES:
                if move_counts[move] > 0:
                    move_values[move] = move_wins[move] / move_counts[move]
                    ucb_values[move] = move_values[move] + c * np.sqrt(
                        np.log(total_count + 1) / move_counts[move]
                    )
                else:
                    ucb_values[move] = float('inf')
            
            # Predict user's best move (highest UCB) and counter it
            predicted_user_move = max(ucb_values, key=ucb_values.get)
            return self.BEATS[predicted_user_move]
        except Exception:
            return random.choice(self.MOVES)
    
    def _build_features(self):
        """Build features for Random Forest"""
        X, y = [], []
        history_list = list(self.user_history)
        ai_history_list = list(self.ai_history)
        outcome_list = list(self.outcome_history)
        
        for i in range(1, len(history_list)):
            features = []
            
            # Last N moves as features
            for j in range(min(5, i)):
                idx = i - 1 - j
                if idx >= 0:
                    features.append(self.MOVES.index(history_list[idx]))
                    if idx < len(ai_history_list):
                        features.append(self.MOVES.index(ai_history_list[idx]))
                    else:
                        features.append(0)
                    if idx < len(outcome_list):
                        # Encode outcome: -1=loss, 0=tie, 1=win (from AI perspective)
                        outcome = outcome_list[idx]
                        features.append(1 if outcome == 'ai' else (-1 if outcome == 'user' else 0))
                    else:
                        features.append(0)
                else:
                    features.extend([0, 0, 0])
            
            # Pad to fixed size (15 features = 5 moves * 3 features each)
            while len(features) < 15:
                features.append(0)
            
            X.append(features[:15])
            y.append(self.MOVES.index(history_list[i]))
        
        return X, y
    
    def _get_last_features(self):
        """Get features for last move pattern"""
        features = []
        history_list = list(self.user_history)
        ai_history_list = list(self.ai_history)
        outcome_list = list(self.outcome_history)
        
        for j in range(min(5, len(history_list))):
            idx = len(history_list) - 1 - j
            if idx >= 0:
                features.append(self.MOVES.index(history_list[idx]))
                if idx < len(ai_history_list):
                    features.append(self.MOVES.index(ai_history_list[idx]))
                else:
                    features.append(0)
                if idx < len(outcome_list):
                    outcome = outcome_list[idx]
                    features.append(1 if outcome == 'ai' else (-1 if outcome == 'user' else 0))
                else:
                    features.append(0)
            else:
                features.extend([0, 0, 0])
        
        while len(features) < 15:
            features.append(0)
        
        return np.array(features[:15])
    
    def select_arm_ucb(self):
        """Select arm using Upper Confidence Bound algorithm"""
        if self.total_plays < self.num_arms:
            # Exploration: try each arm once initially
            return self.total_plays
        
        # Calculate UCB for each arm
        ucb_values = np.zeros(self.num_arms)
        c = np.sqrt(2 * np.log(self.total_plays + 1))
        
        for arm in range(self.num_arms):
            if self.arm_counts[arm] == 0:
                ucb_values[arm] = float('inf')
            else:
                ucb_values[arm] = self.arm_values[arm] + c * np.sqrt(
                    np.log(self.total_plays + 1) / self.arm_counts[arm]
                )
        
        # Select arm with highest UCB
        return np.argmax(ucb_values)
    
    def update_arm(self, arm, reward):
        """Update arm statistics using reward"""
        self.arm_counts[arm] += 1
        self.total_plays += 1
        
        # Update average reward
        n = self.arm_counts[arm]
        old_value = self.arm_values[arm]
        self.arm_values[arm] = ((n - 1) / n) * old_value + (1 / n) * reward
        self.arm_rewards[arm] += reward
    
    def predict_user_move(self):
        """Main prediction using MAB - returns (move, selected_arm)"""
        # Select strategy (arm) using UCB
        selected_arm = self.select_arm_ucb()
        
        # Get move from selected strategy
        if selected_arm == 0:
            move = self.predict_user_move_rf()
        elif selected_arm == 1:
            move = self.predict_user_move_pattern()
        else:  # selected_arm == 2
            move = self.predict_user_move_ucb_direct()
        
        return move, selected_arm
    
    def play_round(self, user_move):
        """Play a single round"""
        # Predict and select AI move
        ai_move, selected_arm = self.predict_user_move()
        winner = self.get_winner(user_move, ai_move)
        reward = self.get_reward(winner)
        
        # Update MAB with reward
        self.update_arm(selected_arm, reward)
        
        # Update history
        self.user_history.append(user_move)
        self.ai_history.append(ai_move)
        self.outcome_history.append(winner)
        
        # Update scores
        self.update_scores(winner)
        
        return ai_move, winner, selected_arm
    
    def reset_game(self):
        """Reset game state"""
        self.user_history.clear()
        self.ai_history.clear()
        self.outcome_history.clear()
        self.arm_counts = np.zeros(self.num_arms)
        self.arm_rewards = np.zeros(self.num_arms)
        self.arm_values = np.zeros(self.num_arms)
        self.total_plays = 0
        self.rf_fitted = False
        self.user_score = 0
        self.ai_score = 0
