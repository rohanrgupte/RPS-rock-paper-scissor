"""
Enhanced Rock Paper Scissor Game with Multi-Armed Bandit AI
Uses modular strategies from strategies directory
"""

import random
import numpy as np
from collections import deque, defaultdict
from strategies import (
    AlwaysRockStrategy,
    AlwaysPaperStrategy,
    AlwaysScissorStrategy,
    CycleStrategy,
    PairPatternStrategy,
    IocainePowderStrategy,
    MarkovChainsStrategy,
    WinStayLoseShiftStrategy,
    FrequencyDecayStrategy,
    RandomForestStrategy,
    TransitionMatrixStrategy
)


class MABRPSGame:
    """
    Enhanced Rock Paper Scissor game with 11 strategies (5 simple + 6 advanced)
    Uses Multi-Armed Bandit (UCB) with decay for recent transactions only
    """
    
    MOVES = ['rock', 'paper', 'scissor']
    BEATS = {'rock': 'scissor', 'paper': 'rock', 'scissor': 'paper'}
    
    def __init__(self, history_size=200, mab_window=20):
        self.history_size = history_size
        self.mab_window = mab_window  # Only use recent N transactions for MAB
        self.user_history = deque(maxlen=history_size)
        self.ai_history = deque(maxlen=history_size)
        self.outcome_history = deque(maxlen=history_size)
        
        # Initialize all strategies
        self.strategies = [
            AlwaysRockStrategy(),
            AlwaysPaperStrategy(),
            AlwaysScissorStrategy(),
            CycleStrategy(),
            PairPatternStrategy(),
            IocainePowderStrategy(),
            MarkovChainsStrategy(),
            WinStayLoseShiftStrategy(),
            FrequencyDecayStrategy(),
            RandomForestStrategy(),
            TransitionMatrixStrategy()
        ]
        
        self.num_arms = len(self.strategies)
        
        # MAB with decay: track recent transactions
        self.arm_recent_rewards = []  # List of (arm, reward, round) tuples for recent rounds
        self.arm_counts = np.zeros(self.num_arms)
        self.arm_rewards = np.zeros(self.num_arms)
        self.arm_values = np.zeros(self.num_arms)
        self.total_plays = 0
        
        # Win percentage tracking for leaderboard
        self.arm_wins = np.zeros(self.num_arms)
        self.arm_losses = np.zeros(self.num_arms)
        self.arm_ties = np.zeros(self.num_arms)
        
        # Scores
        self.user_score = 0
        self.ai_score = 0
    
    def get_winner(self, user_move, ai_move):
        """Determine round winner"""
        if user_move == ai_move:
            return 'tie'
        return 'user' if self.BEATS[user_move] == ai_move else 'ai'
    
    def get_reward(self, winner):
        """Get reward for MAB update: win=1, tie=0.1, loss=-1"""
        if winner == 'ai':
            return 1.0
        elif winner == 'tie':
            return 0.1
        else:
            return -1.0
    
    def update_scores(self, winner):
        """Update game scores"""
        if winner == 'user':
            self.user_score += 1
        elif winner == 'ai':
            self.ai_score += 1
    
    def select_arm_ucb(self):
        """Select arm using Upper Confidence Bound with decay for recent transactions"""
        if self.total_plays < self.num_arms:
            return self.total_plays
        
        # Recalculate values based on recent transactions only
        recent_rewards = self.arm_recent_rewards[-self.mab_window:]
        
        # Calculate decayed values for each arm
        ucb_values = np.zeros(self.num_arms)
        arm_counts_recent = np.zeros(self.num_arms)
        arm_values_recent = np.zeros(self.num_arms)
        
        # Apply decay factor (older in window get less weight)
        decay_factor = 0.95
        for i, (arm, reward, round_num) in enumerate(recent_rewards):
            age = len(recent_rewards) - i - 1
            weight = decay_factor ** age
            arm_counts_recent[arm] += weight
            arm_values_recent[arm] += reward * weight
        
        # Normalize values
        for arm in range(self.num_arms):
            if arm_counts_recent[arm] > 0:
                arm_values_recent[arm] /= arm_counts_recent[arm]
        
        # Calculate UCB with recent values
        c = np.sqrt(2 * np.log(self.total_plays + 1))
        for arm in range(self.num_arms):
            if arm_counts_recent[arm] == 0:
                ucb_values[arm] = float('inf')
            else:
                ucb_values[arm] = arm_values_recent[arm] + c * np.sqrt(
                    np.log(self.total_plays + 1) / (arm_counts_recent[arm] + 1)
                )
        
        return np.argmax(ucb_values)
    
    def update_arm(self, arm, reward, winner):
        """Update arm statistics with decay"""
        self.arm_counts[arm] += 1
        self.total_plays += 1
        
        # Track in recent rewards list
        self.arm_recent_rewards.append((arm, reward, self.total_plays))
        
        # Keep only recent transactions
        if len(self.arm_recent_rewards) > self.mab_window:
            self.arm_recent_rewards.pop(0)
        
        # Update win/loss/tie tracking for leaderboard
        if winner == 'ai':
            self.arm_wins[arm] += 1
        elif winner == 'user':
            self.arm_losses[arm] += 1
        else:
            self.arm_ties[arm] += 1
        
        # Update average reward
        n = self.arm_counts[arm]
        old_value = self.arm_values[arm]
        self.arm_values[arm] = ((n - 1) / n) * old_value + (1 / n) * reward
        self.arm_rewards[arm] += reward
    
    def get_win_percentage(self, arm):
        """Calculate win percentage for an arm"""
        total_games = self.arm_wins[arm] + self.arm_losses[arm] + self.arm_ties[arm]
        if total_games == 0:
            return 0.0
        return (self.arm_wins[arm] / total_games) * 100
    
    def get_leaderboard(self):
        """Get sorted leaderboard by win percentage"""
        leaderboard = []
        for i, strategy in enumerate(self.strategies):
            win_pct = self.get_win_percentage(i)
            total_games = int(self.arm_wins[i] + self.arm_losses[i] + self.arm_ties[i])
            leaderboard.append({
                'arm': i,
                'name': strategy.name,
                'win_pct': win_pct,
                'wins': int(self.arm_wins[i]),
                'losses': int(self.arm_losses[i]),
                'ties': int(self.arm_ties[i]),
                'total_games': total_games,
                'avg_reward': float(self.arm_values[i])
            })
        
        # Sort by win percentage (descending), then by total games
        leaderboard.sort(key=lambda x: (x['win_pct'], x['total_games']), reverse=True)
        return leaderboard
    
    def predict_user_move(self):
        """Main prediction using MAB - returns (move, selected_arm)"""
        selected_arm = self.select_arm_ucb()
        strategy = self.strategies[selected_arm]
        
        # Try to get prediction from strategy
        user_hist = list(self.user_history)
        ai_hist = list(self.ai_history)
        outcome_hist = list(self.outcome_history)
        
        if strategy.can_predict(user_hist):
            move = strategy.predict(user_hist, ai_hist, outcome_hist)
            if move:
                return move, selected_arm
        
        # Fallback to random if strategy can't predict
        return random.choice(self.MOVES), selected_arm
    
    def play_round(self, user_move):
        """Play a single round"""
        # Predict and select AI move
        ai_move, selected_arm = self.predict_user_move()
        winner = self.get_winner(user_move, ai_move)
        reward = self.get_reward(winner)
        
        # Update MAB with reward
        self.update_arm(selected_arm, reward, winner)
        
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
        self.arm_recent_rewards.clear()
        self.arm_counts = np.zeros(self.num_arms)
        self.arm_rewards = np.zeros(self.num_arms)
        self.arm_values = np.zeros(self.num_arms)
        self.total_plays = 0
        self.arm_wins = np.zeros(self.num_arms)
        self.arm_losses = np.zeros(self.num_arms)
        self.arm_ties = np.zeros(self.num_arms)
        self.user_score = 0
        self.ai_score = 0
