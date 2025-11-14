"""
Simple pattern detection strategies
"""

from collections import Counter
from strategies.base_strategy import BaseStrategy


class AlwaysRockStrategy(BaseStrategy):
    """Detects if opponent always plays rock and counters with paper"""
    
    def __init__(self):
        super().__init__(
            name="Always Rock",
            description="If the opponent consistently plays rock (e.g., 80%+ of the time), this strategy counters with paper. This is effective against opponents who favor a single move.",
            short_description="Counters opponents who always play rock"
        )
    
    def predict(self, user_history, ai_history, outcome_history):
        if len(user_history) < 3:
            return None
        
        # Check if user always or mostly plays rock
        move_counts = Counter(user_history)
        total = len(user_history)
        
        rock_ratio = move_counts.get('rock', 0) / total
        
        # If 80%+ are rock, counter with paper
        if rock_ratio >= 0.8:
            return self.BEATS['rock']  # Return 'paper'
        
        return None


class AlwaysPaperStrategy(BaseStrategy):
    """Detects if opponent always plays paper and counters with scissor"""
    
    def __init__(self):
        super().__init__(
            name="Always Paper",
            description="If the opponent consistently plays paper (e.g., 80%+ of the time), this strategy counters with scissor. This detects and exploits single-move patterns.",
            short_description="Counters opponents who always play paper"
        )
    
    def predict(self, user_history, ai_history, outcome_history):
        if len(user_history) < 3:
            return None
        
        move_counts = Counter(user_history)
        total = len(user_history)
        
        paper_ratio = move_counts.get('paper', 0) / total
        
        if paper_ratio >= 0.8:
            return self.BEATS['paper']  # Return 'scissor'
        
        return None


class AlwaysScissorStrategy(BaseStrategy):
    """Detects if opponent always plays scissor and counters with rock"""
    
    def __init__(self):
        super().__init__(
            name="Always Scissor",
            description="If the opponent consistently plays scissor (e.g., 80%+ of the time), this strategy counters with rock. This identifies repetitive single-move behavior.",
            short_description="Counters opponents who always play scissor"
        )
    
    def predict(self, user_history, ai_history, outcome_history):
        if len(user_history) < 3:
            return None
        
        move_counts = Counter(user_history)
        total = len(user_history)
        
        scissor_ratio = move_counts.get('scissor', 0) / total
        
        if scissor_ratio >= 0.8:
            return self.BEATS['scissor']  # Return 'rock'
        
        return None


class CycleStrategy(BaseStrategy):
    """Detects rock->paper->scissor cycles and counters"""
    
    def __init__(self):
        super().__init__(
            name="Cycle Detection",
            description="Detects if the opponent follows a repeating cycle pattern (rock->paper->scissor->rock...). Once detected, it predicts the next move in the cycle and counters it. This exploits predictable sequential patterns.",
            short_description="Detects and counters rock-paper-scissor cycles"
        )
    
    def min_history_length(self):
        return 6  # Need at least 2 full cycles to detect
    
    def predict(self, user_history, ai_history, outcome_history):
        if len(user_history) < 6:
            return None
        
        # Check for cycle pattern
        cycle_pattern = ['rock', 'paper', 'scissor']
        
        # Check if recent moves follow the cycle
        recent = list(user_history[-6:])
        
        # Try to match cycle starting from different positions
        for start_idx in range(3):
            matches = 0
            for i in range(min(6, len(recent) - start_idx)):
                expected_move = cycle_pattern[(start_idx + i) % 3]
                if recent[start_idx + i] == expected_move:
                    matches += 1
                else:
                    break
            
            # If we matched at least 4 moves, assume cycle
            if matches >= 4:
                # Predict next move in cycle
                position = (start_idx + matches) % 3
                predicted_move = cycle_pattern[position]
                return self.BEATS[predicted_move]
        
        return None


class PairPatternStrategy(BaseStrategy):
    """Detects repeating pairs (e.g., rock-paper-rock-paper) and counters"""
    
    def __init__(self):
        super().__init__(
            name="Pair Pattern",
            description="Detects when the opponent repeats a pair of moves (e.g., rock-paper-rock-paper or scissor-rock-scissor-rock). Once identified, it predicts the next move in the pair pattern and counters it. This catches alternating patterns.",
            short_description="Detects and counters repeating two-move patterns"
        )
    
    def min_history_length(self):
        return 4  # Need at least 2 pairs to detect
    
    def predict(self, user_history, ai_history, outcome_history):
        if len(user_history) < 4:
            return None
        
        recent = list(user_history[-6:])  # Check last 6 moves
        
        # Try different pair lengths
        for pair_len in [2, 3]:
            if len(recent) < pair_len * 2:
                continue
            
            # Check if pattern repeats
            pattern = tuple(recent[-pair_len:])
            
            # Check if this pattern appears earlier
            matches = 0
            for i in range(len(recent) - pair_len * 2, -1, -pair_len):
                if i >= 0 and tuple(recent[i:i+pair_len]) == pattern:
                    matches += 1
                else:
                    break
            
            # If we found repeating pattern (at least 2 instances)
            if matches >= 1:
                # Predict next move in pattern
                predicted_move = pattern[0]  # First move of pattern repeats
                return self.BEATS[predicted_move]
        
        # Check for simple alternating pairs
        if len(recent) >= 4:
            # Check if last 4 are alternating (A-B-A-B pattern)
            if recent[-4] == recent[-2] and recent[-3] == recent[-1] and recent[-4] != recent[-3]:
                # Predict next will be the first of the pair
                predicted_move = recent[-2]
                return self.BEATS[predicted_move]
        
        return None

