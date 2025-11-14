"""
Simple pattern detection strategies
"""

from collections import Counter
from strategies.base_strategy import BaseStrategy


class AlwaysRockStrategy(BaseStrategy):
    """Always predicts opponent will play rock and counters with paper"""
    
    def __init__(self):
        super().__init__(
            name="Always Rock",
            description="Always predicts the opponent will play rock and counters with paper. This is a simple strategy that works well against opponents who favor rock.",
            short_description="Always counters rock with paper"
        )
    
    def min_history_length(self):
        return 0  # No history needed - always works
    
    def predict(self, user_history, ai_history, outcome_history):
        # Simplified: always return paper to counter rock
        # To beat rock, we need paper (rock is beaten by paper)
        return self.BEATEN_BY['rock']  # Return 'paper'


class AlwaysPaperStrategy(BaseStrategy):
    """Always predicts opponent will play paper and counters with scissor"""
    
    def __init__(self):
        super().__init__(
            name="Always Paper",
            description="Always predicts the opponent will play paper and counters with scissor. This is a simple strategy that works well against opponents who favor paper.",
            short_description="Always counters paper with scissor"
        )
    
    def min_history_length(self):
        return 0  # No history needed - always works
    
    def predict(self, user_history, ai_history, outcome_history):
        # Simplified: always return scissor to counter paper
        # To beat paper, we need scissor (paper is beaten by scissor)
        return self.BEATEN_BY['paper']  # Return 'scissor'


class AlwaysScissorStrategy(BaseStrategy):
    """Always predicts opponent will play scissor and counters with rock"""
    
    def __init__(self):
        super().__init__(
            name="Always Scissor",
            description="Always predicts the opponent will play scissor and counters with rock. This is a simple strategy that works well against opponents who favor scissor.",
            short_description="Always counters scissor with rock"
        )
    
    def min_history_length(self):
        return 0  # No history needed - always works
    
    def predict(self, user_history, ai_history, outcome_history):
        # Simplified: always return rock to counter scissor
        # To beat scissor, we need rock (scissor is beaten by rock)
        return self.BEATEN_BY['scissor']  # Return 'rock'


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
                return self.BEATEN_BY[predicted_move]
        
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
                return self.BEATEN_BY[predicted_move]
        
        # Check for simple alternating pairs
        if len(recent) >= 4:
            # Check if last 4 are alternating (A-B-A-B pattern)
            if recent[-4] == recent[-2] and recent[-3] == recent[-1] and recent[-4] != recent[-3]:
                # Predict next will be the first of the pair
                predicted_move = recent[-2]
                return self.BEATEN_BY[predicted_move]
        
        return None

