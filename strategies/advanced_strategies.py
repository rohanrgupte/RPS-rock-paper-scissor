"""
Advanced strategies for Rock Paper Scissor AI
"""

import random
import numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from strategies.base_strategy import BaseStrategy


class IocainePowderStrategy(BaseStrategy):
    """Iocaine Powder: Multi-pattern voting system"""
    
    def __init__(self):
        super().__init__(
            name="Iocaine Powder",
            description="This strategy checks multiple pattern lengths (1-5 moves) simultaneously and creates a voting system. Each pattern length predicts what move follows, and the predictions are weighted and combined to make a final decision. Humans often have patterns at different scales - they might repeat the last move, or follow 2-3 move sequences, or even longer patterns. By checking all lengths simultaneously, we catch patterns that single-length approaches miss. This is inspired by top-performing Kaggle solutions.",
            short_description="Multi-pattern voting system checking lengths 1-5"
        )
        self.pattern_lengths = [1, 2, 3, 4, 5]
        self.decay_factor = 0.95
    
    def min_history_length(self):
        return 3
    
    def predict(self, user_history, ai_history, outcome_history):
        if len(user_history) < 3:
            return None
        
        votes = defaultdict(float)
        history = list(user_history)
        
        for pattern_len in self.pattern_lengths:
            if len(history) < pattern_len + 1:
                continue
            
            recent_pattern = tuple(history[-pattern_len:])
            next_moves = defaultdict(float)
            
            for i in range(len(history) - pattern_len):
                pattern = tuple(history[i:i+pattern_len])
                if pattern == recent_pattern:
                    next_move = history[i + pattern_len]
                    weight = self.decay_factor ** (len(history) - i - pattern_len)
                    next_moves[next_move] += weight
            
            if next_moves:
                predicted = max(next_moves, key=next_moves.get)
                pattern_weight = next_moves[predicted] * (pattern_len / max(self.pattern_lengths))
                votes[predicted] += pattern_weight
        
        if sum(votes.values()) > 0:
            predicted_move = max(votes, key=votes.get)
            return self.BEATEN_BY[predicted_move]
        return None


class MarkovChainsStrategy(BaseStrategy):
    """Enhanced Markov Chains: Multi-order probability models"""
    
    def __init__(self):
        super().__init__(
            name="Markov Chains",
            description="Uses multiple-order Markov chains (1st, 2nd, 3rd order) to model transition probabilities. Each order tracks what move follows sequences of different lengths, creating multiple predictions that are combined. Different players have different memory spans - some remember just their last move, others remember 2-3 move sequences. By using multiple orders, we capture patterns at different levels of complexity.",
            short_description="Multi-order Markov models for transition patterns"
        )
        self.decay_factor = 0.95
    
    def min_history_length(self):
        return 4
    
    def predict(self, user_history, ai_history, outcome_history):
        if len(user_history) < 4:
            return None
        
        history = list(user_history)
        predictions = []
        
        for order in [1, 2, 3]:
            if len(history) < order + 1:
                continue
            
            recent = tuple(history[-order:])
            transitions = defaultdict(float)
            
            for i in range(len(history) - order):
                pattern = tuple(history[i:i+order])
                if pattern == recent:
                    next_move = history[i + order]
                    weight = self.decay_factor ** (len(history) - i - order)
                    transitions[next_move] += weight
            
            if transitions:
                predicted = max(transitions, key=transitions.get)
                confidence = sum(transitions.values())
                predictions.append((predicted, confidence, order))
        
        if predictions:
            votes = defaultdict(float)
            for move, confidence, order in predictions:
                votes[move] += confidence * order
            
            predicted_move = max(votes, key=votes.get)
            return self.BEATEN_BY[predicted_move]
        
        return None


class WinStayLoseShiftStrategy(BaseStrategy):
    """Win-Stay/Lose-Shift: Detects common human patterns"""
    
    def __init__(self):
        super().__init__(
            name="Win-Stay/Lose-Shift",
            description="Detects if the opponent follows the common human heuristic of repeating moves when they win and changing moves when they lose. This is a very common subconscious strategy for humans. After winning, people tend to stick with what worked. After losing, they try something different. By detecting this pattern, we can exploit it effectively.",
            short_description="Detects win-stay/lose-shift behavior patterns"
        )
    
    def min_history_length(self):
        return 2
    
    def predict(self, user_history, ai_history, outcome_history):
        if len(user_history) < 2 or len(outcome_history) < 1:
            return None
        
        history = list(user_history)
        outcomes = list(outcome_history)
        
        last_outcome = outcomes[-1]
        last_move = history[-1]
        
        if last_outcome == 'user':
            return self.BEATEN_BY[last_move]
        elif last_outcome == 'ai':
            shift_map = {'rock': 'paper', 'paper': 'scissor', 'scissor': 'rock'}
            predicted = shift_map.get(last_move)
            
            backward_shifts = 0
            forward_shifts = 0
            
            for i in range(len(history) - 1):
                if i < len(outcomes) and outcomes[i] == 'ai':
                    if i + 1 < len(history):
                        current = history[i]
                        next_move = history[i + 1]
                        
                        if shift_map.get(current) == next_move:
                            forward_shifts += 1
                        backward_shift_map = {'paper': 'rock', 'scissor': 'paper', 'rock': 'scissor'}
                        if backward_shift_map.get(current) == next_move:
                            backward_shifts += 1
            
            if backward_shifts > forward_shifts:
                backward_shift_map = {'paper': 'rock', 'scissor': 'paper', 'rock': 'scissor'}
                predicted = backward_shift_map.get(last_move)
            
            return self.BEATEN_BY[predicted] if predicted else None
        
        return None


class FrequencyDecayStrategy(BaseStrategy):
    """Frequency Decay: Weighted frequency analysis"""
    
    def __init__(self):
        super().__init__(
            name="Frequency Decay",
            description="Analyzes move frequencies with exponential decay, giving more weight to recent moves. This captures recent preferences and adapts to changing strategies quickly. Players often have favorite moves or recent trends. By using exponential decay, we emphasize recent behavior while still considering overall patterns, allowing quick adaptation to strategy changes.",
            short_description="Weighted frequency analysis with exponential decay"
        )
        self.decay_factor = 0.95
    
    def min_history_length(self):
        return 2
    
    def predict(self, user_history, ai_history, outcome_history):
        if len(user_history) < 2:
            return None
        
        history = list(user_history)
        freq = defaultdict(float)
        total_weight = 0
        
        for i, move in enumerate(history):
            weight = self.decay_factor ** (len(history) - i - 1)
            freq[move] += weight
            total_weight += weight
        
        if total_weight > 0:
            for move in freq:
                freq[move] /= total_weight
            
            predicted = max(freq, key=freq.get)
            return self.BEATEN_BY[predicted]
        
        return None


class RandomForestStrategy(BaseStrategy):
    """Random Forest: ML pattern recognition"""
    
    def __init__(self):
        super().__init__(
            name="Random Forest",
            description="Uses Random Forest machine learning to learn complex patterns from move history, opponent responses, and game outcomes. Features include recent moves, move frequencies, patterns, and outcome history. Random Forest can learn non-linear patterns and interactions between different features that simple heuristics miss. It adapts to the opponent's playing style by learning from all available information.",
            short_description="Machine learning model for complex pattern recognition"
        )
        self.rf_model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8)
        self.rf_fitted = False
    
    def min_history_length(self):
        return 6
    
    def _build_features(self, user_history, ai_history, outcome_history):
        X, y = [], []
        history = list(user_history)
        ai_hist = list(ai_history)
        outcomes = list(outcome_history)
        
        MOVES_TO_IDX = self.MOVES_TO_IDX
        
        for i in range(1, len(history)):
            features = []
            
            for j in range(min(6, i)):
                idx = i - 1 - j
                if idx >= 0:
                    features.append(MOVES_TO_IDX[history[idx]])
                    if idx < len(ai_hist):
                        features.append(MOVES_TO_IDX[ai_hist[idx]])
                    else:
                        features.append(0)
                    if idx < len(outcomes):
                        outcome = outcomes[idx]
                        features.append(1 if outcome == 'ai' else (-1 if outcome == 'user' else 0))
                    else:
                        features.append(0)
                else:
                    features.extend([0, 0, 0])
            
            move_counts = Counter(history[:i])
            for move in self.MOVES:
                features.append(move_counts.get(move, 0) / max(i, 1))
            
            if i >= 3:
                features.extend([MOVES_TO_IDX[m] for m in history[i-3:i]])
            else:
                features.extend([0, 0, 0])
            
            X.append(features)
            y.append(MOVES_TO_IDX[history[i]])
        
        return X, y
    
    def _get_features(self, user_history, ai_history, outcome_history):
        features = []
        history = list(user_history)
        ai_hist = list(ai_history)
        outcomes = list(outcome_history)
        MOVES_TO_IDX = self.MOVES_TO_IDX
        
        for j in range(min(6, len(history))):
            idx = len(history) - 1 - j
            if idx >= 0:
                features.append(MOVES_TO_IDX[history[idx]])
                if idx < len(ai_hist):
                    features.append(MOVES_TO_IDX[ai_hist[idx]])
                else:
                    features.append(0)
                if idx < len(outcomes):
                    outcome = outcomes[idx]
                    features.append(1 if outcome == 'ai' else (-1 if outcome == 'user' else 0))
                else:
                    features.append(0)
            else:
                features.extend([0, 0, 0])
        
        move_counts = Counter(history)
        for move in self.MOVES:
            features.append(move_counts.get(move, 0) / max(len(history), 1))
        
        if len(history) >= 3:
            features.extend([MOVES_TO_IDX[m] for m in history[-3:]])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def predict(self, user_history, ai_history, outcome_history):
        if len(user_history) < 6:
            return None
        
        try:
            X, y = self._build_features(user_history, ai_history, outcome_history)
            if len(X) < 3:
                return None
            
            X = np.array(X)
            y = np.array(y)
            
            self.rf_model.fit(X, y)
            self.rf_fitted = True
            
            last_features = self._get_features(user_history, ai_history, outcome_history).reshape(1, -1)
            predicted_idx = self.rf_model.predict(last_features)[0]
            predicted_move = self.MOVES[predicted_idx]
            
            return self.BEATEN_BY[predicted_move]
        except Exception:
            return None


class TransitionMatrixStrategy(BaseStrategy):
    """Transition Matrix: Statistical transition model"""
    
    def __init__(self):
        super().__init__(
            name="Transition Matrix",
            description="Builds probability matrices showing the likelihood of transitions from one move to another. Creates a statistical model of move sequences. By tracking exact transitions between moves, we create a probabilistic model of the opponent's behavior. This captures sequential dependencies that frequency analysis alone might miss.",
            short_description="Statistical transition probability matrices"
        )
        self.decay_factor = 0.95
    
    def min_history_length(self):
        return 4
    
    def predict(self, user_history, ai_history, outcome_history):
        if len(user_history) < 4:
            return None
        
        history = list(user_history)
        transitions = defaultdict(lambda: defaultdict(float))
        
        for i in range(len(history) - 1):
            from_move = history[i]
            to_move = history[i + 1]
            weight = self.decay_factor ** (len(history) - i - 1)
            transitions[from_move][to_move] += weight
        
        last_move = history[-1]
        if last_move in transitions and transitions[last_move]:
            predicted = max(transitions[last_move], key=transitions[last_move].get)
            return self.BEATEN_BY[predicted]
        
        return None

