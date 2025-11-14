"""
Enhanced Rock Paper Scissor Game with Multi-Armed Bandit AI
Uses advanced strategies inspired by top Kaggle solutions:
- Iocaine Powder (multi-pattern voting)
- Enhanced Markov Chains
- Win-Stay/Lose-Shift detection
- Frequency decay analysis
- Random Forest with enhanced features
- Transition matrix predictions
"""

import random
import numpy as np
from collections import deque, Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier


class MABRPSGame:
    """
    Enhanced Rock Paper Scissor game with 6 advanced strategies
    Uses Multi-Armed Bandit (UCB) to select best strategy
    """
    
    MOVES = ['rock', 'paper', 'scissor']
    BEATS = {'rock': 'scissor', 'paper': 'rock', 'scissor': 'paper'}
    MOVES_TO_IDX = {'rock': 0, 'paper': 1, 'scissor': 2}
    
    # Strategy descriptions
    STRATEGY_DESCRIPTIONS = {
        0: "Iocaine Powder: Checks multiple pattern lengths (1-5) simultaneously and votes on prediction",
        1: "Markov Chains: Uses multi-order Markov models to predict next move based on transition patterns",
        2: "Win-Stay/Lose-Shift: Detects if opponent repeats winning moves and changes after losses",
        3: "Frequency Decay: Analyzes move frequencies with exponential decay, weighting recent moves more",
        4: "Random Forest: Machine learning model trained on move history, outcomes, and patterns",
        5: "Transition Matrix: Builds probability matrices showing likelihood of move transitions"
    }
    
    def __init__(self, history_size=200):
        self.history_size = history_size
        self.user_history = deque(maxlen=history_size)
        self.ai_history = deque(maxlen=history_size)
        self.outcome_history = deque(maxlen=history_size)  # Track wins/losses
        
        # Iocaine Powder: Pattern lengths to check
        self.pattern_lengths = [1, 2, 3, 4, 5]
        
        # Markov transition dictionaries for different orders
        self.markov_transitions = {length: defaultdict(lambda: defaultdict(int)) 
                                   for length in self.pattern_lengths}
        
        # Win-Stay/Lose-Shift tracking
        self.win_stay_pattern = None
        self.lose_shift_detected = False
        
        # Frequency decay factor (weights recent moves more heavily)
        self.decay_factor = 0.95
        
        # MAB: 6 arms = 6 different strategies
        self.num_arms = 6
        self.arm_counts = np.zeros(self.num_arms)
        self.arm_rewards = np.zeros(self.num_arms)
        self.arm_values = np.zeros(self.num_arms)
        self.total_plays = 0
        
        # Random Forest for pattern recognition
        self.rf_model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8)
        self.rf_fitted = False
        
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
            return 0.1  # Small positive reward for ties
        else:
            return -1.0
    
    def update_scores(self, winner):
        """Update game scores"""
        if winner == 'user':
            self.user_score += 1
        elif winner == 'ai':
            self.ai_score += 1
    
    # Strategy 0: Iocaine Powder
    def predict_iocaine_powder(self):
        """
        Iocaine Powder Strategy (inspired by top Kaggle solutions)
        
        Explanation: This strategy checks multiple pattern lengths (1-5 moves) simultaneously
        and creates a voting system. Each pattern length predicts what move follows, and
        the predictions are weighted and combined to make a final decision.
        
        Why it works: Humans often have patterns at different scales - they might repeat
        the last move, or follow 2-3 move sequences, or even longer patterns. By checking
        all lengths simultaneously, we catch patterns that single-length approaches miss.
        """
        if len(self.user_history) < 3:
            return random.choice(self.MOVES)
        
        votes = defaultdict(float)
        history = list(self.user_history)
        
        for pattern_len in self.pattern_lengths:
            if len(history) < pattern_len + 1:
                continue
            
            # Get recent pattern
            recent_pattern = tuple(history[-pattern_len:])
            
            # Count what follows this pattern in history
            next_moves = defaultdict(float)
            for i in range(len(history) - pattern_len):
                pattern = tuple(history[i:i+pattern_len])
                if pattern == recent_pattern:
                    next_move = history[i + pattern_len]
                    # Weight by recency: more recent patterns matter more
                    weight = self.decay_factor ** (len(history) - i - pattern_len)
                    next_moves[next_move] += weight
            
            if next_moves:
                # Find most likely next move for this pattern length
                predicted = max(next_moves, key=next_moves.get)
                # Vote with weight based on pattern length and confidence
                # Longer patterns are often more reliable but less frequent
                pattern_weight = next_moves[predicted] * (pattern_len / max(self.pattern_lengths))
                votes[predicted] += pattern_weight
        
        if sum(votes.values()) > 0:
            predicted_move = max(votes, key=votes.get)
            return self.BEATS[predicted_move]
        return random.choice(self.MOVES)
    
    # Strategy 1: Enhanced Markov Chains
    def predict_markov_enhanced(self):
        """
        Enhanced Markov Chain Strategy
        
        Explanation: Uses multiple-order Markov chains (1st, 2nd, 3rd order) to model
        transition probabilities. Each order tracks what move follows sequences of
        different lengths, creating multiple predictions that are combined.
        
        Why it works: Different players have different memory spans. Some remember
        just their last move, others remember 2-3 move sequences. By using multiple
        orders, we capture patterns at different levels of complexity.
        """
        if len(self.user_history) < 4:
            return random.choice(self.MOVES)
        
        history = list(self.user_history)
        predictions = []
        
        # Check multiple Markov orders
        for order in [1, 2, 3]:
            if len(history) < order + 1:
                continue
            
            recent = tuple(history[-order:])
            transitions = defaultdict(float)
            
            # Find all instances of this pattern and what followed
            for i in range(len(history) - order):
                pattern = tuple(history[i:i+order])
                if pattern == recent:
                    next_move = history[i + order]
                    # Weight by recency
                    weight = self.decay_factor ** (len(history) - i - order)
                    transitions[next_move] += weight
            
            if transitions:
                predicted = max(transitions, key=transitions.get)
                confidence = sum(transitions.values())
                predictions.append((predicted, confidence, order))
        
        if predictions:
            # Weighted vote by confidence and order (higher order = more reliable)
            votes = defaultdict(float)
            for move, confidence, order in predictions:
                votes[move] += confidence * order  # Higher order gets more weight
            
            predicted_move = max(votes, key=votes.get)
            return self.BEATS[predicted_move]
        
        return random.choice(self.MOVES)
    
    # Strategy 2: Win-Stay/Lose-Shift
    def predict_win_stay_lose_shift(self):
        """
        Win-Stay/Lose-Shift Detection Strategy
        
        Explanation: Detects if the opponent follows the common human heuristic of
        repeating moves when they win and changing moves when they lose.
        
        Why it works: This is a very common subconscious strategy for humans. After
        winning, people tend to stick with what worked. After losing, they try something
        different. By detecting this pattern, we can exploit it effectively.
        """
        if len(self.user_history) < 2 or len(self.outcome_history) < 1:
            return random.choice(self.MOVES)
        
        history = list(self.user_history)
        outcomes = list(self.outcome_history)
        
        # Check last outcome
        last_outcome = outcomes[-1]
        last_move = history[-1]
        
        if last_outcome == 'user':
            # Opponent won - they might stay with the same move
            return self.BEATS[last_move]
        elif last_outcome == 'ai':
            # Opponent lost - they might shift
            # Common shift patterns: rock->paper, paper->scissor, scissor->rock
            shift_map = {'rock': 'paper', 'paper': 'scissor', 'scissor': 'rock'}
            predicted = shift_map.get(last_move, random.choice(self.MOVES))
            
            # But also check: do they cycle backwards sometimes?
            # Count shift patterns in history
            backward_shifts = 0
            forward_shifts = 0
            
            for i in range(len(history) - 1):
                if i < len(outcomes) and outcomes[i] == 'ai':  # After loss
                    if i + 1 < len(history):
                        current = history[i]
                        next_move = history[i + 1]
                        
                        # Forward shift
                        if shift_map.get(current) == next_move:
                            forward_shifts += 1
                        # Backward shift (less common)
                        backward_shift_map = {'paper': 'rock', 'scissor': 'paper', 'rock': 'scissor'}
                        if backward_shift_map.get(current) == next_move:
                            backward_shifts += 1
            
            # Choose based on historical pattern
            if backward_shifts > forward_shifts:
                backward_shift_map = {'paper': 'rock', 'scissor': 'paper', 'rock': 'scissor'}
                predicted = backward_shift_map.get(last_move, random.choice(self.MOVES))
            
            return self.BEATS[predicted]
        
        return random.choice(self.MOVES)
    
    # Strategy 3: Frequency Decay
    def predict_frequency_decay(self):
        """
        Frequency Decay Analysis Strategy
        
        Explanation: Analyzes move frequencies with exponential decay, giving more
        weight to recent moves. This captures recent preferences and adapts to
        changing strategies quickly.
        
        Why it works: Players often have favorite moves or recent trends. By using
        exponential decay, we emphasize recent behavior while still considering
        overall patterns, allowing quick adaptation to strategy changes.
        """
        if len(self.user_history) < 2:
            return random.choice(self.MOVES)
        
        history = list(self.user_history)
        
        # Calculate weighted frequencies with exponential decay
        freq = defaultdict(float)
        total_weight = 0
        
        for i, move in enumerate(history):
            # More recent moves get exponentially more weight
            weight = self.decay_factor ** (len(history) - i - 1)
            freq[move] += weight
            total_weight += weight
        
        if total_weight > 0:
            # Normalize to get probabilities
            for move in freq:
                freq[move] /= total_weight
            
            # Predict most frequent move (weighted by recency)
            predicted = max(freq, key=freq.get)
            return self.BEATS[predicted]
        
        return random.choice(self.MOVES)
    
    # Strategy 4: Enhanced Random Forest
    def predict_rf_enhanced(self):
        """
        Enhanced Random Forest Strategy
        
        Explanation: Uses Random Forest machine learning to learn complex patterns
        from move history, opponent responses, and game outcomes. Features include
        recent moves, move frequencies, patterns, and outcome history.
        
        Why it works: Random Forest can learn non-linear patterns and interactions
        between different features that simple heuristics miss. It adapts to the
        opponent's playing style by learning from all available information.
        """
        if len(self.user_history) < 6:
            return random.choice(self.MOVES)
        
        try:
            X, y = self._build_enhanced_features()
            if len(X) < 3:
                return random.choice(self.MOVES)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train model
            self.rf_model.fit(X, y)
            self.rf_fitted = True
            
            # Predict next move
            last_features = self._get_enhanced_features().reshape(1, -1)
            predicted_idx = self.rf_model.predict(last_features)[0]
            predicted_move = self.MOVES[predicted_idx]
            
            return self.BEATS[predicted_move]
        except Exception:
            return random.choice(self.MOVES)
    
    # Strategy 5: Transition Matrix
    def predict_transition_matrix(self):
        """
        Transition Matrix Strategy
        
        Explanation: Builds probability matrices showing the likelihood of transitions
        from one move to another. Creates a statistical model of move sequences.
        
        Why it works: By tracking exact transitions between moves, we create a
        probabilistic model of the opponent's behavior. This captures sequential
        dependencies that frequency analysis alone might miss.
        """
        if len(self.user_history) < 4:
            return random.choice(self.MOVES)
        
        history = list(self.user_history)
        
        # Build transition probability matrix
        transitions = defaultdict(lambda: defaultdict(float))
        total_transitions = 0
        
        # Count transitions with decay weighting
        for i in range(len(history) - 1):
            from_move = history[i]
            to_move = history[i + 1]
            weight = self.decay_factor ** (len(history) - i - 1)
            transitions[from_move][to_move] += weight
            total_transitions += weight
        
        # Predict based on last move
        last_move = history[-1]
        if last_move in transitions and transitions[last_move]:
            # Find most likely next move given last move
            predicted = max(transitions[last_move], key=transitions[last_move].get)
            return self.BEATS[predicted]
        
        return random.choice(self.MOVES)
    
    def _build_enhanced_features(self):
        """Build enhanced features for Random Forest"""
        X, y = [], []
        history = list(self.user_history)
        ai_history = list(self.ai_history)
        outcomes = list(self.outcome_history)
        
        for i in range(1, len(history)):
            features = []
            
            # Last 6 moves (18 features: move, ai_move, outcome for each)
            for j in range(min(6, i)):
                idx = i - 1 - j
                if idx >= 0:
                    features.append(self.MOVES_TO_IDX[history[idx]])
                    if idx < len(ai_history):
                        features.append(self.MOVES_TO_IDX[ai_history[idx]])
                    else:
                        features.append(0)
                    if idx < len(outcomes):
                        # Encode outcome: -1=loss, 0=tie, 1=win (from AI perspective)
                        outcome = outcomes[idx]
                        features.append(1 if outcome == 'ai' else (-1 if outcome == 'user' else 0))
                    else:
                        features.append(0)
                else:
                    features.extend([0, 0, 0])
            
            # Frequency features (3 features: proportion of each move)
            move_counts = Counter(history[:i])
            for move in self.MOVES:
                features.append(move_counts.get(move, 0) / max(i, 1))
            
            # Last 3-move pattern (3 features)
            if i >= 3:
                features.extend([self.MOVES_TO_IDX[m] for m in history[i-3:i]])
            else:
                features.extend([0, 0, 0])
            
            X.append(features)
            y.append(self.MOVES_TO_IDX[history[i]])
        
        return X, y
    
    def _get_enhanced_features(self):
        """Get features for current state"""
        features = []
        history = list(self.user_history)
        ai_history = list(self.ai_history)
        outcomes = list(self.outcome_history)
        
        # Last 6 moves
        for j in range(min(6, len(history))):
            idx = len(history) - 1 - j
            if idx >= 0:
                features.append(self.MOVES_TO_IDX[history[idx]])
                if idx < len(ai_history):
                    features.append(self.MOVES_TO_IDX[ai_history[idx]])
                else:
                    features.append(0)
                if idx < len(outcomes):
                    outcome = outcomes[idx]
                    features.append(1 if outcome == 'ai' else (-1 if outcome == 'user' else 0))
                else:
                    features.append(0)
            else:
                features.extend([0, 0, 0])
        
        # Frequency features
        move_counts = Counter(history)
        for move in self.MOVES:
            features.append(move_counts.get(move, 0) / max(len(history), 1))
        
        # Last 3-move pattern
        if len(history) >= 3:
            features.extend([self.MOVES_TO_IDX[m] for m in history[-3:]])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
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
        strategies = [
            self.predict_iocaine_powder,
            self.predict_markov_enhanced,
            self.predict_win_stay_lose_shift,
            self.predict_frequency_decay,
            self.predict_rf_enhanced,
            self.predict_transition_matrix
        ]
        
        if selected_arm < len(strategies):
            move = strategies[selected_arm]()
            return move, selected_arm
        else:
            return random.choice(self.MOVES), selected_arm
    
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
        
        # Update Markov transitions
        history = list(self.user_history)
        for length in self.pattern_lengths:
            if len(history) >= length + 1:
                pattern = tuple(history[-(length+1):-1])
                next_move = history[-1]
                self.markov_transitions[length][pattern][next_move] += 1
        
        # Update scores
        self.update_scores(winner)
        
        return ai_move, winner, selected_arm
    
    def reset_game(self):
        """Reset game state"""
        self.user_history.clear()
        self.ai_history.clear()
        self.outcome_history.clear()
        self.markov_transitions = {length: defaultdict(lambda: defaultdict(int)) 
                                   for length in self.pattern_lengths}
        self.arm_counts = np.zeros(self.num_arms)
        self.arm_rewards = np.zeros(self.num_arms)
        self.arm_values = np.zeros(self.num_arms)
        self.total_plays = 0
        self.rf_fitted = False
        self.win_stay_pattern = None
        self.lose_shift_detected = False
        self.user_score = 0
        self.ai_score = 0
