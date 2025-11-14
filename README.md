# Rock Paper Scissor Dojo

Advanced AI-powered Rock Paper Scissor game using Multi-Armed Bandit with 11 different strategies (5 simple + 6 advanced). The AI learns and adapts by selecting the best-performing strategies based on recent performance.

## Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

The AI uses a **Multi-Armed Bandit (UCB)** algorithm to dynamically select the best strategy. Only the most recent 20 rounds are considered for strategy selection, with exponential decay giving more weight to recent results. This ensures quick adaptation to changing opponent patterns.

### Multi-Armed Bandit (UCB Algorithm)

The Upper Confidence Bound (UCB) algorithm balances exploration and exploitation:

**UCB Formula:**
```
UCB_i = μ_i + c × √(ln(n) / N_i)
```

Where:
- **μ_i** = Average reward for strategy i (with exponential decay)
- **c** = Exploration factor (default: √2 ≈ 1.414, adjustable in UI)
- **n** = Total number of rounds played
- **N_i** = Number of times strategy i has been used

**How it works:**
- **Exploration:** Higher c or lower N_i increases UCB value, encouraging trying less-used strategies
- **Exploitation:** Higher μ_i increases UCB value, encouraging use of successful strategies
- **Recent Focus:** Only last 20 rounds considered with exponential decay (0.95 factor)

### Strategy Selection

- Uses Upper Confidence Bound (UCB) algorithm
- Only considers recent 20 transactions (configurable)
- Applies exponential decay (0.95 factor) to weight recent results more
- Tracks win percentages, wins, losses, and ties for each strategy
- Automatically switches to best-performing strategies
- **Exploration factor (c) adjustable in UI** - control exploration vs exploitation balance

## The 11 Strategies

### 🎯 Simple Pattern Detection Strategies

#### 1. Always Rock Strategy
**Description:** Always predicts the opponent will play rock and counters with paper.

**How it works:** 
- Returns `BEATEN_BY['rock'] = 'paper'` on every round
- No history needed - always works

**Best against:** Opponents who favor rock.

#### 2. Always Paper Strategy
**Description:** Always predicts the opponent will play paper and counters with scissor.

**How it works:**
- Returns `BEATEN_BY['paper'] = 'scissor'` on every round
- No history needed - always works

**Best against:** Opponents who favor paper.

#### 3. Always Scissor Strategy
**Description:** Always predicts the opponent will play scissor and counters with rock.

**How it works:**
- Returns `BEATEN_BY['scissor'] = 'rock'` on every round
- No history needed - always works

**Best against:** Opponents who favor scissor.

#### 4. Cycle Detection Strategy
**Description:** Detects if the opponent follows a repeating cycle pattern (rock→paper→scissor→rock...). Once detected, it predicts the next move in the cycle and counters it.

**How it works:** 
- Checks for repeating sequences of rock-paper-scissor
- If a cycle is detected (at least 4-6 moves match the pattern), predicts next move in cycle
- Uses pattern matching: `pattern[i:i+pattern_len] == recent_pattern`

**Mathematical approach:**
```
For pattern length L:
  If history[-L:] matches pattern seen earlier:
    predicted_move = cycle[(position + L) % 3]
    return BEATEN_BY[predicted_move]
```

**Best against:** Opponents who follow predictable sequential patterns.

#### 5. Pair Pattern Strategy
**Description:** Detects when the opponent repeats a pair of moves (e.g., rock-paper-rock-paper or scissor-rock-scissor-rock). Once identified, it predicts the next move in the pair pattern and counters it.

**How it works:**
- Identifies alternating two-move patterns
- Checks for A-B-A-B patterns in recent history
- Predicts next move based on pattern repetition

**Best against:** Opponents who alternate between two moves.

### 🚀 Advanced Strategies

#### 6. Iocaine Powder Strategy
**Description:** Checks multiple pattern lengths (1-5 moves) simultaneously and creates a voting system. Each pattern length predicts what move follows, and predictions are weighted and combined.

**How it works:** 
- Analyzes patterns at different scales simultaneously (1, 2, 3, 4, 5 moves)
- Each pattern length votes on what move comes next
- Longer patterns get more weight as they're more reliable
- Recent patterns weighted more heavily using exponential decay

**Mathematical approach:**
```
For each pattern length L in [1, 2, 3, 4, 5]:
  recent_pattern = history[-L:]
  votes = {}
  For each occurrence of recent_pattern in history:
    next_move = history[occurrence_index + L]
    weight = decay_factor ^ (history_length - occurrence_index - L)
    votes[next_move] += weight × (L / max_pattern_length)
  
predicted_move = argmax(votes)
return BEATEN_BY[predicted_move]
```

**Why it works:** Humans have patterns at different scales. By checking all lengths simultaneously, the strategy catches patterns that single-length approaches miss.

**Credits:** Inspired by top-performing Kaggle solutions, particularly the Iocaine Powder approach from the Rock Paper Scissors competition.

**Best against:** Opponents with complex multi-scale patterns.

#### 7. Markov Chains Strategy
**Description:** Uses multiple-order Markov chains (1st, 2nd, 3rd order) to model transition probabilities. Each order tracks what move follows sequences of different lengths.

**How it works:** 
- **1st order:** What move follows the last move?
- **2nd order:** What move follows the last 2 moves?
- **3rd order:** What move follows the last 3 moves?

All predictions are combined with higher-order chains getting more weight. Recent transitions are weighted more heavily.

**Mathematical approach:**
```
For each order O in [1, 2, 3]:
  recent = history[-O:]
  transitions = {}
  For each occurrence of recent in history:
    next_move = history[occurrence_index + O]
    weight = decay_factor ^ (history_length - occurrence_index - O)
    transitions[next_move] += weight
  
  predicted = argmax(transitions)
  confidence = sum(transitions.values())
  predictions.append((predicted, confidence, O))

# Weighted vote: higher order gets more weight
votes[predicted] += confidence × order
predicted_move = argmax(votes)
return BEATEN_BY[predicted_move]
```

**Why it works:** Different players have different memory spans. Some remember just their last move, others remember 2-3 move sequences. By using multiple orders, the strategy captures patterns at different levels of complexity.

**Credits:** Based on Markov Chain approaches from Kaggle Rock Paper Scissors competition solutions.

**Best against:** Opponents with sequential dependencies at various lengths.

#### 8. Win-Stay/Lose-Shift Strategy
**Description:** Detects if the opponent follows the common human heuristic of repeating moves when they win and changing moves when they lose.

**How it works:** 
- If opponent won last round: Predicts they'll repeat the same move → Counter with `BEATEN_BY[last_move]`
- If opponent lost last round: Predicts they'll shift (usually forward: rock→paper→scissor)

The strategy also tracks historical shift patterns (forward vs backward) to adapt to the opponent's specific behavior.

**Mathematical approach:**
```
If last_outcome == 'user':
  predicted_move = last_move  # Win-stay
  return BEATEN_BY[predicted_move]

If last_outcome == 'ai':
  # Count forward vs backward shifts in history
  forward_shifts = count(rock→paper, paper→scissor, scissor→rock)
  backward_shifts = count(paper→rock, scissor→paper, rock→scissor)
  
  If forward_shifts > backward_shifts:
    predicted_move = shift_forward(last_move)
  Else:
    predicted_move = shift_backward(last_move)
  
  return BEATEN_BY[predicted_move]
```

**Why it works:** This is a very common subconscious strategy for humans. After winning, people tend to stick with what worked. After losing, they try something different.

**Credits:** Based on behavioral psychology research and common patterns observed in human RPS play.

**Best against:** Opponents who subconsciously follow win-stay/lose-shift behavior.

#### 9. Frequency Decay Strategy
**Description:** Analyzes move frequencies with exponential decay, giving more weight to recent moves. This captures recent preferences and adapts quickly.

**How it works:** 
Counts move frequencies but weights recent moves exponentially more (decay factor 0.95).

**Mathematical approach:**
```
freq = {}
total_weight = 0

For each move in history (from oldest to newest):
  age = history_length - move_index - 1
  weight = decay_factor ^ age  # decay_factor = 0.95
  freq[move] += weight
  total_weight += weight

# Normalize
For each move:
  freq[move] /= total_weight

predicted_move = argmax(freq)
return BEATEN_BY[predicted_move]
```

**Why it works:** Players often have favorite moves or recent trends. By using exponential decay, the strategy emphasizes recent behavior while still considering overall patterns, allowing quick adaptation to strategy changes.

**Credits:** Inspired by frequency-based approaches with temporal decay from Kaggle solutions.

**Best against:** Opponents with changing preferences or recent trends.

#### 10. Random Forest Strategy
**Description:** Uses Random Forest machine learning to learn complex patterns from move history, opponent responses, and game outcomes.

**How it works:** 
- **Features include:** Recent moves (last 6), opponent's previous moves, outcome history, move frequencies, pattern sequences
- Trains a Random Forest model (150 trees, max depth 8)
- Learns non-linear patterns and feature interactions
- Adapts to opponent's playing style by learning from all available information

**Mathematical approach:**
```
Features for prediction:
  X = [
    last_6_moves (18 features: move, ai_move, outcome for each),
    move_frequencies (3 features: proportion of each move),
    last_3_move_pattern (3 features)
  ]
  Total: 24 features

Training:
  For each historical move i:
    X_i = extract_features(history[:i])
    y_i = history[i]
    RF.fit(X, y)

Prediction:
  X_current = extract_features(current_history)
  predicted_move = RF.predict(X_current)
  return BEATEN_BY[predicted_move]
```

**Why it works:** Random Forest can learn complex non-linear patterns and interactions between features that simple heuristics miss. It adapts automatically to the opponent's playing style.

**Credits:** Based on Random Forest approaches from Kaggle Rock Paper Scissors competition, particularly solutions using ensemble methods.

**Best against:** Opponents with complex, non-linear patterns that can't be captured by simple rules.

#### 11. Transition Matrix Strategy
**Description:** Builds probability matrices showing the likelihood of transitions from one move to another. Creates a statistical model of move sequences.

**How it works:** 
Tracks exact transitions between moves (e.g., rock→paper, paper→scissor) with exponential decay weighting recent transitions more. Predicts the next move based on the most likely transition from the last move.

**Mathematical approach:**
```
# Build transition matrix with decay
transitions[from_move][to_move] = 0

For each transition in history:
  from_move = history[i]
  to_move = history[i+1]
  age = history_length - i - 1
  weight = decay_factor ^ age
  transitions[from_move][to_move] += weight

# Predict based on last move
last_move = history[-1]
predicted_move = argmax(transitions[last_move])
return BEATEN_BY[predicted_move]
```

**Why it works:** By tracking exact transitions, the strategy creates a probabilistic model of the opponent's behavior. This captures sequential dependencies that frequency analysis alone might miss.

**Credits:** Based on transition matrix and statistical modeling approaches from Kaggle solutions.

**Best against:** Opponents with strong sequential dependencies.

## Credits & Acknowledgments

This project incorporates strategies and ideas inspired by top-performing solutions from the [Kaggle Rock Paper Scissors competition](https://www.kaggle.com/competitions/rock-paper-scissors):

- **Iocaine Powder Strategy:** Inspired by the classic Iocaine Powder approach, adapted for this implementation
- **Markov Chains & Transition Matrices:** Based on probabilistic modeling approaches from various Kaggle solutions
- **Random Forest & ML Approaches:** Inspired by ensemble methods and machine learning solutions
- **Frequency Decay & Pattern Detection:** Based on temporal pattern recognition approaches

**Key Kaggle Notebooks Referenced:**
- [alincijov/rock-paper-scissors-genetic-algorithm](https://www.kaggle.com/code/alincijov/rock-paper-scissors-genetic-algorithm) - Genetic algorithm approaches
- [georg-streich-2nd-place-solution](https://www.kaggle.com/competitions/rock-paper-scissors/writeups/georg-streich-2nd-place-solution) - Advanced strategy combinations
- [superant/rps-geometry-silver-rank-by-minimal-logic](https://www.kaggle.com/code/superant/rps-geometry-silver-rank-by-minimal-logic) - Minimal logic and geometric approaches

**Note:** This implementation adapts and combines ideas from these sources into a unified Multi-Armed Bandit framework with modular strategy architecture.

## Project Structure

```
├── app.py                  # Streamlit UI application
├── rps_game.py            # Main game logic with MAB
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── strategies/           # Strategy modules
    ├── __init__.py
    ├── base_strategy.py  # Base strategy class
    ├── simple_patterns.py    # Simple pattern detection strategies
    └── advanced_strategies.py # Advanced ML and statistical strategies
```

## Key Features

- **11 Different Strategies:** 5 simple pattern detectors + 6 advanced ML/statistical strategies
- **Multi-Armed Bandit:** Automatically selects best-performing strategies using UCB algorithm
- **Recent Window Focus:** Only uses last 20 rounds for strategy selection with decay
- **Win Percentage Leaderboard:** Real-time tracking of strategy performance
- **Adaptive Learning:** Strategies adapt based on opponent behavior
- **Detailed Explanations:** Each strategy fully explained in UI and README with equations
- **Configurable Exploration:** Adjust exploration factor (c) in UI to control exploration vs exploitation

## Tech Stack

- **Python 3.7+**
- **Streamlit** - Web UI framework
- **scikit-learn** - Random Forest implementation
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation for leaderboard

## Strategy Performance

The UI displays a real-time leaderboard showing:
- Win percentage for each strategy
- Total wins, losses, and ties
- Average reward per strategy
- Currently selected strategy for each round

**All strategies are evaluated every round** - even strategies not selected get their hypothetical performance tracked and displayed.

## How the AI Learns

1. **Exploration Phase:** Initially tries all strategies at least once
2. **Exploitation Phase:** Uses UCB algorithm to balance exploration vs exploitation
3. **Recent Focus:** Only considers last 20 rounds (configurable via `mab_window`)
4. **Decay Weighting:** Recent results weighted exponentially more than old results
5. **Adaptive Selection:** Automatically switches to best-performing strategies
6. **All Strategies Evaluated:** Every round, all strategies make predictions and their hypothetical results are tracked

## Configuration

In `rps_game.py`, you can configure:
- `history_size`: Maximum history length (default: 200)
- `mab_window`: Number of recent rounds for MAB (default: 20)
- `exploration_c`: Exploration factor for UCB (default: √2 ≈ 1.414, adjustable in UI)
- Strategy-specific parameters in each strategy file

## Contributing

To add a new strategy:
1. Create a class in `strategies/` directory inheriting from `BaseStrategy`
2. Implement `predict()` method
3. Add to strategy list in `rps_game.py`
4. Update `strategies/__init__.py`

## License

MIT License
