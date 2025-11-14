# Rock Paper Scissor Dojo

AI-powered Rock Paper Scissor game using Multi-Armed Bandit with 11 different strategies. The AI learns and adapts by selecting the best-performing strategies based on recent performance.

## Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rps-ai-playground.streamlit.app/)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

The AI uses a Multi-Armed Bandit (UCB) algorithm to dynamically select the best strategy. Only the most recent 20 rounds are considered for strategy selection, with exponential decay giving more weight to recent results.

### UCB Algorithm

The Upper Confidence Bound algorithm balances exploration and exploitation:

```
UCB_i = μ_i + c × √(ln(n) / N_i)
```

Where:
- μ_i = Average reward for strategy i (with exponential decay)
- c = Exploration factor (default: √2 ≈ 1.414, adjustable in UI)
- n = Total number of rounds played
- N_i = Number of times strategy i has been used

Higher c or lower N_i increases exploration. Higher μ_i increases exploitation. Only the last 20 rounds are considered with exponential decay (0.95 factor).

## The 11 Strategies

### Simple Pattern Detection Strategies

#### 1. Always Rock Strategy
Always predicts the opponent will play rock and counters with paper. No history needed.

#### 2. Always Paper Strategy
Always predicts the opponent will play paper and counters with scissor. No history needed.

#### 3. Always Scissor Strategy
Always predicts the opponent will play scissor and counters with rock. No history needed.

#### 4. Cycle Detection Strategy
Detects if the opponent follows a repeating cycle pattern (rock→paper→scissor→rock...). Once detected, predicts the next move in the cycle and counters it.

```
For pattern length L:
  If history[-L:] matches pattern seen earlier:
    predicted_move = cycle[(position + L) % 3]
    return BEATEN_BY[predicted_move]
```

#### 5. Pair Pattern Strategy
Detects when the opponent repeats a pair of moves (e.g., rock-paper-rock-paper). Once identified, predicts the next move in the pair pattern and counters it.

### Advanced Strategies

#### 6. Iocaine Powder Strategy
Checks multiple pattern lengths (1-5 moves) simultaneously and creates a voting system. Each pattern length predicts what move follows, and predictions are weighted and combined.

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

Credits: Inspired by Iocaine Powder approach from Kaggle Rock Paper Scissors competition.

#### 7. Markov Chains Strategy
Uses multiple-order Markov chains (1st, 2nd, 3rd order) to model transition probabilities. Each order tracks what move follows sequences of different lengths.

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

votes[predicted] += confidence × order
predicted_move = argmax(votes)
return BEATEN_BY[predicted_move]
```

Credits: Based on Markov Chain approaches from Kaggle Rock Paper Scissors competition solutions.

#### 8. Win-Stay/Lose-Shift Strategy
Detects if the opponent follows the common human heuristic of repeating moves when they win and changing moves when they lose.

```
If last_outcome == 'user':
  predicted_move = last_move  # Win-stay
  return BEATEN_BY[predicted_move]

If last_outcome == 'ai':
  forward_shifts = count(rock→paper, paper→scissor, scissor→rock)
  backward_shifts = count(paper→rock, scissor→paper, rock→scissor)
  
  If forward_shifts > backward_shifts:
    predicted_move = shift_forward(last_move)
  Else:
    predicted_move = shift_backward(last_move)
  
  return BEATEN_BY[predicted_move]
```

#### 9. Frequency Decay Strategy
Analyzes move frequencies with exponential decay, giving more weight to recent moves.

```
freq = {}
total_weight = 0

For each move in history (from oldest to newest):
  age = history_length - move_index - 1
  weight = decay_factor ^ age  # decay_factor = 0.95
  freq[move] += weight
  total_weight += weight

For each move:
  freq[move] /= total_weight

predicted_move = argmax(freq)
return BEATEN_BY[predicted_move]
```

#### 10. Random Forest Strategy
Uses Random Forest machine learning to learn complex patterns from move history, opponent responses, and game outcomes.

Features include: Recent moves (last 6), opponent's previous moves, outcome history, move frequencies, pattern sequences. Trains a Random Forest model (150 trees, max depth 8).

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

#### 11. Transition Matrix Strategy
Builds probability matrices showing the likelihood of transitions from one move to another.

```
transitions[from_move][to_move] = 0

For each transition in history:
  from_move = history[i]
  to_move = history[i+1]
  age = history_length - i - 1
  weight = decay_factor ^ age
  transitions[from_move][to_move] += weight

last_move = history[-1]
predicted_move = argmax(transitions[last_move])
return BEATEN_BY[predicted_move]
```

## Credits

This project incorporates strategies inspired by top-performing solutions from the [Kaggle Rock Paper Scissors competition](https://www.kaggle.com/competitions/rock-paper-scissors):

- Iocaine Powder Strategy: Inspired by the classic Iocaine Powder approach
- Markov Chains Strategy: Based on probabilistic modeling approaches from Kaggle solutions
- Key notebooks referenced:
  - [alincijov/rock-paper-scissors-genetic-algorithm](https://www.kaggle.com/code/alincijov/rock-paper-scissors-genetic-algorithm)
  - [georg-streich-2nd-place-solution](https://www.kaggle.com/competitions/rock-paper-scissors/writeups/georg-streich-2nd-place-solution)
  - [superant/rps-geometry-silver-rank-by-minimal-logic](https://www.kaggle.com/code/superant/rps-geometry-silver-rank-by-minimal-logic)

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

## Configuration

In `rps_game.py`, you can configure:
- `history_size`: Maximum history length (default: 200)
- `mab_window`: Number of recent rounds for MAB (default: 20)
- `exploration_c`: Exploration factor for UCB (default: √2 ≈ 1.414, adjustable in UI)

## License

MIT License
