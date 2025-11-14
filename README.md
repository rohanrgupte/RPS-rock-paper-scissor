# Rock Paper Scissor - AI Opponent

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

### Strategy Selection

- Uses Upper Confidence Bound (UCB) algorithm
- Only considers recent 20 transactions (configurable)
- Applies exponential decay (0.95 factor) to weight recent results more
- Tracks win percentages, wins, losses, and ties for each strategy
- Automatically switches to best-performing strategies

## The 11 Strategies

### 🎯 Simple Pattern Detection Strategies

#### 1. Always Rock Strategy
**Description:** If the opponent consistently plays rock (e.g., 80%+ of the time), this strategy counters with paper.

**How it works:** Monitors the frequency of rock moves. When rock appears in 80% or more of the opponent's moves, it assumes they will continue playing rock and counters with paper.

**Best against:** Opponents who favor a single move or are stuck in a pattern.

#### 2. Always Paper Strategy
**Description:** If the opponent consistently plays paper (e.g., 80%+ of the time), this strategy counters with scissor.

**How it works:** Similar to Always Rock, but detects and exploits paper-favoring opponents.

**Best against:** Opponents who repeatedly play paper.

#### 3. Always Scissor Strategy
**Description:** If the opponent consistently plays scissor (e.g., 80%+ of the time), this strategy counters with rock.

**How it works:** Detects scissor-favoring opponents and counters accordingly.

**Best against:** Opponents stuck on scissor.

#### 4. Cycle Detection Strategy
**Description:** Detects if the opponent follows a repeating cycle pattern (rock→paper→scissor→rock...). Once detected, it predicts the next move in the cycle and counters it.

**How it works:** Looks for repeating sequences of rock-paper-scissor. If a cycle is detected (at least 4-6 moves match the pattern), it predicts what comes next in the cycle and counters it.

**Best against:** Opponents who follow predictable sequential patterns.

#### 5. Pair Pattern Strategy
**Description:** Detects when the opponent repeats a pair of moves (e.g., rock-paper-rock-paper or scissor-rock-scissor-rock). Once identified, it predicts the next move in the pair pattern and counters it.

**How it works:** Identifies alternating two-move patterns. For example, if the pattern is rock-paper-rock-paper, it predicts rock will come next and counters with paper.

**Best against:** Opponents who alternate between two moves.

### 🚀 Advanced Strategies

#### 6. Iocaine Powder Strategy
**Description:** Checks multiple pattern lengths (1-5 moves) simultaneously and creates a voting system. Each pattern length predicts what move follows, and predictions are weighted and combined.

**How it works:** Analyzes patterns at different scales simultaneously (1 move, 2 moves, 3 moves, etc.). Each pattern length votes on what move comes next. Longer patterns get more weight as they're more reliable. Recent patterns are weighted more heavily using exponential decay.

**Why it works:** Humans have patterns at different scales. By checking all lengths simultaneously, the strategy catches patterns that single-length approaches miss. This is inspired by top-performing Kaggle solutions.

**Best against:** Opponents with complex multi-scale patterns.

#### 7. Markov Chains Strategy
**Description:** Uses multiple-order Markov chains (1st, 2nd, 3rd order) to model transition probabilities. Each order tracks what move follows sequences of different lengths.

**How it works:** 
- 1st order: What move follows the last move?
- 2nd order: What move follows the last 2 moves?
- 3rd order: What move follows the last 3 moves?

All predictions are combined with higher-order chains getting more weight. Recent transitions are weighted more heavily.

**Why it works:** Different players have different memory spans. Some remember just their last move, others remember 2-3 move sequences. By using multiple orders, the strategy captures patterns at different levels of complexity.

**Best against:** Opponents with sequential dependencies at various lengths.

#### 8. Win-Stay/Lose-Shift Strategy
**Description:** Detects if the opponent follows the common human heuristic of repeating moves when they win and changing moves when they lose.

**How it works:** 
- If opponent won last round: Predicts they'll repeat the same move
- If opponent lost last round: Predicts they'll shift (usually forward: rock→paper→scissor)

The strategy also tracks historical shift patterns (forward vs backward) to adapt to the opponent's specific behavior.

**Why it works:** This is a very common subconscious strategy for humans. After winning, people tend to stick with what worked. After losing, they try something different. By detecting this pattern, the strategy can exploit it effectively.

**Best against:** Opponents who subconsciously follow win-stay/lose-shift behavior.

#### 9. Frequency Decay Strategy
**Description:** Analyzes move frequencies with exponential decay, giving more weight to recent moves. This captures recent preferences and adapts quickly.

**How it works:** Counts move frequencies but weights recent moves exponentially more (decay factor 0.95). So a move played 10 rounds ago has much less weight than a move played 1 round ago.

**Why it works:** Players often have favorite moves or recent trends. By using exponential decay, the strategy emphasizes recent behavior while still considering overall patterns, allowing quick adaptation to strategy changes.

**Best against:** Opponents with changing preferences or recent trends.

#### 10. Random Forest Strategy
**Description:** Uses Random Forest machine learning to learn complex patterns from move history, opponent responses, and game outcomes.

**How it works:** 
- Features include: Recent moves (last 6), opponent's previous moves, outcome history, move frequencies, pattern sequences
- Trains a Random Forest model (150 trees, max depth 8)
- Learns non-linear patterns and feature interactions
- Adapts to opponent's playing style by learning from all available information

**Why it works:** Random Forest can learn complex non-linear patterns and interactions between features that simple heuristics miss. It adapts automatically to the opponent's playing style.

**Best against:** Opponents with complex, non-linear patterns that can't be captured by simple rules.

#### 11. Transition Matrix Strategy
**Description:** Builds probability matrices showing the likelihood of transitions from one move to another. Creates a statistical model of move sequences.

**How it works:** Tracks exact transitions between moves (e.g., rock→paper, paper→scissor) with exponential decay weighting recent transitions more. Predicts the next move based on the most likely transition from the last move.

**Why it works:** By tracking exact transitions, the strategy creates a probabilistic model of the opponent's behavior. This captures sequential dependencies that frequency analysis alone might miss.

**Best against:** Opponents with strong sequential dependencies.

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
- **Multi-Armed Bandit:** Automatically selects best-performing strategies
- **Recent Window Focus:** Only uses last 20 rounds for strategy selection with decay
- **Win Percentage Leaderboard:** Real-time tracking of strategy performance
- **Adaptive Learning:** Strategies adapt based on opponent behavior
- **Detailed Explanations:** Each strategy fully explained in UI and README

## Tech Stack

- **Python 3.7+**
- **Streamlit** - Web UI framework
- **scikit-learn** - Random Forest implementation
- **NumPy** - Numerical computations

## Strategy Performance

The UI displays a real-time leaderboard showing:
- Win percentage for each strategy
- Total wins, losses, and ties
- Average reward per strategy
- Currently selected strategy for each round

## How the AI Learns

1. **Exploration Phase:** Initially tries all strategies at least once
2. **Exploitation Phase:** Uses UCB algorithm to balance exploration vs exploitation
3. **Recent Focus:** Only considers last 20 rounds (configurable via `mab_window`)
4. **Decay Weighting:** Recent results weighted exponentially more than old results
5. **Adaptive Selection:** Automatically switches to best-performing strategies

## Configuration

In `rps_game.py`, you can configure:
- `history_size`: Maximum history length (default: 200)
- `mab_window`: Number of recent rounds for MAB (default: 20)
- Strategy-specific parameters in each strategy file

## Contributing

To add a new strategy:
1. Create a class in `strategies/` directory inheriting from `BaseStrategy`
2. Implement `predict()` method
3. Add to strategy list in `rps_game.py`
4. Update `strategies/__init__.py`

## License

MIT License
