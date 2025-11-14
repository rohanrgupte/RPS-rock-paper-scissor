"""
Test script for Rock Paper Scissor strategies
Tests strategies with simple scenarios
"""

from rps_game import MABRPSGame


def test_always_rock_strategy():
    """Test that Always Rock strategy is selected and returns paper when user plays only rock"""
    print("=" * 60)
    print("Test 1: Always Rock Strategy")
    print("=" * 60)
    
    game = MABRPSGame()
    
    # Play 50 rocks
    strategies_used = []
    wins = 0
    losses = 0
    
    for i in range(50):
        ai_move, winner, selected_arm = game.play_round('rock')
        strategy = game.strategies[selected_arm]
        strategies_used.append(strategy.name)
        
        if winner == 'ai':
            wins += 1
        elif winner == 'user':
            losses += 1
        
        # Check if AI played paper (which beats rock)
        if ai_move == 'paper':
            print(f"Round {i+1}: ✓ AI correctly played paper (beat rock)")
        else:
            print(f"Round {i+1}: ✗ AI played {ai_move} instead of paper")
    
    print(f"\nResults after 50 rocks:")
    print(f"AI Wins: {wins}, User Wins: {losses}, Ties: {50 - wins - losses}")
    print(f"\nStrategies used:")
    strategy_counts = {}
    for s in strategies_used:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  {strategy}: {count} times")
    
    # Check leaderboard
    leaderboard = game.get_leaderboard()
    print(f"\nTop 3 strategies by win %:")
    for i, entry in enumerate(leaderboard[:3], 1):
        print(f"  {i}. {entry['name']}: {entry['win_pct']:.1f}% ({entry['wins']}W/{entry['losses']}L)")
    
    # Expected: Always Rock strategy should be dominant
    if leaderboard[0]['name'] == 'Always Rock':
        print("\n✓ TEST PASSED: Always Rock strategy is top performer")
    else:
        print(f"\n✗ TEST FAILED: Top strategy is {leaderboard[0]['name']}, expected Always Rock")
    
    return wins, losses


def test_always_paper_strategy():
    """Test that Always Paper strategy is selected and returns scissor when user plays only paper"""
    print("\n" + "=" * 60)
    print("Test 2: Always Paper Strategy")
    print("=" * 60)
    
    game = MABRPSGame()
    
    # Play 50 papers
    strategies_used = []
    wins = 0
    losses = 0
    
    for i in range(50):
        ai_move, winner, selected_arm = game.play_round('paper')
        strategy = game.strategies[selected_arm]
        strategies_used.append(strategy.name)
        
        if winner == 'ai':
            wins += 1
        elif winner == 'user':
            losses += 1
        
        # Check if AI played scissor (which beats paper)
        if ai_move == 'scissor':
            print(f"Round {i+1}: ✓ AI correctly played scissor (beat paper)")
        else:
            print(f"Round {i+1}: ✗ AI played {ai_move} instead of scissor")
    
    print(f"\nResults after 50 papers:")
    print(f"AI Wins: {wins}, User Wins: {losses}, Ties: {50 - wins - losses}")
    print(f"\nStrategies used:")
    strategy_counts = {}
    for s in strategies_used:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  {strategy}: {count} times")
    
    leaderboard = game.get_leaderboard()
    print(f"\nTop 3 strategies by win %:")
    for i, entry in enumerate(leaderboard[:3], 1):
        print(f"  {i}. {entry['name']}: {entry['win_pct']:.1f}% ({entry['wins']}W/{entry['losses']}L)")
    
    if leaderboard[0]['name'] == 'Always Paper':
        print("\n✓ TEST PASSED: Always Paper strategy is top performer")
    else:
        print(f"\n✗ TEST FAILED: Top strategy is {leaderboard[0]['name']}, expected Always Paper")
    
    return wins, losses


def test_strategy_convergence():
    """Test that MAB converges to best strategy"""
    print("\n" + "=" * 60)
    print("Test 3: MAB Convergence")
    print("=" * 60)
    
    game = MABRPSGame()
    
    # Play 100 rocks
    for i in range(100):
        game.play_round('rock')
    
    leaderboard = game.get_leaderboard()
    
    print("Final leaderboard:")
    for i, entry in enumerate(leaderboard[:5], 1):
        print(f"  {i}. {entry['name']}: {entry['win_pct']:.1f}% "
              f"(W:{entry['wins']} L:{entry['losses']} T:{entry['ties']})")
    
    # Check if Always Rock is at top
    if leaderboard[0]['name'] == 'Always Rock' and leaderboard[0]['win_pct'] > 80:
        print("\n✓ TEST PASSED: MAB converged to Always Rock with >80% win rate")
    else:
        print(f"\n✗ TEST FAILED: MAB did not converge properly")
        print(f"   Top strategy: {leaderboard[0]['name']} with {leaderboard[0]['win_pct']:.1f}% win rate")


if __name__ == "__main__":
    print("Running strategy tests...\n")
    
    wins1, losses1 = test_always_rock_strategy()
    wins2, losses2 = test_always_paper_strategy()
    test_strategy_convergence()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Test 1 (Always Rock): AI won {wins1}/50 rounds")
    print(f"Test 2 (Always Paper): AI won {wins2}/50 rounds")
    print("\nIf tests pass, strategies are working correctly!")

