"""
Streamlit UI for Rock Paper Scissor Game
Enhanced with 11 AI strategies (5 simple + 6 advanced)
"""

import streamlit as st
import numpy as np
import pandas as pd
from rps_game import MABRPSGame

# Page config
st.set_page_config(
    page_title="Rock Paper Scissor",
    page_icon="✂️",
    layout="wide"
)

# Initialize game
if 'game' not in st.session_state:
    st.session_state.game = MABRPSGame()

# Header
st.title("✂️ Rock Paper Scissor")
st.markdown("**AI-Powered Opponent using 11 Strategies (5 Simple + 6 Advanced) with Multi-Armed Bandit**")

# Scoreboard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Your Score", st.session_state.game.user_score)
with col2:
    st.metric("AI Score", st.session_state.game.ai_score)
with col3:
    total = st.session_state.game.user_score + st.session_state.game.ai_score
    st.metric("Rounds", total)
with col4:
    # Show current best strategy
    if st.session_state.game.total_plays > 0:
        leaderboard = st.session_state.game.get_leaderboard()
        if leaderboard:
            best = leaderboard[0]
            best_strategy = st.session_state.game.strategies[best['arm']]
            st.metric("Best Strategy", f"{best_strategy.name} ({best['win_pct']:.1f}%)")
        else:
            st.metric("Best Strategy", "Exploring")
    else:
        st.metric("Best Strategy", "Exploring")

st.divider()

# Main game area
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎮 Make Your Move")
    
    move_cols = st.columns(3)
    with move_cols[0]:
        rock = st.button("🪨 Rock", use_container_width=True, type="primary")
    with move_cols[1]:
        paper = st.button("📄 Paper", use_container_width=True, type="primary")
    with move_cols[2]:
        scissor = st.button("✂️ Scissor", use_container_width=True, type="primary")
    
    # Process move
    selected_arm = None
    ai_move = None
    winner = None
    predicted_user_move = None
    
    if rock or paper or scissor:
        user_move = 'rock' if rock else ('paper' if paper else 'scissor')
        ai_move, winner, selected_arm = st.session_state.game.play_round(user_move)
        predicted_user_move = [k for k, v in st.session_state.game.BEATS.items() if v == ai_move][0]
    
    # Display results if move was made
    if ai_move:
        st.divider()
        st.markdown("### 📊 Round Result")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.markdown("#### Your Move")
            emoji = "🪨" if user_move == "rock" else ("📄" if user_move == "paper" else "✂️")
            st.markdown(f"## {emoji}")
            st.markdown(f"**{user_move.title()}**")
        
        with result_col2:
            st.markdown("#### AI Move")
            emoji = "🪨" if ai_move == "rock" else ("📄" if ai_move == "paper" else "✂️")
            st.markdown(f"## {emoji}")
            st.markdown(f"**{ai_move.title()}**")
            
            # Show selected strategy VERY prominently
            if selected_arm is not None and selected_arm < len(st.session_state.game.strategies):
                strategy = st.session_state.game.strategies[selected_arm]
                st.markdown("---")
                st.markdown("### 🎯 Selected Strategy")
                st.success(f"**{strategy.name}**")
                st.caption(f"*{strategy.short_description}*")
        
        with result_col3:
            st.markdown("#### Result")
            if winner == 'user':
                st.success("## 🎉 You Win!")
            elif winner == 'ai':
                st.error("## 🤖 AI Wins!")
            else:
                st.info("## 🤝 Tie!")
        
        # Strategy prediction details
        if predicted_user_move:
            st.markdown(f"**AI Prediction:** You would play `{predicted_user_move}` → AI countered with `{ai_move}`")

with col_right:
    st.subheader("📈 Strategy Leaderboard")
    st.caption("Shows overall performance across ALL rounds where each strategy was used")
    
    leaderboard = st.session_state.game.get_leaderboard()
    
    if len(leaderboard) > 0:
        # Prepare data for dataframe
        table_data = []
        for rank, entry in enumerate(leaderboard, 1):
            strategy = st.session_state.game.strategies[entry['arm']]
            is_current = (selected_arm is not None and entry['arm'] == selected_arm)
            
            table_data.append({
                'Rank': rank,
                'Strategy': f"⭐ {strategy.name}" if is_current else strategy.name,
                'Win %': f"{entry['win_pct']:.1f}%",
                'Wins': entry['wins'],
                'Losses': entry['losses'],
                'Ties': entry['ties'],
                'Total': entry['total_games'],
                'Avg Reward': f"{entry['avg_reward']:.2f}"
            })
        
        df = pd.DataFrame(table_data)
        
        # Display table with proper formatting
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Strategy": st.column_config.TextColumn("Strategy", width="medium"),
                "Win %": st.column_config.TextColumn("Win %", width="small"),
                "Wins": st.column_config.NumberColumn("Wins", width="small"),
                "Losses": st.column_config.NumberColumn("Losses", width="small"),
                "Ties": st.column_config.NumberColumn("Ties", width="small"),
                "Total": st.column_config.NumberColumn("Total", width="small"),
                "Avg Reward": st.column_config.TextColumn("Avg Reward", width="small")
            }
        )
        
        if selected_arm is not None:
            current_strategy = st.session_state.game.strategies[selected_arm]
            st.info(f"💡 Current round used: **{current_strategy.name}** (marked with ⭐)")
    else:
        st.info("Play some rounds to see the leaderboard!")

st.divider()

# Strategy Details Section
if ai_move and selected_arm is not None:
    with st.expander(f"🔍 Detailed Strategy Explanation: {st.session_state.game.strategies[selected_arm].name}"):
        strategy = st.session_state.game.strategies[selected_arm]
        
        st.markdown(f"### {strategy.name}")
        st.markdown(f"**{strategy.short_description}**")
        st.divider()
        st.markdown("**Full Explanation:**")
        st.markdown(strategy.description)
        
        # Show performance stats
        st.divider()
        win_pct = st.session_state.game.get_win_percentage(selected_arm)
        total_games = int(
            st.session_state.game.arm_wins[selected_arm] +
            st.session_state.game.arm_losses[selected_arm] +
            st.session_state.game.arm_ties[selected_arm]
        )
        
        if total_games > 0:
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Win Rate", f"{win_pct:.1f}%")
            with col_stat2:
                st.metric("Wins", int(st.session_state.game.arm_wins[selected_arm]))
            with col_stat3:
                st.metric("Losses", int(st.session_state.game.arm_losses[selected_arm]))
            with col_stat4:
                st.metric("Avg Reward", f"{st.session_state.game.arm_values[selected_arm]:.2f}")

st.divider()

# All Strategies Overview
with st.expander("📚 Learn About All 11 AI Strategies"):
    st.markdown("### Strategy Overview")
    st.markdown("The AI uses a Multi-Armed Bandit algorithm to select the best strategy based on recent performance. "
                "Only the most recent 20 rounds are used for strategy selection with exponential decay.")
    st.divider()
    
    # Simple strategies
    st.markdown("#### 🎯 Simple Pattern Detection Strategies")
    for i, strategy in enumerate(st.session_state.game.strategies[:5]):
        st.markdown(f"**{i+1}. {strategy.name}**")
        st.markdown(f"*{strategy.short_description}*")
        st.markdown(strategy.description)
        st.divider()
    
    # Advanced strategies
    st.markdown("#### 🚀 Advanced Strategies")
    for i, strategy in enumerate(st.session_state.game.strategies[5:], 6):
        st.markdown(f"**{i}. {strategy.name}**")
        st.markdown(f"*{strategy.short_description}*")
        st.markdown(strategy.description)
        if i < len(st.session_state.game.strategies):
            st.divider()

st.divider()

# New game button
if st.button("🔄 New Game", use_container_width=True, type="secondary"):
    st.session_state.game.reset_game()
    st.rerun()

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Uses Multi-Armed Bandit with 11 Strategies (Recent 20 rounds with decay)")
