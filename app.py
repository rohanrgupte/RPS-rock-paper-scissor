"""
Streamlit UI for Rock Paper Scissor Game
Enhanced with 6 advanced AI strategies
"""

import streamlit as st
import numpy as np
from rps_game import MABRPSGame

# Page config
st.set_page_config(
    page_title="Rock Paper Scissor",
    page_icon="✂️",
    layout="centered"
)

# Initialize game
if 'game' not in st.session_state:
    st.session_state.game = MABRPSGame()

# Strategy names and icons
STRATEGY_NAMES = [
    "Iocaine Powder",
    "Markov Chains",
    "Win-Stay/Lose-Shift",
    "Frequency Decay",
    "Random Forest",
    "Transition Matrix"
]

STRATEGY_ICONS = ["💎", "🔗", "🔄", "📊", "🌲", "🔢"]

STRATEGY_DESCRIPTIONS_SHORT = [
    "Multi-pattern voting system",
    "Multi-order probability models",
    "Detects win/loss patterns",
    "Weighted frequency analysis",
    "ML pattern recognition",
    "Statistical transition model"
]

# Header
st.title("✂️ Rock Paper Scissor")
st.markdown("**AI-Powered Opponent using 6 Advanced Strategies with Multi-Armed Bandit**")

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
    # Show current best strategy (if available)
    if hasattr(st.session_state.game, 'arm_values') and st.session_state.game.total_plays > 0:
        best_arm = np.argmax(st.session_state.game.arm_values) if np.any(st.session_state.game.arm_counts > 0) else None
        if best_arm is not None and best_arm < len(STRATEGY_NAMES) and st.session_state.game.arm_counts[best_arm] > 0:
            st.metric("Best Strategy", f"{STRATEGY_ICONS[best_arm]} {STRATEGY_NAMES[best_arm]}")
        else:
            st.metric("Best Strategy", "Exploring")
    else:
        st.metric("Best Strategy", "Exploring")

st.divider()

# Game controls
st.subheader("Make Your Move")

move_cols = st.columns(3)
with move_cols[0]:
    rock = st.button("🪨 Rock", use_container_width=True, type="primary")
with move_cols[1]:
    paper = st.button("📄 Paper", use_container_width=True, type="primary")
with move_cols[2]:
    scissor = st.button("✂️ Scissor", use_container_width=True, type="primary")

# Process move
if rock or paper or scissor:
    user_move = 'rock' if rock else ('paper' if paper else 'scissor')
    ai_move, winner, selected_arm = st.session_state.game.play_round(user_move)
    
    # Display results
    st.divider()
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.markdown("### Your Move")
        emoji = "🪨" if user_move == "rock" else ("📄" if user_move == "paper" else "✂️")
        st.markdown(f"## {emoji}")
        st.markdown(f"**{user_move.title()}**")
    
    with result_col2:
        st.markdown("### AI Move")
        emoji = "🪨" if ai_move == "rock" else ("📄" if ai_move == "paper" else "✂️")
        st.markdown(f"## {emoji}")
        st.markdown(f"**{ai_move.title()}**")
        
        # Show strategy used prominently
        if selected_arm < len(STRATEGY_NAMES):
            strategy_name = STRATEGY_NAMES[selected_arm]
            strategy_icon = STRATEGY_ICONS[selected_arm]
            strategy_desc = STRATEGY_DESCRIPTIONS_SHORT[selected_arm]
            st.caption(f"{strategy_icon} **Strategy:** {strategy_name}")
            st.caption(f"*{strategy_desc}*")
        else:
            st.caption("❓ **Strategy:** Unknown")
    
    with result_col3:
        st.markdown("### Result")
        if winner == 'user':
            st.success("🎉 You Win!")
        elif winner == 'ai':
            st.error("🤖 AI Wins!")
        else:
            st.info("🤝 Tie!")
    
    # Show AI prediction explanation
    with st.expander("🤖 AI Strategy Details"):
        if selected_arm < len(STRATEGY_NAMES):
            strategy_name = STRATEGY_NAMES[selected_arm]
            strategy_icon = STRATEGY_ICONS[selected_arm]
            strategy_desc = STRATEGY_DESCRIPTIONS_SHORT[selected_arm]
            full_description = st.session_state.game.STRATEGY_DESCRIPTIONS.get(selected_arm, "No description available")
        else:
            strategy_name = "Unknown"
            strategy_icon = "❓"
            strategy_desc = "Unknown strategy"
            full_description = "No description available"
        
        # Highlight current strategy
        st.markdown(f"### {strategy_icon} **Current Strategy:** {strategy_name} (Arm {selected_arm})")
        st.markdown(f"*{strategy_desc}*")
        
        st.divider()
        st.markdown("**📖 Full Explanation:**")
        st.info(full_description)
        
        st.divider()
        
        if len(st.session_state.game.user_history) > 1:
            # Show what AI predicted (the move it tried to counter)
            predicted_user_move = [k for k, v in st.session_state.game.BEATS.items() if v == ai_move][0]
            st.markdown(f"**AI predicted:** You would play `{predicted_user_move}`")
            st.markdown(f"**AI countered:** By playing `{ai_move}`")
        else:
            st.markdown(f"**AI move:** `{ai_move}` (exploring)")
        
        st.divider()
        
        # Show MAB stats with visual indicators
        if st.session_state.game.total_plays > 0:
            st.markdown("### 📈 Strategy Performance")
            for i in range(st.session_state.game.num_arms):
                if i < len(STRATEGY_NAMES):
                    name = STRATEGY_NAMES[i]
                    icon = STRATEGY_ICONS[i]
                    short_desc = STRATEGY_DESCRIPTIONS_SHORT[i]
                else:
                    name = f"Strategy {i}"
                    icon = "❓"
                    short_desc = "Unknown"
                
                if st.session_state.game.arm_counts[i] > 0:
                    avg_reward = st.session_state.game.arm_values[i]
                    count = int(st.session_state.game.arm_counts[i])
                    
                    # Color code based on performance
                    if avg_reward > 0:
                        reward_indicator = "🟢"
                    elif avg_reward == 0:
                        reward_indicator = "🟡"
                    else:
                        reward_indicator = "🔴"
                    
                    # Highlight current strategy
                    if i == selected_arm:
                        st.markdown(f"**{icon} {name}** (Current) - {reward_indicator} Avg: {avg_reward:.2f} | Uses: {count}")
                        st.caption(f"   *{short_desc}*")
                    else:
                        st.markdown(f"{icon} {name} - {reward_indicator} Avg: {avg_reward:.2f} | Uses: {count}")
                        st.caption(f"   *{short_desc}*")
                else:
                    if i == selected_arm:
                        st.markdown(f"**{icon} {name}** (Current) - ⚪ Not used yet")
                        st.caption(f"   *{short_desc}*")
                    else:
                        st.markdown(f"{icon} {name} - ⚪ Not used yet")
        
        st.divider()
        
        if hasattr(st.session_state.game, 'rf_fitted') and st.session_state.game.rf_fitted:
            st.success("✓ Random Forest pattern recognition active")
        else:
            st.info("Learning your patterns...")

st.divider()

# Strategy Overview Section
with st.expander("📚 Learn About the 6 AI Strategies"):
    st.markdown("### Strategy Overview")
    
    for i, (name, icon, desc) in enumerate(zip(STRATEGY_NAMES, STRATEGY_ICONS, STRATEGY_DESCRIPTIONS_SHORT)):
        if i in st.session_state.game.STRATEGY_DESCRIPTIONS:
            full_desc = st.session_state.game.STRATEGY_DESCRIPTIONS[i]
            
            st.markdown(f"#### {icon} {name}")
            st.markdown(f"*{desc}*")
            st.markdown(full_desc)
            
            if i < len(STRATEGY_NAMES) - 1:
                st.divider()

st.divider()

# New game button
if st.button("🔄 New Game", use_container_width=True, type="secondary"):
    st.session_state.game.reset_game()
    st.rerun()

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Uses Multi-Armed Bandit with 6 Advanced Strategies")
