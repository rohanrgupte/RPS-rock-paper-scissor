"""
Streamlit UI for Rock Paper Scissor Game
"""

import streamlit as st
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

# Header
st.title("✂️ Rock Paper Scissor")
st.markdown("**AI-Powered Opponent using Multi-Armed Bandit & Random Forest**")

# Scoreboard
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Your Score", st.session_state.game.user_score)
with col2:
    st.metric("AI Score", st.session_state.game.ai_score)
with col3:
    total = st.session_state.game.user_score + st.session_state.game.ai_score
    st.metric("Rounds", total)

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
    
    with result_col3:
        st.markdown("### Result")
        if winner == 'user':
            st.success("🎉 You Win!")
        elif winner == 'ai':
            st.error("🤖 AI Wins!")
        else:
            st.info("🤝 Tie!")
    
    # Show AI prediction explanation
    with st.expander("🤖 AI Strategy"):
        strategy_names = ["Random Forest", "Pattern Detection", "UCB Direct"]
        strategy_name = strategy_names[selected_arm] if selected_arm < len(strategy_names) else "Unknown"
        
        st.markdown(f"**Strategy used:** `{strategy_name}` (Arm {selected_arm})")
        
        if len(st.session_state.game.user_history) > 1:
            # Show what AI predicted (the move it tried to counter)
            predicted_user_move = [k for k, v in st.session_state.game.BEATS.items() if v == ai_move][0]
            st.markdown(f"**AI predicted:** You would play `{predicted_user_move}`")
            st.markdown(f"**AI countered:** By playing `{ai_move}`")
        else:
            st.markdown(f"**AI move:** `{ai_move}` (exploring)")
        
        # Show MAB stats
        if st.session_state.game.total_plays > 0:
            st.markdown("**Strategy Performance:**")
            for i, name in enumerate(strategy_names):
                if st.session_state.game.arm_counts[i] > 0:
                    avg_reward = st.session_state.game.arm_values[i]
                    count = int(st.session_state.game.arm_counts[i])
                    st.markdown(f"- {name}: Avg reward = {avg_reward:.2f} ({count} uses)")
        
        if st.session_state.game.rf_fitted:
            st.success("✓ Random Forest pattern recognition active")
        else:
            st.info("Learning your patterns...")

st.divider()

# New game button
if st.button("🔄 New Game", use_container_width=True, type="secondary"):
    st.session_state.game.reset_game()
    st.rerun()

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Uses Multi-Armed Bandit & Random Forest")

