import gradio as gr
import torch
import numpy as np
import time
import sys
import os

# Add current directory to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from color_war_env import ChainReactionEnv
from neural_structure import AlphaZeroNet
from MCTS import MCTS

# --- Configuration ---
MODEL_PATH = "best_model.pth" 
GRID_SIZE = 5

# INCREASED SIMULATIONS
MCTS_SIMS = 800  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
def load_model():
    print(f"Initializing MLP Model on {DEVICE}...")
    model = AlphaZeroNet(grid_size=GRID_SIZE).to(DEVICE)
    
    path = MODEL_PATH
    if not os.path.exists(path):
        if os.path.exists("latest_model.pth"):
            path = "latest_model.pth"
        else:
            print(f"⚠️ Warning: No model found at {MODEL_PATH} or latest_model.pth. AI will play randomly.")
            return model

    print(f"Loading weights from {path}...")
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        print("✅ Model loaded successfully.")
    except RuntimeError as e:
        print(f"❌ Architecture Mismatch: {e}")
        print("Ensure neural_structure.py defines the MLP (Fully Connected) architecture.")
    
    return model

MODEL = load_model()
GAME = ChainReactionEnv(grid_size=GRID_SIZE)

# --- CSS Styling ---
CUSTOM_CSS = """
/* The container for the whole board */
#board-container {
    width: 450px !important;
    margin: 0 auto !important;
    background-color: #222;
    padding: 10px;
    border-radius: 10px;
}

/* Remove default gaps in Gradio rows to make it look like a grid */
.board-row {
    gap: 5px !important;
    margin-bottom: 5px !important; 
    justify-content: center !important; 
    display: flex !important;           
}

/* Individual Cells */
.cell-btn {
    height: 70px !important;
    width: 70px !important;
    min-width: 70px !important; 
    max-width: 70px !important; 
    font-size: 24px !important;
    font-weight: bold !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    border-radius: 6px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1 !important;
    box-shadow: none !important;
    transition: all 0.2s; /* Add smooth transition */
}

/* Colors */
.cell-neutral { background-color: #e0e0e0 !important; color: transparent !important; }
.cell-neutral:hover { background-color: #d0d0d0 !important; }

.cell-red { 
    background-color: #ff9999 !important; 
    color: #cc0000 !important;
    border: 3px solid #cc0000 !important;
}

.cell-blue { 
    background-color: #99ccff !important; 
    color: #0033cc !important;
    border: 3px solid #0033cc !important;
}

/* Highlighting the AI's last move */
.last-move {
    box-shadow: 0 0 15px 5px #ffff00 !important; /* Yellow glow */
    border-color: #ffff00 !important; /* Yellow border */
    z-index: 10; 
    transform: scale(1.05); 
}

/* Status Box */
#status-box textarea {
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    border: 2px solid #555;
}
"""

# --- Game Logic ---

def format_board(state, last_ai_move_idx=None):
    """
    Renders the board.
    last_ai_move_idx: The index of the cell the AI just played on (to highlight).
    """
    updates = []
    atoms = state[:, :, 0].flatten()
    colors = state[:, :, 1].flatten()
    
    for i in range(25):
        count = int(atoms[i])
        owner = int(colors[i])
        
        classes = ["cell-btn"]
        text = ""

        if owner == 1:
            classes.append("cell-red")
            text = "●" * count 
        elif owner == -1:
            classes.append("cell-blue")
            text = "●" * count
        else:
            classes.append("cell-neutral")
        
        # Apply Highlight if this is the AI's last move
        if last_ai_move_idx is not None and i == last_ai_move_idx:
            classes.append("last-move")
            
        updates.append(gr.update(value=text, elem_classes=classes))
            
    return updates

def ai_move_logic(state, move_count, ai_player_id):
    """Executes AI move and returns (action, new_state, reward, terminated)"""
    args = {'num_simulations': MCTS_SIMS, 'c_puct': 1.0}
    mcts = MCTS(GAME, MODEL, args)
    
    start_time = time.time()
    # Search from the perspective of the AI player
    probs = mcts.search(state, ai_player_id, move_count)
    duration = time.time() - start_time
    
    best_move = np.argmax(probs)
    
    # Determine flags for Chain Reaction env
    is_first_r = (move_count == 0)
    is_first_b = (move_count == 1)
    
    new_state = GAME.get_next_state(state, best_move, ai_player_id, first_move_r=is_first_r, first_move_b=is_first_b)
    
    is_first_round = move_count < 2
    reward, terminated = GAME.get_value_and_terminated(new_state, best_move, is_first_round)
    
    print(f"🤖 AI Thought Time: {duration:.3f}s | Sims: {MCTS_SIMS} | Choice: {best_move}")
    return best_move, new_state, reward, terminated

def on_click(btn_idx, state, move_count, human_player_id, last_ai_move):
    """
    human_player_id: 1 (Red) or -1 (Blue)
    """
    if state is None:
        return restart_game("🔴 玩家先手 (紅)") # Default fallback

    # Determine Opponent ID
    ai_player_id = -human_player_id

    # --- 1. Validate HUMAN Move ---
    flat_colors = state[:,:,1].reshape(-1)
    is_first_round = move_count < 2
    
    if is_first_round:
        if flat_colors[btn_idx] != 0:
            msg = "⚠️ Invalid: First move must be on empty cell."
            return format_board(state, last_ai_move) + [msg, state, move_count, human_player_id, last_ai_move]
    else:
        # Cannot place on opponent
        if flat_colors[btn_idx] == ai_player_id:
             msg = "⚠️ Invalid: Cannot place on Opponent's cell."
             return format_board(state, last_ai_move) + [msg, state, move_count, human_player_id, last_ai_move]
        
        # Must place on own color if you have any
        has_own_cells = np.sum(flat_colors == human_player_id) > 0
        if flat_colors[btn_idx] == 0 and has_own_cells:
             msg = "⚠️ Invalid: Must expand from your existing cells."
             return format_board(state, last_ai_move) + [msg, state, move_count, human_player_id, last_ai_move]

    # --- 2. Execute HUMAN Move ---
    is_first_r = (move_count == 0)
    is_first_b = (move_count == 1)
    
    new_state = GAME.get_next_state(state, btn_idx, human_player_id, first_move_r=is_first_r, first_move_b=is_first_b)
    move_count += 1
    
    reward, terminated = GAME.get_value_and_terminated(new_state, btn_idx, move_count < 2)
    
    if terminated:
        status = "🏆 YOU WIN! 🏆" if reward == 1 else "💀 YOU LOST! 💀"
        return format_board(new_state, None) + [status, new_state, move_count, human_player_id, None]
            
    # --- 3. AI Turn ---
    ai_action, final_state, reward, terminated = ai_move_logic(new_state, move_count, ai_player_id)
    move_count += 1
    
    updates = format_board(final_state, last_ai_move_idx=ai_action)
    
    status = f"🔴 { 'Red' if human_player_id == 1 else 'Blue'} (Your Turn)"
    if terminated:
        status = "💀 AI WINS! 💀" if reward == 1 else "🏆 YOU WIN! 🏆"
            
    return updates + [status, final_state, move_count, human_player_id, ai_action]

def restart_game(player_choice):
    """
    player_choice: String from Radio button
    """
    state = GAME.get_initial_state()
    move_count = 0
    last_ai_move = None
    
    # --- BUG FIX HERE: Correctly check for "Player" (玩家) in Chinese string ---
    if "玩家" in player_choice:
        human_player_id = 1  # Human is Red
        status = "🔴 Your Turn (Red)"
        updates = format_board(state, None)
    else:
        # AI Plays First (Red)
        human_player_id = -1 # Human is Blue
        status = "🔵 AI (Red) is thinking..."
        
        # AI Plays First Move immediately
        ai_action, state, reward, terminated = ai_move_logic(state, move_count, 1) # AI is Red (1)
        move_count += 1
        last_ai_move = ai_action
        
        status = "🔵 Your Turn (Blue)"
        updates = format_board(state, last_ai_move_idx=ai_action)

    return updates + [status, state, move_count, human_player_id, last_ai_move]

# --- UI Construction ---

RULES_MARKDOWN = """
###  How to Play
1. **目標:** 清除棋盤上對方的所有點。
2. **規則:** 每回合在己方顏色的格子增加一個點(第一回合可放三個)。
3. **爆炸與擴散:** 一個格子在原子數達到 **4** 時會爆炸，原子會向四周擴散，可能造成連鎖反應。
4. **順序** - **Player First:** 你是紅方(先手).
   - **AI First:** AI 是紅方(先手), 你是藍方(後手).
"""

with gr.Blocks(title="Chain Reaction AI", css=CUSTOM_CSS) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Chain Reaction trained by AlphaZero</h1>")
    
    with gr.Accordion("規則 Game Rules", open=False):
        gr.Markdown(RULES_MARKDOWN)

    # Persistent State
    board_state = gr.State()
    move_count = gr.State()
    human_player = gr.State(value=1) # 1 = Red, -1 = Blue
    last_ai_move = gr.State(value=None) 
    
    with gr.Row():
        with gr.Column(scale=1):
            player_radio = gr.Radio(
                choices=["🔴 玩家先手 (紅)", "🔵 AI 先手 (紅)"], 
                value="🔴 玩家先手 (紅)", 
                label="選擇先後手",
                interactive=True
            )
            start_btn = gr.Button("🔄 New Game", variant="primary")
        
        status_box = gr.Textbox(label="Game Status", value="Press Start", elem_id="status-box", interactive=False, scale=3)

    buttons = []
    with gr.Column(elem_id="board-container"):
        for r in range(GRID_SIZE):
            with gr.Row(elem_classes="board-row"):
                for c in range(GRID_SIZE):
                    btn = gr.Button(value="", elem_classes=["cell-btn", "cell-neutral"])
                    buttons.append(btn)
    
    # Output list
    outputs = buttons + [status_box, board_state, move_count, human_player, last_ai_move]
    
    start_btn.click(
        restart_game, 
        inputs=[player_radio], 
        outputs=outputs
    )
    
    for i, btn in enumerate(buttons):
        btn.click(
            lambda s, m, h, l, idx=i: on_click(idx, s, m, h, l),
            inputs=[board_state, move_count, human_player, last_ai_move],
            outputs=outputs
        )
    
    # Initialize logic on load
    demo.load(restart_game, inputs=[player_radio], outputs=outputs)

if __name__ == "__main__":
    demo.launch(share=True)