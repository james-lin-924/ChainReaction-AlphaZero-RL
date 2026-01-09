import gradio as gr
import torch
import numpy as np

from color_war_env import ChainReactionEnv
from neural_structure import AlphaZeroNet
from MCTS import MCTS

# --- Configuration ---
MODEL_PATH = "best_model.pth" 
GRID_SIZE = 5
MCTS_SIMS = 50 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
def load_model():
    print(f"Loading model from {MODEL_PATH} to {DEVICE}...")
    model = AlphaZeroNet(grid_size=GRID_SIZE).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Warning: Model file not found. AI will play randomly.")
    return model

MODEL = load_model()
GAME = ChainReactionEnv(grid_size=GRID_SIZE)

# --- CSS Styling (Visuals Only) ---
# We no longer use CSS for the 5x5 layout (Python does that now).
# This CSS just makes the buttons look good.
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
    justify-content: center !important; /* Center buttons in the row */
    display: flex !important;           /* Force flex behavior */
}

/* Individual Cells */
.cell-btn {
    height: 70px !important;
    width: 70px !important;
    min-width: 70px !important; /* Prevent shrinking */
    max-width: 70px !important; /* Prevent growing */
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

/* Status Box */
#status-box textarea {
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    border: 2px solid #555;
}
"""

# --- Game Logic ---

def get_initial_state():
    return GAME.get_initial_state(), 0, "🔴 Your Turn (Red)" 

def format_board(state):
    updates = []
    atoms = state[:, :, 0].flatten()
    colors = state[:, :, 1].flatten()
    
    for i in range(25):
        count = int(atoms[i])
        owner = int(colors[i])
        
        if owner == 1:
            css_class = "cell-red"
            text = "●" * count 
        elif owner == -1:
            css_class = "cell-blue"
            text = "●" * count
        else:
            css_class = "cell-neutral"
            text = ""
            
        updates.append(gr.update(value=text, elem_classes=["cell-btn", css_class]))
            
    return updates

def ai_move(state, move_count):
    args = {'num_simulations': MCTS_SIMS, 'c_puct': 1.0}
    mcts = MCTS(GAME, MODEL, args)
    probs = mcts.search(state, -1, move_count)
    return np.argmax(probs)

def on_click(btn_idx, state, move_count):
    if state is None:
        state, move_count, _ = get_initial_state()
        updates = format_board(state)
        return updates + ["Error. Restarting...", state, move_count]

    # --- 1. Validate User Move (Red) ---
    flat_colors = state[:,:,1].reshape(-1)
    is_first_round = move_count < 2
    
    if is_first_round:
        if flat_colors[btn_idx] != 0:
            updates = format_board(state)
            return updates + ["⚠️ Invalid: First move must be on empty cell.", state, move_count]
    else:
        if flat_colors[btn_idx] == -1:
             updates = format_board(state)
             return updates + ["⚠️ Invalid: Cannot place on Blue.", state, move_count]
        if flat_colors[btn_idx] != 1 and flat_colors[btn_idx] != 0:
             updates = format_board(state)
             return updates + ["⚠️ Invalid: Must place on Red cell.", state, move_count]
        if flat_colors[btn_idx] == 0 and np.sum(flat_colors == 1) > 0:
             updates = format_board(state)
             return updates + ["⚠️ Invalid: Must expand from your existing cells.", state, move_count]

    # --- 2. Execute User Move (Red) ---
    is_first_r = (move_count == 0)
    is_first_b = False 
    
    new_state = GAME.get_next_state(state, btn_idx, 1, first_move_r=is_first_r, first_move_b=is_first_b)
    move_count += 1
    
    reward, terminated = GAME.get_value_and_terminated(new_state, btn_idx, move_count < 2)
    if terminated:
        if reward == 1:
            updates = format_board(new_state)
            return updates + ["🏆 RED WINS! 🏆", new_state, move_count]
            
    # --- 3. AI Turn (Blue) ---
    ai_action = ai_move(new_state, move_count)
    is_first_r = False
    is_first_b = (move_count == 1) 
    
    final_state = GAME.get_next_state(new_state, ai_action, -1, first_move_r=is_first_r, first_move_b=is_first_b)
    move_count += 1
    
    reward, terminated = GAME.get_value_and_terminated(final_state, ai_action, move_count < 2)
    updates = format_board(final_state)
    
    status = "🔴 Your Turn (Red)"
    if terminated:
        status = "💀 BLUE WINS! 💀" if reward == 1 else "🏆 RED WINS! 🏆"
            
    return updates + [status, final_state, move_count]

def restart_game():
    state, mc, status = get_initial_state()
    updates = format_board(state)
    return updates + [status, state, mc]

# --- UI Construction ---

RULES_MARKDOWN = """
### 📜 How to Play
1. **Red (You)** moves first. **Blue (AI)** moves second.
2. **First Move Bonus:** Both players place **3 atoms** on their very first turn.
3. **Explosions:** A cell explodes when it reaches **4 atoms**, scattering atoms to neighbors.
4. **Capture:** Exploding into an opponent's cell captures it.
5. **Win:** Eliminate all opponent atoms.
"""

with gr.Blocks(title="Chain Reaction AI", css=CUSTOM_CSS) as demo:
    gr.Markdown("<h1 style='text-align: center;'>⚛️ Chain Reaction vs AlphaZero</h1>")
    
    with gr.Accordion("📖 Game Rules", open=False):
        gr.Markdown(RULES_MARKDOWN)

    # State
    board_state = gr.State()
    move_count = gr.State()
    
    with gr.Row():
        status_box = gr.Textbox(label="Game Status", value="Press Start", elem_id="status-box", interactive=False, scale=3)
        start_btn = gr.Button("🔄 New Game", variant="primary", scale=1)

    # --- THE LAYOUT FIX: Pure Python Rows ---
    # We create 5 horizontal rows, each with 5 buttons.
    # We store them in a single 'buttons' list for easy index access (0-24).
    buttons = []
    
    with gr.Column(elem_id="board-container"):
        for r in range(GRID_SIZE):
            with gr.Row(elem_classes="board-row"):
                for c in range(GRID_SIZE):
                    # Create button
                    btn = gr.Button(value="", elem_classes=["cell-btn", "cell-neutral"])
                    buttons.append(btn)
    
    # Events
    start_btn.click(restart_game, inputs=[], outputs=buttons + [status_box, board_state, move_count])
    
    for i, btn in enumerate(buttons):
        btn.click(
            lambda s, m, idx=i: on_click(idx, s, m),
            inputs=[board_state, move_count],
            outputs=buttons + [status_box, board_state, move_count]
        )
    
    demo.load(restart_game, inputs=[], outputs=buttons + [status_box, board_state, move_count])

if __name__ == "__main__":
    demo.launch(share=True)