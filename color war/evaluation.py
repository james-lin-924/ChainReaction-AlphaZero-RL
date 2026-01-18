from color_war_env import ChainReactionEnv
from neural_structure import AlphaZeroNet
import torch
import numpy as np

def play_match(model, game, ai_plays_first=True):
    state = game.get_initial_state()
    player = 1
    move_count = 0
    
    while True:
        # 1. Get Valid Moves
        is_first = move_count < 2
        valid_moves = game.get_valid_moves(state, player, is_first)
        valid_indices = np.where(valid_moves == 1)[0]
        
        if len(valid_indices) == 0: break 
        
        # 2. Select Action
        if (ai_plays_first and player == 1) or (not ai_plays_first and player == -1):
            # --- AI Turn ---
            # Preprocess state for AI
            canonical = state.copy()
            canonical[:,:,1] *= player # Flip perspective
            
            # Convert to Tensor
            tensor = torch.FloatTensor(canonical).permute(2,0,1).unsqueeze(0).to('cuda')
            
            # Predict (No MCTS for speed, just raw Policy Head)
            with torch.no_grad():
                pi, v = model(tensor)
            
            probs = torch.exp(pi).cpu().numpy()[0]
            probs = probs * valid_moves # Mask invalid
            action = np.argmax(probs) # Greedy move
        else:
            # --- Random Turn ---
            action = np.random.choice(valid_indices)
            
        # 3. Apply Move
        is_r = (move_count == 0)
        is_b = (move_count == 1)
        state = game.get_next_state(state, action, player, is_r, is_b)
        
        # 4. Check Winner
        val, term = game.get_value_and_terminated(state, action, is_first)
        if term:
            # If current player won, return 1 if AI is current player, else 0
            if (ai_plays_first and player == 1) or (not ai_plays_first and player == -1):
                return 1 # AI Won
            return 0 # AI Lost
            
        player = -player
        move_count += 1
        if move_count > 200: return 0.5 # Draw

def evaluate_performance():
    game = ChainReactionEnv(grid_size=5)
    net = AlphaZeroNet(grid_size=5).to('cuda')
    net.load_state_dict(torch.load("latest_model.pth"))
    net.eval()
    
    wins = 0
    n_games = 50
    
    print(f"Evaluating AI vs Random over {n_games} games...")
    
    for i in range(n_games):
        # AI plays first half the time
        ai_first = (i % 2 == 0)
        result = play_match(net, game, ai_plays_first=ai_first)
        wins += result
        
    print(f"Win Rate: {wins/n_games * 100:.1f}%")

if __name__ == "__main__":
    evaluate_performance()