import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from color_war_env import ChainReactionEnv 
from neural_structure import AlphaZeroNet
from MCTS import MCTS

# --- Configuration ---
ARGS = {
    'num_iterations': 100,       
    'num_episodes': 20,          # Increased: GPU can handle more episodes faster
    'num_simulations': 50,       
    'c_puct': 1.0,               
    'batch_size': 128,           # Increased: GPUs prefer larger batches
    'epochs': 10,                
    'lr': 0.001,
    'grid_size': 5
}

# 1. Setup Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def execute_episode(game, model, args):
    train_examples = []
    state = game.get_initial_state()
    current_player = 1
    move_count = 0
    mcts = MCTS(game, model, args)
    
    while True:
        mcts_probs = mcts.search(state, current_player, move_count)
        
        canonical_state = state.copy()
        canonical_state[:,:,1] = canonical_state[:,:,1] * current_player
        train_examples.append([canonical_state, current_player, mcts_probs, None])
        
        action = np.random.choice(len(mcts_probs), p=mcts_probs)
        
        is_first_r = (move_count == 0)
        is_first_b = (move_count == 1)
        state = game.get_next_state(state, action, current_player, is_first_r, is_first_b)
        
        is_first_round = move_count < 2
        reward, terminated = game.get_value_and_terminated(state, action, is_first_round)
        
        if terminated:
            final_data = []
            for x in train_examples:
                saved_state, saved_player, saved_pi, _ = x
                v = 1 if saved_player == current_player else -1
                final_data.append((saved_state, saved_pi, v))
            return final_data

        current_player = -current_player
        move_count += 1
        
        if move_count > 200:
            return []

def train(model, optimizer, samples):
    model.train()
    batch_size = ARGS['batch_size']
    random.shuffle(samples)
    
    avg_loss = 0
    batch_count = 0
    
    for _ in range(ARGS['epochs']):
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            if len(batch) < 4: continue 
            
            # --- GPU FIX: Move data to DEVICE ---
            states = torch.FloatTensor(np.array([x[0] for x in batch])).permute(0, 3, 1, 2).to(DEVICE)
            target_pis = torch.FloatTensor(np.array([x[1] for x in batch])).to(DEVICE)
            target_vs = torch.FloatTensor(np.array([x[2] for x in batch])).view(-1, 1).to(DEVICE)
            
            out_pi, out_v = model(states)
            
            loss_v = F.mse_loss(out_v, target_vs)
            loss_pi = -torch.mean(torch.sum(target_pis * out_pi, 1))
            total_loss = loss_v + loss_pi
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            avg_loss += total_loss.item()
            batch_count += 1
            
    return avg_loss / batch_count if batch_count > 0 else 0

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Total Loss')
    plt.title(f'AlphaZero Training Loss ({DEVICE})')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

if __name__ == "__main__":
    print(f"Initializing AlphaZero environment on {DEVICE}...", flush=True)
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    game = ChainReactionEnv(grid_size=ARGS['grid_size'])
    
    # --- GPU FIX: Move model to DEVICE ---
    nnet = AlphaZeroNet(grid_size=ARGS['grid_size']).to(DEVICE)
    
    optimizer = optim.Adam(nnet.parameters(), lr=ARGS['lr'])
    
    if os.path.exists("latest_model.pth"):
        print("Loading latest_model.pth...")
        # Load map_location handles loading GPU model on CPU or vice versa automatically if needed
        nnet.load_state_dict(torch.load("latest_model.pth", map_location=DEVICE))

    loss_history = []
    best_loss = float('inf')
    
    try:
        for i in range(ARGS['num_iterations']):
            print(f"\n--- Iteration {i+1}/{ARGS['num_iterations']} ---")
            
            iteration_samples = []
            
            for _ in tqdm(range(ARGS['num_episodes']), desc="Self Play"):
                data = execute_episode(game, nnet, ARGS)
                iteration_samples.extend(data)
            
            print(f"Collected {len(iteration_samples)} samples. Training network...")
            
            if len(iteration_samples) > 0:
                curr_loss = train(nnet, optimizer, iteration_samples)
                loss_history.append(curr_loss)
                print(f"Iteration Loss: {curr_loss:.4f}")
                
                plot_history(loss_history)
                torch.save(nnet.state_dict(), "latest_model.pth")
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    torch.save(nnet.state_dict(), "best_model.pth")
                    print("New best model saved!")
            else:
                print("No samples collected. Skipping train.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        plot_history(loss_history)
        torch.save(nnet.state_dict(), "interrupted_model.pth")
        print("Saved state. Exiting.")