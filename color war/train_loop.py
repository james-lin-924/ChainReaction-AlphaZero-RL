import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

# Import your modules
from color_war_env import ChainReactionEnv 
from neural_structure import AlphaZeroNet
from MCTS import MCTS

# --- Configuration ---
ARGS = {
    'num_iterations': 1000,       
    'num_episodes': 20,          
    'num_simulations': 50,       
    'c_puct': 1.0,               
    'batch_size': 256,           
    'epochs': 5,                 
    'lr': 0.001,
    'grid_size': 5,
    'buffer_size': 30000,        
    'num_workers': 6             
}

TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_symmetries(state, pi):
    syms = []
    pi_board = np.reshape(pi, (ARGS['grid_size'], ARGS['grid_size']))
    for i in range(4):
        state_rot = np.rot90(state, i)
        pi_rot = np.rot90(pi_board, i)
        syms.append((state_rot, pi_rot.flatten()))
        state_flip = np.fliplr(state_rot)
        pi_flip = np.fliplr(pi_rot)
        syms.append((state_flip, pi_flip.flatten()))
    return syms

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
        
        symmetries = get_symmetries(canonical_state, mcts_probs)
        for sym_state, sym_pi in symmetries:
            train_examples.append([sym_state, current_player, sym_pi, None])
        
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
        
        if move_count > 100: 
            return []

def worker_self_play(shared_state_dict):
    torch.set_num_threads(1)
    
    local_game = ChainReactionEnv(grid_size=ARGS['grid_size'])
    local_net = AlphaZeroNet(grid_size=ARGS['grid_size']).to('cpu')
    
    local_net.load_state_dict(shared_state_dict)
    local_net.eval()
    
    return execute_episode(local_game, local_net, ARGS)

def train(model, optimizer, samples):
    model.train()
    batch_size = ARGS['batch_size']
    
    if len(samples) > 10000:
        samples = random.sample(samples, 10000)
    else:
        random.shuffle(samples)
    
    avg_loss = 0
    batch_count = 0
    
    for _ in range(ARGS['epochs']):
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            if len(batch) < 4: continue 
            
            states = torch.FloatTensor(np.array([x[0] for x in batch])).permute(0, 3, 1, 2).to(TRAIN_DEVICE)
            target_pis = torch.FloatTensor(np.array([x[1] for x in batch])).to(TRAIN_DEVICE)
            target_vs = torch.FloatTensor(np.array([x[2] for x in batch])).view(-1, 1).to(TRAIN_DEVICE)
            
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
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(history, label='Total Loss')
        plt.title(f'AlphaZero Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.close()
    except Exception:
        pass

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    print(f"Initializing AlphaZero on {TRAIN_DEVICE}...", flush=True)
    
    nnet = AlphaZeroNet(grid_size=ARGS['grid_size']).to(TRAIN_DEVICE)
    optimizer = optim.Adam(nnet.parameters(), lr=ARGS['lr'])
    
    replay_buffer = deque(maxlen=ARGS['buffer_size'])
    
    if os.path.exists("latest_model.pth"):
        print("Checking checkpoint...")
        try:
            nnet.load_state_dict(torch.load("latest_model.pth", map_location=TRAIN_DEVICE))
            print("Checkpoint loaded successfully.")
        except RuntimeError:
            print("Architecture mismatch detected (CNN -> MLP). Starting fresh.")
            os.rename("latest_model.pth", "archived_cnn_model.pth")

    loss_history = []
    # --- RESTORED FEATURE: Best Loss Tracking ---
    best_loss = float('inf') 
    # --------------------------------------------
    
    try:
        for i in range(ARGS['num_iterations']):
            print(f"\n--- Iteration {i+1}/{ARGS['num_iterations']} ---")
            
            cpu_weights = {k: v.cpu() for k, v in nnet.state_dict().items()}
            worker_args = [cpu_weights for _ in range(ARGS['num_episodes'])]
            
            iteration_samples = []
            
            with mp.Pool(processes=ARGS['num_workers']) as pool:
                for result in tqdm(pool.imap_unordered(worker_self_play, worker_args), total=ARGS['num_episodes'], desc="Self-Play"):
                    iteration_samples.extend(result)
            
            replay_buffer.extend(iteration_samples)
            print(f"Collected {len(iteration_samples)} samples. Buffer: {len(replay_buffer)}")
            
            if len(replay_buffer) > 500:
                curr_loss = train(nnet, optimizer, list(replay_buffer))
                loss_history.append(curr_loss)
                print(f"Loss: {curr_loss:.4f}")
                
                plot_history(loss_history)
                torch.save(nnet.state_dict(), "latest_model.pth")
                
                # --- RESTORED FEATURE: Save Best Model ---
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    torch.save(nnet.state_dict(), "best_model.pth")
                    print(f"⭐ New Best Model Saved! (Loss: {best_loss:.4f})")
                # -----------------------------------------
            else:
                print("Buffer too small to train.")

    except KeyboardInterrupt:
        print("\nSaving and exiting...")
        torch.save(nnet.state_dict(), "interrupted_model.pth")