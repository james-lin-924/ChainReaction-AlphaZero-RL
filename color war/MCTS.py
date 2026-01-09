import numpy as np
import torch
import math

class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None, prior=0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = {} 
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior 
        
    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, game_env, model, args):
        self.game = game_env
        self.model = model
        self.args = args 

    def search(self, root_state, current_player, move_count):
        root = MCTSNode(root_state, prior=0)
        
        # Detect device automatically from the model
        device = next(self.model.parameters()).device

        for _ in range(self.args['num_simulations']):
            node = root
            sim_player = current_player
            sim_move_count = move_count
            curr_state = root_state.copy()

            # 1. Selection
            while node.is_expanded():
                best_score = -float('inf')
                best_action = -1
                
                for action, child in node.children.items():
                    q_value = -child.value()
                    u_value = self.args['c_puct'] * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
                    score = q_value + u_value
                    
                    if score > best_score:
                        best_score = score
                        best_action = action
                
                node = node.children[best_action]
                
                is_first_r = (sim_move_count == 0)
                is_first_b = (sim_move_count == 1)
                curr_state = self.game.get_next_state(curr_state, best_action, sim_player, is_first_r, is_first_b)
                sim_player = -sim_player
                sim_move_count += 1

            # 2. Expansion & Evaluation
            is_first_round = sim_move_count < 2
            value, terminated = self.game.get_value_and_terminated(curr_state, None, is_first_round)
            
            if terminated:
                leaf_value = -value 
            else:
                valid_moves = self.game.get_valid_moves(curr_state, sim_player, is_first_round)
                
                canonical_state = curr_state.copy()
                canonical_state[:,:,1] = canonical_state[:,:,1] * sim_player
                
                # --- GPU FIX HERE ---
                # Move tensor to the same device as the model
                tensor_state = torch.FloatTensor(canonical_state).permute(2, 0, 1).unsqueeze(0).to(device)
                
                self.model.eval()
                with torch.no_grad():
                    policy_logits, v_res = self.model(tensor_state)
                
                # Move back to CPU for numpy operations
                policy_probs = torch.exp(policy_logits).cpu().numpy()[0]
                leaf_value = v_res.item()
                
                policy_probs = policy_probs * valid_moves
                sum_probs = np.sum(policy_probs)
                if sum_probs > 0:
                    policy_probs /= sum_probs
                else:
                    policy_probs = valid_moves / np.sum(valid_moves)

                for action_idx, prob in enumerate(policy_probs):
                    if valid_moves[action_idx] > 0:
                        node.children[action_idx] = MCTSNode(None, parent=node, action_taken=action_idx, prior=prob)
            
            # 3. Backpropagation
            while node is not None:
                node.visit_count += 1
                node.value_sum += leaf_value
                node = node.parent
                leaf_value = -leaf_value 

        counts = np.array([root.children[a].visit_count if a in root.children else 0 for a in range(self.game.action_size)])
        if np.sum(counts) == 0:
            return np.ones(self.game.action_size) / self.game.action_size
        return counts / np.sum(counts)