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
        
        device = next(self.model.parameters()).device
        
        # Expand root immediately to add noise
        # This is a mini-expansion step just for the root
        self._expand_node(root, root_state, current_player, move_count, device, add_noise=True)

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
                    # UCB Formula
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

            # 2. Expansion & Evaluation (if not terminal and not already expanded)
            is_first_round = sim_move_count < 2
            value, terminated = self.game.get_value_and_terminated(curr_state, None, is_first_round)
            
            leaf_value = 0
            if terminated:
                leaf_value = -value 
                # Backpropagate terminal value immediately
                self._backpropagate(node, leaf_value)
            else:
                # Expand leaf
                # Note: We don't add noise to non-root nodes
                leaf_value = self._expand_node(node, curr_state, sim_player, sim_move_count, device, add_noise=False)
                # Backpropagate neural net value
                self._backpropagate(node, leaf_value)

        counts = np.array([root.children[a].visit_count if a in root.children else 0 for a in range(self.game.action_size)])
        
        if np.sum(counts) == 0:
            return np.ones(self.game.action_size) / self.game.action_size
        return counts / np.sum(counts)

    def _backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
            value = -value 

    def _expand_node(self, node, state, player, move_count, device, add_noise=False):
        """
        Expands the node using the Neural Network prediction.
        Returns the Value (v) of the state.
        """
        # Prepare state for Model
        is_first_round = move_count < 2
        valid_moves = self.game.get_valid_moves(state, player, is_first_round)
        
        canonical_state = state.copy()
        canonical_state[:,:,1] = canonical_state[:,:,1] * player
        
        tensor_state = torch.FloatTensor(canonical_state).permute(2, 0, 1).unsqueeze(0).to(device)
        
        self.model.eval()
        with torch.no_grad():
            policy_logits, v_res = self.model(tensor_state)
        
        policy_probs = torch.exp(policy_logits).cpu().numpy()[0]
        leaf_value = v_res.item()
        
        # Mask invalid moves
        policy_probs = policy_probs * valid_moves
        sum_probs = np.sum(policy_probs)
        if sum_probs > 0:
            policy_probs /= sum_probs
        else:
            policy_probs = valid_moves / np.sum(valid_moves)

        # --- NEW: Dirichlet Noise (Exploration) ---
        # Only add noise to the root node during training
        if add_noise and self.model.training:
            alpha = 0.3  # Standard for grid games
            noise = np.random.dirichlet([alpha] * len(policy_probs))
            epsilon = 0.25
            policy_probs = (1 - epsilon) * policy_probs + epsilon * noise
            # Renormalize just in case valid_moves masked some noise
            policy_probs = policy_probs * valid_moves
            policy_probs /= np.sum(policy_probs)

        for action_idx, prob in enumerate(policy_probs):
            if valid_moves[action_idx] > 0:
                node.children[action_idx] = MCTSNode(None, parent=node, action_taken=action_idx, prior=prob)
                
        return leaf_value