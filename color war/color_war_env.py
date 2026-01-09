import numpy as np


class ChainReactionEnv:
    """
    A class to represent the Chain Reaction game environment suitable for 
    AI/Reinforcement Learning, using a 5x5 grid.
    
    Players: 1 (Red) and -1 (Blue).
    State: A 5x
    5x2 NumPy array. 
           Layer 0: Atom count (0-3)
           Layer 1: Owner (1 for Red, -1 for Blue, 0 for Neutral)
    """

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.action_size = grid_size * grid_size
        self.critical_mass = 4
    
    # --- State Management ---
    
    def get_initial_state(self):
        """
        Returns the initial game state as a 5x5x2 NumPy array.
        [Layer 0: Atoms, Layer 1: Color]
        """
        # (grid_size, grid_size, 2) array initialized to zeros
        return np.zeros((self.grid_size, self.grid_size, 2), dtype=np.int8)

    def _coords_to_action(self, r, c):
        """Converts grid coordinates (row, col) to a flat action index (0-24)."""
        return r * self.grid_size + c

    def _action_to_coords(self, action):
        """Converts a flat action index (0-24) to grid coordinates (row, col)."""
        r = action // self.grid_size
        c = action % self.grid_size
        return r, c

    # --- Core Game Logic ---

    def get_next_state(self, state, action, player, first_move_r=True, first_move_b=True):
        """
        Calculates the new state after a player makes a move (action).
        Note: This returns a deep copy to prevent modifying the original state.
        """
        # Create a deep copy of the state to work with
        new_state = state.copy() 
        r, c = self._action_to_coords(action)

        # 1. Place atom(s)
        # Check if it's a first move and apply the +3 bonus (modifying the original rule slightly)
        atom_increase = 1
        if player == 1 and first_move_r:
            atom_increase = 3
        elif player == -1 and first_move_b:
            atom_increase = 3
        
        # Place the atom(s) and set/claim ownership
        new_state[r, c, 0] += atom_increase
        new_state[r, c, 1] = player

        # 2. Handle Chain Reactions (Explosions)
        while True:
            # Find all cells that are over the critical mass
            exploding_cells = np.where(new_state[:, :, 0] >= self.critical_mass)
            
            if len(exploding_cells[0]) == 0:
                break # No more explosions, board is stable

            # Process all exploding cells in this wave
            for er, ec in zip(*exploding_cells):
                # Explosion removes critical mass
                explosion_color = new_state[er, ec, 1]
                new_state[er, ec, 0] = 0 # Atoms reset
                new_state[er, ec, 1] = 0 # Cell becomes neutral

                # Spread to neighbors
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                for dr, dc in directions:
                    nr, nc = er + dr, ec + dc
                    
                    # Check bounds
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        # Give atom
                        new_state[nr, nc, 0] += 1
                        # Claim neighbor's color (ownership transfer)
                        new_state[nr, nc, 1] = explosion_color
                        
        return new_state

    # --- Move & Win Condition Checks ---

    def get_valid_moves(self, state, player, first_round):
        """
        Returns a 1D array (size 25) where 1 indicates a valid move and 0 an invalid move.
        """
        # Get the color layer (Layer 1) and flatten it
        colors = state[:, :, 1].reshape(-1)
        valid_moves = np.zeros(self.action_size, dtype=np.uint8)

        if first_round:
            # First round: must place on a neutral cell (color == 0)
            valid_moves[colors == 0] = 1
        else:
            # After first round: must place on an owned cell (color == player)
            valid_moves[colors == player] = 1
            
        return valid_moves

    def get_cell_counts(self, state):
        """Returns the number of cells owned by Red (1) and Blue (-1)."""
        colors = state[:, :, 1]
        red_count = np.sum(colors == 1)
        blue_count = np.sum(colors == -1)
        return red_count, blue_count

    def get_value_and_terminated(self, state, action, first_round):
        """
        Determines the game outcome and termination status.
        Value: 1 for current player win, -1 for current player loss, 0 for draw/ongoing.
        Terminated: True/False.
        
        Note: The 'action' here is the last action that led to this state.
        """
        if first_round:
            # Game cannot end in the first round (must wait for ownership transfers)
            return 0, False

        red_count, blue_count = self.get_cell_counts(state)
        
        # The player who just moved is the one whose turn *was* next.
        # Check who lost all cells:
        
        # If the next player (player who didn't move) has no cells, the current player won.
        # We need to know who the *previous* player (the one who made the move) was.
        # A simpler approach is to check who remains on the board.
        
        if red_count == 0 and blue_count > 0:
            # Blue is the winner. If last player was Red (-1), Red lost (Value=-1).
            # If last player was Blue (1), Blue won (Value=1).
            # We don't have the last player info, so we assume the winner gets 1.
            return 1, True # Blue wins
            
        elif blue_count == 0 and red_count > 0:
            return 1, True # Red wins

        # The Chain Reaction game cannot naturally end in a draw (there is always a winner
        # as long as moves are possible).
        
        # Check if any valid move is left. In Chain Reaction, moves are almost always possible
        # unless one player is trapped, which isn't the win condition.
        # Since the win condition is defined by ownership count, if it's not a win, it's ongoing.
        return 0, False
        
    def get_opponent(self, player):
        """Returns the opposite player."""
        return -player
    
    # --- Visualization (Optional/For Debugging) ---
    
    def print_state(self, state):
        """Prints the state in a readable format."""
        atom_layer = state[:, :, 0]
        color_layer = state[:, :, 1]
        
        print("\n--- Atom Counts ---")
        print(atom_layer*color_layer)
        print("-" * 20)
