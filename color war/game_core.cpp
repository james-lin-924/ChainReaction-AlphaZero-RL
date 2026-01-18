#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

// Constants
const int GRID_SIZE = 5;
const int CRITICAL_MASS = 4;

// Helper to get raw pointer access for speed
int get_idx(int r, int c) {
    return r * GRID_SIZE + c;
}

// Check bounds
bool is_valid(int r, int c) {
    return r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE;
}

// The heavy logic: Computing the next state
py::array_t<int> get_next_state(py::array_t<int> state_array, int action, int player, bool is_first_r, bool is_first_b) {
    // Request buffer info
    py::buffer_info buf = state_array.request();
    int* ptr = static_cast<int*>(buf.ptr);

    // Create a copy for the new state (Standard C++ Vector for speed)
    std::vector<int> atoms(GRID_SIZE * GRID_SIZE);
    std::vector<int> owners(GRID_SIZE * GRID_SIZE);

    // Copy data from Numpy to C++ vector
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        atoms[i] = ptr[i * 2 + 0]; // Layer 0
        owners[i] = ptr[i * 2 + 1]; // Layer 1
    }

    // 1. Place Atom
    int r = action / GRID_SIZE;
    int c = action % GRID_SIZE;
    int idx = get_idx(r, c);

    int atom_increase = 1;
    if ((player == 1 && is_first_r) || (player == -1 && is_first_b)) {
        atom_increase = 3; // Bonus rule
    }

    atoms[idx] += atom_increase;
    owners[idx] = player;

    // 2. Handle Explosions (The slow part in Python)
    bool unstable = true;
    while (unstable) {
        unstable = false;
        std::vector<int> current_atoms = atoms; // Snapshot for this wave
        
        for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            if (current_atoms[i] >= CRITICAL_MASS) {
                unstable = true;
                int explosion_color = owners[i];
                
                // Reset exploded cell
                atoms[i] -= CRITICAL_MASS; 
                if (atoms[i] < 0) atoms[i] = 0; // Safety
                if (atoms[i] == 0) owners[i] = 0;

                // Neighbors
                int cr = i / GRID_SIZE;
                int cc = i % GRID_SIZE;
                int dr[] = {0, 0, 1, -1};
                int dc[] = {1, -1, 0, 0};

                for (int d = 0; d < 4; d++) {
                    int nr = cr + dr[d];
                    int nc = cc + dc[d];
                    if (is_valid(nr, nc)) {
                        int n_idx = get_idx(nr, nc);
                        atoms[n_idx]++;
                        owners[n_idx] = explosion_color;
                    }
                }
            }
        }
    }

    // Convert back to Numpy
    auto result = py::array_t<int>({GRID_SIZE, GRID_SIZE, 2});
    auto r_buf = result.request();
    int* r_ptr = static_cast<int*>(r_buf.ptr);

    for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        r_ptr[i * 2 + 0] = atoms[i];
        r_ptr[i * 2 + 1] = owners[i];
    }

    return result;
}

PYBIND11_MODULE(game_core, m) {
    m.def("get_next_state", &get_next_state, "Fast C++ Next State");
}