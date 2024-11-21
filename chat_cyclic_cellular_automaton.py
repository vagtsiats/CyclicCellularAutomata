import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import generate_binary_structure, iterate_structure

# Parameters
x, y = 200, 200  # Grid size
range_ = 2  # Neighborhood range
threshold = 3  # Activation threshold
states = 5  # Number of states
neighborhood_type = 1  # 1 for Von Neumann, 2 for Moore

# Initialize the automaton grid
array = np.random.randint(0, states, (y, x))

# Generate the neighborhood footprint
footprint = np.array(
    iterate_structure(generate_binary_structure(2, neighborhood_type), range_), dtype=int
)


def compute_func(values):
    """Rule function for the cellular automaton."""
    cur = values[int(len(values) / 2)]
    if cur == (states - 1):
        count = np.count_nonzero(values == 0)
    else:
        count = np.count_nonzero(values == cur + 1)
    if count >= threshold:
        cur += 1
    if cur == states:
        cur = 0
    return cur


def update(frame):
    global array
    array = ndimage.generic_filter(array, compute_func, footprint=footprint, mode="wrap")
    img.set_array(array)
    return (img,)


# Plot setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_axis_off()
img = ax.imshow(array, interpolation="none")

# Animation setup
ani = animation.FuncAnimation(fig, update, interval=100, blit=True, save_count=1000)
plt.show()

# # Directory for saving GIFs
# output_dir = "CCA_gifs"
# os.makedirs(output_dir, exist_ok=True)

# # Save as GIF
# gif_filename = f"{output_dir}/automaton_R{range_}_T{threshold}_S{states}_N{neighborhood_type}.gif"
# ani.save(gif_filename, writer="pillow", fps=10)
# print(f"Animation saved as {gif_filename}")
