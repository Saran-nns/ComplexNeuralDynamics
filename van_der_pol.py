import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import IPython.display as IPdisplay
from PIL import Image


# Traget folder to save the result
save_folder = 'images/van_der_pol-animate'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# Define the initial system state (aka x, y positions in space)
initial_state = [0.1, 0]

# Define the system parameter(s): mu
mu = 0.5

# Define the time points to solve for, evenly spaced between the start and end times
start_time = 1
end_time = 60
interval = 100
time_points = np.linspace(start_time, end_time, end_time * interval)

# Define the van der pol system
def van_der_pol_oscillator_deriv(x, t):
    nx0 = x[1]
    nx1 = -mu * (x[0] ** 2.0 - 1.0) * x[1] - x[0]
    res = np.array([nx0, nx1])
    return res

# Plot the system in 2 dimensions
def van_der_pol_phase_potrait(xy, n):
    fig = plt.figure(figsize=(12, 9))
    ax= fig.gca()
    x = xy[:, 0]
    y = xy[:, 1]
    ax.plot(x, y, color='g', alpha=0.7, linewidth=0.7)
    ax.set_xlim((-10,10))
    ax.set_ylim((-10,10))
    ax.set_title('Van der Pol oscillator')
    
    plt.savefig('{}/{:03d}.png'.format(save_folder, n), dpi=60, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def van_der_pol_limit_cycle(mu):

    # Given llist of mu values; plot the system dynamics using phase potrait

    pass

def van_der_pol_behavior(x, external_forcing= None):

    # Plot the influence of external force on the oscillator dynamics
    
    pass


# Return a list in iteratively larger chunks
def get_chunks(full_list, size):
    size = max(1, size)
    chunks = [full_list[0:i] for i in range(1, len(full_list) + 1, size)]
    return chunks

# Get incrementally larger chunks of the time points, to reveal the attractor one frame at a time
chunks = get_chunks(time_points, size=20)

# get the points to plot, one chunk of time steps at a time, by integrating the system of equations
points = [odeint(van_der_pol_oscillator_deriv, initial_state, chunk) for chunk in chunks]

# plot each set of points, one at a time, saving each plot
for n, point in enumerate(points):
    van_der_pol_phase_potrait(point, n)



# Animate it
# Create an animated gif of all the plots then display it inline

# Create a tuple of display durations, one for each frame
first_last = 100 #show the first and last frames for 100 ms
standard_duration = 5 #show all other frames for 5 ms
durations = tuple([first_last] + [standard_duration] * (len(points) - 2) + [first_last])


# Load all the static images into a list
images = [Image.open(image) for image in glob.glob('{}/*.png'.format(save_folder))]
gif_filepath = 'images/van_der_pol_osc.gif'

# Save as an animated gif
gif = images[0]
gif.info['duration'] = durations #ms per frame
gif.info['loop'] = 0 #how many times to loop (0=infinite)
gif.save(fp=gif_filepath, format='gif', save_all=True, append_images=images[1:])

# Verify that the number of frames in the gif equals the number of image files and durations
Image.open(gif_filepath).n_frames == len(images) == len(durations)


IPdisplay.Image(url=gif_filepath)
