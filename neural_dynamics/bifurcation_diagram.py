import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import IPython.display as IPdisplay
from PIL import Image



# Traget folder to save the result
save_folder = 'images/bifurcation-diagram-animate'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initial condition
x0  = 0
time_points = np.linspace(0.,1.,100)

# Define the system parameter(s): r
r = np.linspace(0.,4.,50)

# Define the logistic map function
def logistic(r, x):
    return r * x * (1 - x)
    
# Define the logistic map function
def logistic_map(r, x):
        """
        Args: 
        x - List of ratio of existing population to the maximum population
        r - Parameter of interest 
         """
        # Take the last point of x and measure the y

        y = logistic(r,x)
        
        return np.array([x,y])

# Plot the system in 2 dimensions
def plot_cobweb_diagram(points, n):
    

    fig = plt.figure(figsize=(12, 9))
    ax= fig.gca()


    # Plot the function

    ax.plot(time_points, logistic(r[n], time_points), 'k', lw=2)
    ax.text(0.1,0.8,'r = %s'%float(r[n]))

    for n,point in enumerate(points):

        x,y = point[0],point[1]
        # Plot a diagonal line where y = x
        ax.plot([0, 1], [0, 1], 'k', lw=2)
    
        # Plot the positions
        # ax.plot(x, y, 'ok', ms=10)
        ax.plot(x, y, color='g', alpha=0.7, linewidth=0.7)
        

        # Plot the two lines
        ax.plot([x, x], [x, y], 'g', lw=1) # Vertical lines
        ax.plot([x, y], [y, y], 'g', lw=1)   # Horizontal lines


    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.set_title('Cobweb diagram')
        
    plt.savefig('{}/{:03d}.png'.format(save_folder,n), dpi=60, bbox_inches='tight', pad_inches=0.1)
    plt.close()


# Return a list in iteratively larger chunks
def get_chunks(full_list, size):
    size = max(1, size)
    chunks = [full_list[0:i] for i in range(1, len(full_list) + 1, size)]
    return chunks


# Get incrementally larger chunks of the time points, to reveal the attractor one frame at a time
############### chunks = get_chunks(time_points, size=5)

# chunks = get_chunks(time_points, size=num_points//len(r))

# print(len(chunks))
# Get the points to plot, one chunk of time steps at a time, by solving the logistic func

############# points = []
# for i in range(len(time_points)):
#         points.append([logistic_map(r, chunk) for chunk in chunks[:i+1]])



X = []
Y = []
x = x0
for i in range(len(r)):
        X.append(x)
        x,y = logistic_map(r[i], x)
        Y.append(y)

points = list(zip(X,Y))       

# Get the chunks of points 

point_chunks = get_chunks(points, size=2)

for n,points in enumerate(point_chunks):
        plot_cobweb_diagram(points, n)


















# Animate it
# Create an animated gif of all the plots then display it inline

# Create a tuple of display durations, one for each frame
first_last = 300 #show the first and last frames for 300 ms
standard_duration = 20 #show all other frames for 20 ms
durations = tuple([first_last] + [standard_duration] * (len(points) - 2) + [first_last])

# durations =  len(os.listdir(os.getcwd()+'/images/bifurcation-diagram-animate'))

# Load all the static images into a list
images = [Image.open(image) for image in glob.glob('{}/*.png'.format(save_folder))]
gif_filepath = 'images/log_map_cobweb_diag.gif'

# Save as an animated gif
gif = images[0]
gif.info['duration'] = durations #ms per frame
gif.info['loop'] = 0 #how many times to loop (0=infinite)
gif.save(fp=gif_filepath, format='gif', save_all=True, append_images=images[1:])

# Verify that the number of frames in the gif equals the number of image files and durations
Image.open(gif_filepath).n_frames == len(images) == durations


IPdisplay.Image(url=gif_filepath)



