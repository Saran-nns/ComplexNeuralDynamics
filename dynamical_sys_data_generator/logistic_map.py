import numpy as np
import matplotlib.pyplot as plt
import IPython.display as IPdisplay
from mpl_toolkits.mplot3d import axes3d, Axes3D
from PIL import Image
import os
import glob


# Initial condition

initial_step = 0.1

# Define the system parameter(s): r
r = 4.

class LogisticMap(object):

    def __init__(self,x,r,animate,step_size):
        
        super().__init__(self)
        self.x = x
        self.r = r
        self.animate = animate
        self.step_size = step_size

        # Traget folder to save the result
        self.save_folder = 'images/logmap-animate'
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)


        # Get incrementally larger chunks of the time points, to reveal the attractor one frame at a time
        self.chunks = self.get_chunks()

    # Return a list in iteratively larger chunks

    def get_chunks(self):
        size = max(1, self.step_size)
        chunks = [self.x[0:i] for i in range(1, len(self.x) + 1, self.step_size)]
        return chunks
        
    # Define the logistic map function
    
    @staticmethod
    def logistic(r, x):
        return r * x * (1 - x)
    
    def logistic_map(self,r, x):
        """
        Args: 
        x - List of ratio of existing population to the maximum population
        r - Parameter of interest 
        """
        # Take the last point of x and measure the y

        y = self.logistic(r,x[-1])
            
        return np.array([x[-1],y])

    # Plot the system in 2 dimensions
    def plot_cobweb_diagram(self,points, n):
        
        fig = plt.figure(figsize=(12, 9))
        ax= fig.gca()

        # Plot the function

        ax.plot(self.x[:n*self.step_size +1], self.logistic(r, self.x[:n*self.step_size+1]), 'k', lw=2)
        
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
            
        plt.savefig('{}/{:03d}.png'.format(self.save_folder,n), dpi=60, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # Get the points to plot, one chunk of time steps at a time, by solving the logistic func

    def plot_logmap(self):

        # Get the points
        self.points = []
        for i in range(len(time_points)):
                self.points.append([self.logistic_map(r, chunk) for chunk in self.chunks[:i+1]])
        #
        for n,point in enumerate(self.points):
                self.plot_cobweb_diagram(point, n)

# Animate it
# Create an animated gif of all the plots then display it inline

# Create a tuple of display durations, one for each frame
first_last = 300 #show the first and last frames for 100 ms
standard_duration = 100 #show all other frames for 50 ms
durations = tuple([first_last] + [standard_duration] * (len(self.points) - 2) + [first_last])

# durations =  len(os.listdir(os.getcwd()+'/images/bifurcation-diagram-animate'))

# Load all the static images into a list
images = [Image.open(image) for image in glob.glob('{}/*.png'.format(self.save_folder))]
gif_filepath = 'images/log_map_cobweb_diag.gif'

# Save as an animated gif
gif = images[0]
gif.info['duration'] = durations #ms per frame
gif.info['loop'] = 0 #how many times to loop (0=infinite)
gif.save(fp=gif_filepath, format='gif', save_all=True, append_images=images[1:])

# Verify that the number of frames in the gif equals the number of image files and durations
Image.open(gif_filepath).n_frames == len(images) == durations


IPdisplay.Image(url=gif_filepath)

def poincare_3dplot(y):

  
    poincore = []
    for i in range(4):
        poincore.append(x[i:i+3])
    p = np.asarray(poincore[:-2])
    
    print(p)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax = plt.axes(projection='3d')
    ax.scatter3D(p[:,0],p[:,1],p[:,2])
    plt.show()

def poincare_2dplot(x):

    poincore = []
    for i in range(4):
        poincore.append(x[i:i+2])
    p = np.asarray(poincore[:-1])
    
    plt.scatter(p[:,0],p[:,1],p[:,2])
    plt.show()

