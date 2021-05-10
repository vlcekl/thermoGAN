from itertools import combinations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import cycler

def plot_ternary(data, labels, classes = None, class_names = None, size=10, plt_title = None, legend_title = None):
    
    color = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler(color=color)
    
    if legend_title is None:
        legend_title = "System, Temperature"

    # If no classes selected, get all
    if classes is None:
        classes = np.unique(labels)

    triplets = list(combinations(range(data.shape[1]), 3))

    colswitch = {1: 1, 2: 2, 3: 2, 4: 2}
    
    with plt.style.context(('seaborn-white')):
        size_ax = size
        size_x = size_ax
        size_y = size*0.75**0.5
        
        ncols = colswitch.get(len(triplets), 3)
        nrows = 1 + (len(triplets)-1)//ncols
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size_x*ncols, size_y*nrows))
        if plt_title is not None:
            fig.suptitle(plt_title, fontsize = 20)
            
        axes = np.ravel(axes)
        
        for ia, (i, j, k) in enumerate(triplets):
            # Make frame (always same)
            frame_x = np.array([0.0, 0.5*size_x/size, size_x/size])
            frame_y = np.array([0.0, size_y/size, 0.0])
            frame_triangle = np.array([[0, 1, 2]])
            frame_triang = tri.Triangulation(frame_x, frame_y, frame_triangle)

            axes[ia].triplot(frame_triang, 'k-')
            axes[ia].plot([frame_x[1], frame_x[1]], [frame_y[1]/3, 0.0], 'k--')
            axes[ia].plot([frame_x[1], frame_x[1]/2], [frame_y[1]/3, frame_y[1]/2], 'k--')
            axes[ia].plot([frame_x[1], (frame_x[1]+frame_x[2])/2], [frame_y[1]/3, (frame_y[1]+frame_y[2])/2], 'k--')
            axes[ia].axis('off')
        
            # translate the data to cartesian coords for each projection
            axes[ia].set_title(f'Projection: {i}-{j}-{k}', fontsize='xx-large')
            for cl in classes:
                idx = np.where(labels == cl)
                a = data[idx, i]
                b = data[idx, j]
                c = data[idx, k]
                x = 0.5 * ( 2.*b+c ) / ( a+b+c )*1
                y = 0.5*np.sqrt(3) * c / (a+b+c)*1
                axes[ia].scatter(x, y, s = 1, label = f'{class_names[cl]}')
            lgnd = axes[ia].legend(fontsize='x-large',
                                   title=legend_title, title_fontsize='x-large',
                                   handletextpad=0.1)
            for handle in lgnd.legendHandles:
                handle.set_sizes([40.0])

        # erase unused subplots
        for ia in range(len(triplets), nrows*ncols):
            axes[ia].axis('off')



def plot_ternary_continuum(data, labels, size=10, plt_title = None, legend_title = None):
    
    #color = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    #mpl.rcParams['axes.prop_cycle'] = cycler(color=color)
    #mpl.rcParams['axes.prop_cycle'] = cycler(color='bg')
    
    if legend_title is None:
        legend_title = "System, Temperature"

    triplets = list(combinations(range(data.shape[1]), 3))

    colswitch = {1: 1, 2: 2, 3: 2, 4: 2}
    
    with plt.style.context(('seaborn-white')):
        size_ax = size
        size_x = size_ax
        size_y = size*0.75**0.5
        
        ncols = colswitch.get(len(triplets), 3)
        nrows = 1 + (len(triplets)-1)//ncols
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size_x*ncols, size_y*nrows))
        
        if plt_title is not None:
            fig.suptitle(plt_title, fontsize = 20)

        axes = np.ravel(axes)
        
        for ia, (i, j, k) in enumerate(triplets):
            # Make frame (always same)
            frame_x = np.array([0.0, 0.5*size_x/size, size_x/size])
            frame_y = np.array([0.0, size_y/size, 0.0])
            frame_triangle = np.array([[0, 1, 2]])
            frame_triang = tri.Triangulation(frame_x, frame_y, frame_triangle)

            axes[ia].triplot(frame_triang, 'k-')
            axes[ia].plot([frame_x[1], frame_x[1]], [frame_y[1]/3, 0.0], 'k--')
            axes[ia].plot([frame_x[1], frame_x[1]/2], [frame_y[1]/3, frame_y[1]/2], 'k--')
            axes[ia].plot([frame_x[1], (frame_x[1]+frame_x[2])/2], [frame_y[1]/3, (frame_y[1]+frame_y[2])/2], 'k--')
            axes[ia].axis('off')
        
            # translate the data to cartesian coords for each projection
            axes[ia].set_title(f'Projection: {i}-{j}-{k}', fontsize='xx-large')

            a = data[:, i]
            b = data[:, j]
            c = data[:, k]
            x = 0.5 * ( 2.*b+c ) / ( a+b+c )*1
            y = 0.5*np.sqrt(3) * c / (a+b+c)*1
            axes[ia].scatter(x, y, c=labels, cmap='RdYlBu',s = 1)

        # erase unused subplots
        for ia in range(len(triplets), nrows*ncols):
            axes[ia].axis('off')


def plot_spherical(data, labels, psize=2):
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi/2:100j, 0.0:pi/2:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    
    #Import data
    xx, yy, zz = np.hsplit(data[:,:3], 3) 
    
    #Set colours and render
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter([0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], c='k')
    ax.plot([0, 0], [0, 0], [0, 1], c='k')
    ax.plot([0, 0], [0, 1], [0, 0], c='k')
    ax.plot([0, 1], [0, 0], [0, 0], c='k')
    
    #ax.plot_surface(x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)
    ax.scatter(x, y, z, color='k', alpha=0.2, s=0.1)
    
    # Plot vectors on the sphere surface
    ax.scatter(xx, yy, zz, c=labels, s=psize)
    
    # Plot class centroids
    cents = []
    for c in np.unique(labels):
        index = np.where(labels == c)
        xc = np.sqrt(np.mean(data[index, 0]**2))
        yc = np.sqrt(np.mean(data[index, 1]**2))
        zc = np.sqrt(np.mean(data[index, 2]**2))
        cents.append([xc, yc, zc, c])
    cents = np.array(cents)*1.02
    ax.scatter(cents[:,0], cents[:,1], cents[:,2], color="red", s=20)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
    

def plot_tetrahedron(data_out, labels_out):

    #Import data
    data = data_out[:,:4]**2
    a = data[:, 0]
    b = data[:, 1]
    c = data[:, 2]
    xx = 0.5 * ( 2.*b+c ) / ( a+b+c )*1
    yy = 0.5*np.sqrt(3) * c / (a+b+c)*1
    zz = data[:, 3]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    #ax.scatter([0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], c='k')
    ax.plot([0, 1], [0, 0], [0, 0], c='k')
    ax.plot([0, 0.5], [0, np.sqrt(1 - 0.5**2)], [0, 0], c='k')
    ax.plot([1, 0.5], [0, np.sqrt(1 - 0.5**2)], [0, 0], c='k')

    # Plot vectors on the sphere surface
    ax.scatter(xx, yy, zz, c=labels_out, s=2)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
