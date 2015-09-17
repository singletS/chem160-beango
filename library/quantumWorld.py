'''This is a library of functions that we've created over time in Che160 course
By importing this library of functions into any ipython notebook, you can save some
time by not having to write or copy and pastethe functions again

To import all of the functions in this library into a notebook, do the following:

from Chem160_library import *

or if you just want to import a single function

from Chem160_library import usefulFunctionName

'''
from IPython.display import HTML
from tempfile import NamedTemporaryFile
import numpy as np
from scipy import misc
from scipy.integrate import simps
from matplotlib import pyplot as plt
from IPython.display import HTML


def embedVideo(afile):
    '''This function returns a HTML embeded video of a file
    Input:
        -- afile : mp4 video file
    '''

    video = io.open(afile, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))



def embedAnimation(anim,plt,frames=20):
    '''This function returns a HTML embeded video of a maptlolib animation
    Input:
        -- anim : matplotlib animation
        -- plt, matplotlib module handle
    '''

    plt.close(anim._fig)
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=frames, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")

    VIDEO_TAG = """<video controls>
     <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
     Your browser does not support the video tag.
    </video>"""

    return HTML(VIDEO_TAG.format(anim._encoded_video))





def probabilityDensity(psi_x):
    ''' get probability density function associated to the wavefunction psi_x
    Input: psi_x --> an array, representing a values of a wavefunction
    '''
    prob = np.conjugate(psi_x)*psi_x
    return prob

def analytical_E_n_1D_PIB(n, L, h_bar=1, m=1):
    '''This function returns energy of the nth eigenstate
    of the 1D particle in a box.
    Input:
    	-- n : quantum number specifying which eigenstate
    	-- L, length of the box
    '''
    E_n = (n*h_bar*np.pi)**2 / (2.0*m*L**2)
    return E_n


def numerical_second_derivative(x, psi_x):
    '''This python function uses a central difference approximation
    to get the second derivative of the function psi_x over the range x
    Input:
        -- x is an array of values
        -- psi_x is an array of values, corresponding to the wave function evaluated at x. (same length as x)
    '''
    dx = x[1] - x[0] #this is delta x
    second_derivative = np.zeros_like(x) #an array of zeroes, same length as x.
    for i in range(len(x)): #for each element in
        if i==0:
            #forward differences for approximating the second derivative of psi_x at the first value of x, x[0]
            second_derivative[i] = ( psi_x[i+2]  - 2*psi_x[i+1] + psi_x[i] ) / dx**2
        elif i==(len(x)-1):
            #backwards differences for approximating the second derivative of psi_x at the last value of x, x[-1]
            second_derivative[i] = ( psi_x[i] - 2*psi_x[i-1] + psi_x[i-2] ) / dx**2
        else:
            #central differences for all other values of x
            second_derivative[i] = ( psi_x[i+1] - 2*psi_x[i] + psi_x[i-1] ) / dx**2

    return second_derivative


def box_1D_eigenfunction(x, L, n):
    '''given x, L, and n returns an eigenfunction for the 1D particle in a box
    Inputs: x -- numpy array.
            L -- scalar, length of the box.
            n -- intenger
    '''
    psi_x = np.sqrt(2.0/L) * np.sin(n*np.pi*x/L)
    return psi_x


def chem160_plotting(x, y, title = 'LABEL ME', legend_label = None, xlabel = 'LABEL ME', ylabel = 'LABEL ME'):
    '''
    It's not really important to understand the innerworkings of this function. Just know that this will be the
    general function that we'll use to plot during this semester. It has nice colours, as well as other defaults set.

    INPUT:
    x: An array or arrays to be plotted. These are the x axes to be plotted
    y: An array or arrays to be plotted. These are the y axes to be plotted
    title: String that defines the plot title. The default title is LABEL ME to remind you to always label your plots
    legend_label: A string or array of strings that define the legend entries to be used
    xlabel: A string that defines the xlabel. This can accept latex
    ylabel: A string that defines the ylabel. This can accept latex
    OUTPUT:
    None. A plot is displayed
    '''
    import prettyplotlib as ppl

    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 8)

    for ind in range(len(y)):
        if legend_label != None:
            ppl.plot(ax, x[ind], y[ind], label=legend_label[ind], linewidth=3)
        else:
            ppl.plot(ax, x[ind], y[ind], linewidth=3)

    ppl.legend(ax, fontsize=18)
    ax.set_title(title, fontsize=24)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(width=3)

    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(width=3)

    plt.grid(b=True, which='major', color='0.65',linestyle='-')

def verlet(x, v, dt, a):
    '''
    This is a simple implementation of the velocity verlet algorithm.
    INPUT
    x: scalar or vector of current positions
    v: scalar or vector of current velocities
    dt: scalar double of the current time step
    a: a function pointer to the acceleration function

    OUTPUT:
    xnew: scalar or vector of the updated positions. The data type (scalar or vector) will be the
          same as what is passed in to x, as the type will be infered.
    vnew: scalar of vector of the updated velocities. The data type (scalar or vector) will be the
          same as what is passed in to v, as the type will be infered.
    '''
    xnew = x + v*dt + a(x)*dt**2/2
    vnew = v + (a(x) + a(xnew))/2*dt
    return xnew, vnew


def ode_integrate(x0, v0, a, startTime = 0.0, stopTime = 7.0, dt = 0.01, mass = 1.0):
    '''
    This is the method that we created to stop the copying and pasting that we were doing to solve
    ODEs.
    INPUT
    x0 = scalar or vector of initial positions
    v0 = scalar or vector of initial velocities
    a = function pointer to the acceleration function. Note that this can only be position dependent
    startTime = optional argument, keyworded. Scalar that defines the starting point of the time array
    stopTime = optional argument, keyworded. Scalar that defines the ending point of the time array
    dt = optional argument, keyworded. Scalar that defines the time step of the time array
    mass = optional argument, keyworded. Scalar that defines the mass of the object
    OUTPUT
    t = vector of times
    xlist = vector of positions from the propagation
    vlist = vector of velocities from the propagation
    '''
    t = np.arange(startTime, stopTime, dt)

    # This creates a zeroed out array that's the shape of the time array. This is important for a few reasons
    # 1) We already know that we want to have collected a position and velocity at each time, t
    # 2) By creating all of our arrays at once, we avoid any troubles with memory that could complicate issues.
    xlist = np.zeros_like(t)
    vlist = np.zeros_like(t)

    # Here, we apply our initial conditions
    xlist[0] = x0
    vlist[0] = v0

    # We've set up a for loop that loops over the entire time array that we've defined above.
    # What this is saying is that it will perform the inside of the loop for each of the values of i
    # and i will range from 1 to the length of t, the time array
    for i in range(1, len(t)):
        xlist[i], vlist[i] = verlet(xlist[i-1],
                                    vlist[i-1],
                                    dt,
                                    a)
    return t, xlist, mass*vlist

def harmonic_oscillator_wf(x, n, m = 1.0, omega = 1.0, hbar = 1.0):
    '''Returns the wavefunction for the 1D Harmonic Oscillator, given the following inputs:
    INPUTS:
        x --> a numpy array
        n --> quantum number, an intenger
        m --> mass (defaults to atomic units)
        omega --> oscillator frequency, defaults to atomic units.
        hbar --> planck's constant divided by 2*pi
    '''
    coeff = np.zeros((n+1, ))
    coeff[n] = 1.0
    prefactor = 1.0/(np.sqrt(2**n*misc.factorial(n)))*(m*omega/(np.pi*hbar))**(1.0/4.0)
    gaussian = np.exp(-(m*omega*x*x)/(2.0*hbar))
    hermite = np.polynomial.hermite.hermval(np.sqrt(m*omega/hbar)*x, coeff)
    return prefactor*gaussian*hermite

def harmonic_oscillator_V(x, m = 1.0, omega = 1.0, V_x0 = 0, x0 = 0):
    '''returns the potential for the 1D Harmonic Oscillator, given the following inputs:
    INPUTS:
        x --> a numpy array
        m --> mass, defaults to atomic units
        omega --> oscillator frequency, defaults to atomic units.
        V_x0 --> Lowest value of potential (shift in y - axis), defaults to 0
        x0 --> x value where potential has a minimum

    '''
    V_x = V_x0 + 1.0/2.0 *m*omega**2*(x-x0)**2
    return V_x

if __name__ == "__main__":
    print("Load me as a module please")

