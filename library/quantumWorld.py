'''This is a library of functions that we've created over time in Che160 course
By importing this library of functions into any ipython notebook, you can save some
time by not having to write or copy and pastethe functions again

To import all of the functions in this library into a notebook, do the following:

from quantumWorld import *

or if you just want to import a single function

from quantumWorld import usefulFunctionName

'''
from IPython.display import HTML
from tempfile import NamedTemporaryFile
import numpy as np
from numpy.polynomial.hermite import hermval
from scipy import misc
from scipy.integrate import simps
import scipy.sparse as sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import HTML
from PyQuante.Constants import ang2bohr
import imolecule
from PyQuante import Ints
from PyQuante.Ints import getbasis
from PyQuante.Molecule import Molecule

from chemview import MolecularViewer
from chemlab.qc import molecular_orbital

# pyquante functions


def visualize_Mol(molecule, angstroms=True):
    mol = molecule.copy()
    # convert angstrom to bohr
    if angstroms:
        for atom in mol:
            coords = [a / ang2bohr for a in atom.pos()]
            atom.update_coords(coords)
    # create as xyz string
    xyz_str = mol.as_string()
    return imolecule.draw(xyz_str, format='xyz', shader="phong")


# DVR functions


def forwardcn(psi, A, Ad):
    '''
    This method takes one step forward using the crank nicholson propagator. As promised, it uses the sparse solver
    to find where A \psi(t + dt) = A^\dagger \psi(t)
    INPUTS:
     psi --> wavefunction vector
     A -> propagator operator
     Ad -> Adjoint of A
    '''
    psi = sparse.linalg.spsolve(A, Ad * psi)
    return psi


def sparse_V(x, vx, hbar, c):
    '''
    This method just returns a sparse diagonal matrix with the potential
    on the diagonals and a Identity matrix of the same size.
    INPUTS:
    x --> grid vector
    vx --> potential evaluated at the grid vector
    hbar -> planks constant
    c -> speed of light
    '''
    nx = len(x)
    k2 = (1j * c) / hbar

    V_diags = [0]
    V = k2 * sparse.spdiags(vx, V_diags, nx, nx)
    I = sparse.identity(nx)
    return V, I


def sparse_T(x, hbar, m, c):
    '''
    This method just returns the tridiagonal kinetic energy.
    It is the finite difference kinetic matrix we all know and love
    but it is incoded in a sparse matrix.
    NPUTS:
    x --> grid vector
    hbar -> planks constant
    m -> mass of expected particle
    c -> speed of light
    '''
    DX = x[1] - x[0]
    nx = len(x)
    prefactor = -(1j * hbar * c) / (2. * m)
    data = np.ones((3, nx))
    data[1] = -2 * data[1]
    diags = [-1, 0, 1]
    D2 = prefactor / DX**2 * sparse.spdiags(data, diags, nx, nx)
    return D2


def tcheby(x):
    '''Returns the kinectic operator T using chebychev polynomials
    INPUTS:
        x --> Grid position vector of size N
    OUTPUT:
        T --> value of cn at time t.
        KEfbr -->
        w -->
    '''

    # figure out info
    N = len(x)
    xmin = np.min(x)
    xmax = np.max(x)
    # start code
    delta = xmax - xmin
    w = np.zeros(N)
    KEfbr = np.zeros(N)
    T = np.zeros((N, N))
    # fill T
    for i, xp in enumerate(x):
        w[i] = delta / (N + 1.0)
        KEfbr[i] = ((i + 1.0) * np.pi / delta) ** 2
        for j in range(N):
            T[i, j] = np.sqrt(2.0 / (N + 1.0)) * \
                np.sin((i + 1.0) * (j + 1.0) * np.pi / (N + 1.0))

    return T, KEfbr, w


def dvr2fb(DVR, T):
    return np.dot(T, np.dot(DVR, T.T))


def fb2dvr(FBR, T):
    return np.dot(T.T, np.dot(FBR, T))


def Hmatrix_dvr(x, vx, hbar=1.0, m=1.0):
    '''Returns the Hamiltonian matrix built with DVR using chebychev polynomials.
        Can be used along with scipy.linalg.eigh() to solve numerically
        and get eigenvalues and eigenstates.
    INPUTS:
        x --> Grid position vector of size N
        vx --> Vector of potential function evaluated at x
        hbar --> (Optional, default=1) Value of plank constant, can vary
        if the equantions to solve are dimensionless or not.
        mass --> (Optional, default=1) Mass of your system

    OUTPUT:
        H --> An N x N matrix representing the hamiltonian operator.
    '''
    # build potential part V
    Vdvr = np.diag(vx)
    # build kinetic operator T
    T, KEfbr, w = tcheby(x)
    KEfbr = np.diag(KEfbr) * (hbar * hbar / 2 / m)
    KEdvr = fb2dvr(KEfbr, T)
    # Hamiltonian matrix
    H = KEdvr + Vdvr
    return H


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


def embedAnimation(anim, plt, frames=20):
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


# Time evolution of coefficient c_n
def cn_t_function(cn_0, t, E, hbar=1):
    '''this function evolves the coefficient cn associated to an
    eigenstate with energy E.
    INPUTS:
        cn_0 --> The value at t=0 of the coefficient
        t --> a numpy array of time values.
        E --> energy of the eigenstate associated to cn_0
    OUTPUT:
        cn_t --> value of cn at time t.
    '''
    exponent = -1j * E * t / hbar
    cn_t = cn_0 * np.exp(exponent)
    return cn_t

# Isotropic 2D harmonic oscillator


def harmonic_oscillator_2D(xx, yy, l, m, mass=1.0, omega=1.0, hbar=1.0):
    '''Returns the wavefunction for the 1D Harmonic Oscillator, given the following inputs:
    INPUTS:
        xx --> x-axis values for a 2D grid
        yy --> y-axis values for a 2D grid
        l --> l quantum number
        m --> m quantum number
        mass --> mass (defaults to atomic units)
        omega --> oscillator frequency, defaults to atomic units.
        hbar --> planck's constant divided by 2*pi
    '''
    # This is related to how the function np.polynomail.hermite.hermval
    # works.
    coeff_l = np.zeros((l + 1, ))
    coeff_l[l] = 1.0
    coeff_m = np.zeros((m + 1, ))
    coeff_m[m] = 1.0
    # Hermite polynomials required for the HO eigenfunctions
    hermite_l = np.polynomial.hermite.hermval(
        np.sqrt(mass * omega / hbar) * xx, coeff_l)
    hermite_m = np.polynomial.hermite.hermval(
        np.sqrt(mass * omega / hbar) * yy, coeff_m)
    # This is the prefactors in the expression for the HO eigenfucntions
    prefactor = (mass * omega / (np.pi * hbar)) ** (1.0 / 2.0) / \
        (np.sqrt(2 ** l * 2 ** m * misc.factorial(l) * misc.factorial(m)))
    # And the gaussians in the expression for the HO eigenfunctions
    gaussian = np.exp(-(mass * omega * (xx ** 2 + yy ** 2)) / (2.0 * hbar))
    # The eigenfunction is the product of all of the above.
    return prefactor * gaussian * hermite_l * hermite_m


def pib_momentum(p_array, L, n):
    '''return the momentum-space wave functions for the
    1D particle in a box.
        p_array --> numpy array of momentum values
        L --> size of box
        n --> quantum number
    '''
    prefactor = n * np.sqrt(L * np.pi)
    term = (1 - (-1) ** n * np.exp(-1j * p_array * L)) / \
        (n ** 2 * np.pi ** 2 - L ** 2 * p_array ** 2)
    psi_p = prefactor * term
    return psi_p


def build_H_matrix(x, V_x, m=1, h_bar=1):
    ''' this function builds the matrix representation of H,
    given x, the position array, and V_x as input
    '''
    a = x[
        1] - x[0]  # x is the dx of the grid.  We can get it by taking the diff of the first two
    # entries in x
    t = h_bar ** 2 / (2 * m * a ** 2)  # the parameter t, as defined by schrier

    # initialize H_matrix as a matrix of zeros, with appropriate size.
    H_matrix = np.zeros((len(V_x), len(V_x)))
    # Start adding the appropriate elements to the matrix
    for i in range(len(V_x)):
        # (ONE LINE)
        # Assignt to H_matrix[i][i],the diagonal elements of H
        # The appropriate values
        H_matrix[i][i] = 2 * t + V_x[i]
        #########
        # special case, first row of H
        if i == 0:
            # Assignt to H_matrix[i][i+1],the off-diagonal elements of H
            # The appropriate values, for the first row
            H_matrix[i][i + 1] = -t
        elif i == len(V_x) - 1:  # special case, last row of H
            H_matrix[i][i - 1] = -t
        else:  # for all the other rows
            # (TWO LINE)
            # Assignt to H_matrix[i][i+1], and H_matrix[i][i-1]
            # the off-diagonal elements of H, the appropriate value, -t
            H_matrix[i][i + 1] = -t
            H_matrix[i][i - 1] = -t
            ################
    return H_matrix


def normalize_wf(x, psi_x, dvr=False):
    '''this function normalizes a wave function
    Input -->
            x, numpy array of position vectors
            psi_x, numpy array representing wave function, same length as x
            dvr, boolean, while normalize differently if wavefunction is in dvr space
    Output:
            wf_norm --> normalized wave function
    '''
    #########
    # 1. Get integral_norm
    integral_norm = norm_wf(psi_x, x, dvr)
    # 2. normalize the wavefunction by dividing psi_x by the square root of integral norm.
    # Assign to wf_norm
    wf_norm = psi_x * np.sqrt(1.0 / integral_norm)
    ############
    return wf_norm


def norm_wf(psi_x, x, dvr=False):
    '''this function returns the norm of a wave function
    Input --> psi_x, numpy array representing wave function, same length as x
            x, numpy array of position vectors
            dvr, boolean, while normalize differently if wavefunction is in dvr space
    Output:
            values --> norm of a wave function
    '''
    integral_norm = 0.0
    if dvr:
        integral_norm = np.vdot(psi_x, psi_x)
    else:
        #########
        # 1. Get the pdf associated to psi_x, assign to pdf
        pdf = probabilityDensity(psi_x)
        # 2. Integrate the pdf over the entire range x.  Use simps and assign to
        # integral_norm
        integral_norm = simps(pdf, x)
        ############

    return integral_norm


def harmonic_oscillator_wf(x, n, m=1.0, omega=1.0, hbar=1.0):
    '''Returns the wavefunction for the 1D Harmonic Oscillator,
    given the following inputs:
    INPUTS:
        x --> a numpy array
        n --> quantum number, an intenger
        m --> mass (defaults to atomic units)
        omega --> oscillator frequency, defaults to atomic units.
        hbar --> planck's constant divided by 2*pi
    '''
    coeff = np.zeros((n + 1, ))
    coeff[n] = 1.0
    prefactor = 1.0 / (np.sqrt(2 ** n * misc.factorial(n))) * \
        (m * omega / (np.pi * hbar)) ** (1.0 / 4.0)
    gaussian = np.exp(-(m * omega * x * x) / (2.0 * hbar))
    hermite = np.polynomial.hermite.hermval(
        np.sqrt(m * omega / hbar) * x, coeff)
    return prefactor * gaussian * hermite


def harmonic_oscillator_V(x, m=1.0, omega=1.0, V_x0=0, x0=0):
    '''returns the potential for the 1D Harmonic Oscillator,
    given the following inputs:
    INPUTS:
        x --> a numpy array
        m --> mass, defaults to atomic units
        omega --> oscillator frequency, defaults to atomic units.
        V_x0 --> Lowest value of potential (shift in y - axis), defaults to 0
        x0 --> x value where potential has a minimum

    '''
    V_x = V_x0 + 1.0 / 2.0 * m * omega ** 2 * (x - x0) ** 2
    return V_x


def probabilityDensity(psi_x):
    ''' get probability density function associated to the wavefunction psi_x
    Input: psi_x --> an array, representing a values of a wavefunction
    '''
    prob = np.conjugate(psi_x) * psi_x
    return prob


def analytical_E_n_1D_PIB(n, L, h_bar=1, m=1):
    '''This function returns energy of the nth eigenstate
    of the 1D particle in a box.
    Input:
        -- n : quantum number specifying which eigenstate
        -- L, length of the box
    '''
    E_n = (n * h_bar * np.pi) ** 2 / (2.0 * m * L ** 2)
    return E_n


def numerical_second_derivative(x, psi_x):
    '''This python function uses a central difference approximation
    to get the second derivative of the function psi_x over the range x
    Input:
        -- x is an array of values
        -- psi_x is an array of values, corresponding to the wave
         function evaluated at x. (same length as x)
    '''
    dx = x[1] - x[0]  # this is delta x
    # an array of zeroes, same length as x.
    second_derivative = np.zeros_like(x)
    for i in range(len(x)):  # for each element in
        if i == 0:
            # forward differences for approximating the second derivative of
            # psi_x at the first value of x, x[0]
            second_derivative[i] = (
                psi_x[i + 2] - 2 * psi_x[i + 1] + psi_x[i]) / dx ** 2
        elif i == (len(x) - 1):
            # backwards differences for approximating the second derivative of
            # psi_x at the last value of x, x[-1]
            second_derivative[i] = (
                psi_x[i] - 2 * psi_x[i - 1] + psi_x[i - 2]) / dx ** 2
        else:
            # central differences for all other values of x
            second_derivative[i] = (
                psi_x[i + 1] - 2 * psi_x[i] + psi_x[i - 1]) / dx ** 2

    return second_derivative


def box_1D_eigenfunction(x, L, n):
    '''given x, L, and n returns an eigenfunction for the 1D particle in a box
    Inputs: x -- numpy array.
            L -- scalar, length of the box.
            n -- intenger
    '''
    psi_x = np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)
    return psi_x


def chem160_plotting(
        x, y, title='LABEL ME', legend_label=None,
        xlabel='LABEL ME', ylabel='LABEL ME'):
    '''
    It's not really important to understand the innerworkings of this function.
    Just know that this will be the
    general function that we'll use to plot during this semester.
     It has nice colours, as well as other defaults set.

    INPUT:
    x: An array or arrays to be plotted. These are the x axes to be plotted
    y: An array or arrays to be plotted. These are the y axes to be plotted
    title: String that defines the plot title.
    The default title is LABEL ME to remind you to always label your plots
    legend_label: A string or array of strings
    that define the legend entries to be used
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

    plt.grid(b=True, which='major', color='0.65', linestyle='-')


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
    xnew = x + v * dt + a(x) * dt ** 2 / 2
    vnew = v + (a(x) + a(xnew)) / 2 * dt
    return xnew, vnew


def ode_integrate(x0, v0, a, startTime=0.0, stopTime=7.0, dt=0.01, mass=1.0):
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
    # 2) By creating all of our arrays at once, we avoid any troubles with
    # memory that could complicate issues.
    xlist = np.zeros_like(t)
    vlist = np.zeros_like(t)

    # Here, we apply our initial conditions
    xlist[0] = x0
    vlist[0] = v0

    # We've set up a for loop that loops over the entire time array that we've defined above.
    # What this is saying is that it will perform the inside of the loop for each of the values of i
    # and i will range from 1 to the length of t, the time array
    for i in range(1, len(t)):
        xlist[i], vlist[i] = verlet(xlist[i - 1],
                                    vlist[i - 1],
                                    dt,
                                    a)
    return t, xlist, mass * vlist


def harmonic_oscillator_wf(x, n, m=1.0, omega=1.0, hbar=1.0):
    '''Returns the wavefunction for the 1D Harmonic Oscillator, given the following inputs:
    INPUTS:
        x --> a numpy array
        n --> quantum number, an intenger
        m --> mass (defaults to atomic units)
        omega --> oscillator frequency, defaults to atomic units.
        hbar --> planck's constant divided by 2*pi
    '''
    coeff = np.zeros((n + 1, ))
    coeff[n] = 1.0
    prefactor = 1.0 / (np.sqrt(2 ** n * misc.factorial(n))) * \
        (m * omega / (np.pi * hbar)) ** (1.0 / 4.0)
    gaussian = np.exp(-(m * omega * x * x) / (2.0 * hbar))
    hermite = np.polynomial.hermite.hermval(
        np.sqrt(m * omega / hbar) * x, coeff)
    return prefactor * gaussian * hermite


def harmonic_oscillator_V(x, m=1.0, omega=1.0, V_x0=0, x0=0):
    '''returns the potential for the 1D Harmonic Oscillator, given the following inputs:
    INPUTS:
        x --> a numpy array
        m --> mass, defaults to atomic units
        omega --> oscillator frequency, defaults to atomic units.
        V_x0 --> Lowest value of potential (shift in y - axis), defaults to 0
        x0 --> x value where potential has a minimum

    '''
    V_x = V_x0 + 1.0 / 2.0 * m * omega ** 2 * (x - x0) ** 2
    return V_x


def my_plotting_function(x, functions_list, labels_list, title='Plot', xlab='x', ylab='f(x)', fts=12, lw=2, fs=(10, 8)):
    plt.figure(figsize=fs)
    c = -1
    """"" DEFINE A FUNCTION THAT RECEIVE THE FOLLOWING INPUT:

    INPUTS (IN ORDER):
        - x: array with x values
        functions_list: list of functions you want to plot
        labels_list: list of labels. It should have the same size as functions_list
        title: title of the plot (Default: 'Plot')
        xlab: name of the xlabel (default: 'x')
        ylab: name of the ylabel (default: 'f(x)')
        fts: fontsize for legend, axes and labels (default: 12)
        lw: linewidth for the lines of the plot (default: 2)
        fs: figure size (default:(10,7))

    TO PLOT THE FUNCTIONS IN functions_list AS A FUNCTION OF x
    """""
    for f_x in functions_list:
        c += 1
        plt.plot(x, f_x, label=labels_list[c], linewidth=lw)
    plt.legend(loc='center left', fontsize=fts, bbox_to_anchor=(1, 0.5))
    plt.ylabel(ylab, fontsize=fts)
    plt.xlabel(xlab, fontsize=fts)
    plt.yticks(fontsize=fts)
    plt.xticks(fontsize=fts)
    plt.title(title, fontsize=fts)
    plt.show()
    return


def fancy_plotting(grid=False):
    """"" Load some fancy plot setting for matplotlib.
    You only have to load it once.

    INPUTS:
    grid (optional) --> a boolean True or False, indicating if you want a grid.

    """""
    # Define colors here
    dark_gray = ".15"
    light_gray = ".8"
    # color palete
    colors = [(0.89411765336990356, 0.10196078568696976, 0.10980392247438431),
              (0.21602460800432691, 0.49487120380588606, 0.71987698697576341),
              (0.30426760128900115, 0.68329106055054012, 0.29293349969620797),
              (0.60083047361934883, 0.30814303335021526, 0.63169552298153153),
              (1.0, 0.50591311045721465, 0.0031372549487095253),
              (0.99315647868549117, 0.9870049982678657, 0.19915417450315812),
              (0.65845446095747107, 0.34122261685483596, 0.1707958535236471),
              (0.95850826852461868, 0.50846600392285513, 0.74492888871361229),
              (0.60000002384185791, 0.60000002384185791, 0.60000002384185791)]

    style_dict = {
        "axes.color_cycle": colors,
        "figure.facecolor": "white",
        "text.color": dark_gray,
        "axes.labelcolor": dark_gray,
        "legend.frameon": False,
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": dark_gray,
        "ytick.color": dark_gray,
        "axes.axisbelow": True,
        "image.cmap": "YlGn",
        "font.family": ["sans-serif"],
        "font.sans-serif": ["Arial", "Liberation Sans",
                            "Bitstream Vera Sans", "sans-serif"],
        "grid.linestyle": "-",
        "lines.solid_capstyle": "round",
        'font.size': 18,
        'axes.titlesize': 'Large',
        'axes.labelsize': 'medium',
        'xtick.labelsize': 'medium',
        'ytick.labelsize': 'medium',
        'figure.figsize': (10, 5),
        "axes.facecolor": "white",
        "axes.edgecolor": dark_gray,
        "axes.linewidth": 2,
        "grid.color": light_gray,
        "legend.fancybox": True,
        "lines.linewidth": 2
    }

    if grid:
        style_dict.update({
            "axes.grid": True,
            "axes.facecolor": "white",
            "axes.edgecolor": light_gray,
            "axes.linewidth": 1,
            "grid.color": light_gray,
        })

    matplotlib.rcParams.update(style_dict)

    return


def anim_to_html(anim):
    VIDEO_TAG = """<video controls>
    <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>"""

    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")

    return VIDEO_TAG.format(anim._encoded_video)


def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))


def display_video(video_file):
    VIDEO_TAG = """<video controls>
    <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>"""

    return HTML(VIDEO_TAG.format(video_file))


def display_matrix(M):
    from IPython.display import Latex
    mat_str = "\\\\\n".join(
        [" & ".join(map('{0:.3f}'.format, line)) for line in M])
    Latex(r"""\[ \begin{bmatrix} %s \end{bmatrix} \] """ % mat_str)
    return Latex


def power2sym(powers):
    sym = 'Error'
    if powers in [(0, 0, 0)]:
        sym = 'S'
    if powers in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        sym = 'P'
    if powers in [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]:
        sym = 'D'
    if powers in [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0),
                  (1, 1, 1), (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)]:
        sym = 'F'

    return sym


def power2xyz(powers):
    xyz_str = ''
    xyz_types = ['x', 'y', 'z']
    for indx, i in enumerate(powers):
        if i > 0:
            if i == 1:
                xyz_str += xyz_types[indx]
            else:
                xyz_str += "%s^%d" % (xyz_types[indx], i)
    return xyz_str


def atno2type(atno):
    attype = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O',
              'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
    return attype[atno]


def get_orbitalInfo(mol, basis_set):

    bfs = getbasis(mol, basis_set)
    g_basis = [[] for i in mol.atoms]

    for i_bf in range(len(bfs)):
        bf = bfs[i_bf]
        cgbf = (power2sym(bf.powers), power2xyz(bf.powers), [])
        for prim in bf.prims:
            cgbf[2].append((prim.exp, prim.coef))
        g_basis[bf.atid].append(cgbf)

    return g_basis, len(bfs)


def get_orbitalList(mol, basis_set):

    bfs = getbasis(mol, basis_set)
    g_basis = [[] for i in mol.atoms]

    for i_bf in range(len(bfs)):
        bf = bfs[i_bf]
        cgbf = (power2sym(bf.powers),  [])
        for prim in bf.prims:
            cgbf[1].append((prim.exp, prim.coef))
        g_basis[bf.atid].append(cgbf)

    return g_basis


def print_orbitalInfo(mol, basis_set):
    """"" Print infomation on a basis set , like number of basis functions,
    symmetry type of the orbtials and gaussian decomposition of the orbitals.

    INPUTS:
    mol --> a Pyquante.Molecule object
    basis_set --> a string indicating the type of basis set
    """""

    g_basis, nbfs = get_orbitalInfo(mol, basis_set)

    print("Molecule is using %d basis functions" % (nbfs))
    for i, atom in enumerate(mol.atoms):
        print("Atom %s, #%d" % (atno2type(atom.atno), i + 1))
        for one_basis in g_basis[i]:
            print("\t Orbital type %s %s, built with %d gaussians:" %
                  (one_basis[0], one_basis[1], len(one_basis[2])))
            for one_g in one_basis[2]:
                print("\t \t exp = %5.5f , coef = %5.5f" %
                      (one_g[0], one_g[1]))
    return


def print_mo_overview(mol, orbe, orbs, basis_set):
    """"" Print infomation on a basis set , like number of basis functions,
    symmetry type of the orbtials and gaussian decomposition of the orbitals.

    INPUTS:
    mol --> a Pyquante.Molecule object
    basis_set --> a string indicating the type of basis set
    """""

    g_basis, nbfs = get_orbitalInfo(mol, basis_set)

    print("Molecule is using %d basis functions" % (nbfs))
    for i, atom in enumerate(mol.atoms):
        print("Atom %s, #%d" % (atno2type(atom.atno), i + 1))
        for one_basis in g_basis[i]:
            print("\t Orbital type %s %s, built with %d gaussians:" %
                  (one_basis[0], one_basis[1], len(one_basis[2])))
            for one_g in one_basis[2]:
                print("\t \t exp = %5.5f , coef = %5.5f" %
                      (one_g[0], one_g[1]))
    return


def view_orbital(mol,  basis_set, orbs, iso_level):
    mol_xyz = []
    mol_type = []
    for atom in mol.atoms:
        mol_type.append(atno2type(atom.atno))
        mol_xyz.append(atom.r)
    mol_xyz = np.array(mol_xyz)

    #mol_xyz -= np.mean(mol_xyz, axis=0)

    g_basis = get_orbitalList(mol, basis_set)
    #bfs = getbasis(mol, basis_set)

    #def f(x, y, z):
        #print(x, y, z)

     #   fxyz = np.zeros(x.shape)
       # for xx in x
      #  for ix, xx in enumerate(x[0, :, 0]):
        #    for iy, yy in enumerate(y[:, 0, 0]):
          #      for iz, zz in enumerate(z[0, 0, :]):
            #        fxyz[ix, iy, iz] = sum( orbs[ibf] * bfs[ibf].amp(xx, yy, zz) for ibf in range(len(bfs)))
        #fxyz = np.abs(fxyz)**2
        #return fxyz

    f = molecular_orbital(mol_xyz / 10, orbs, g_basis)

 #   def f(x, y, z, coords=coords, mocoeffs=mocoeffs, gbasis=gbasis):

    mv = MolecularViewer(mol_xyz/10 , {'atom_types': mol_type})
    mv.ball_and_sticks(ball_radius=0.02,)
    mv.add_isosurface(
        f, isolevel=iso_level, color=0xff0000, resolution=64, style='transparent')
    mv.add_isosurface(
        f, isolevel=iso_level*2, color=0xcc0011, resolution=64, style='transparent')
    mv.add_isosurface(
        f, isolevel=iso_level*4, color=0xaa0033, resolution=64, style='transparent')
    mv.add_isosurface(
        f, isolevel=iso_level*6, color=0x880055, resolution=64, style='transparent')
    mv.add_isosurface(
        f, isolevel=-iso_level, color=0x0000ff, resolution=64, style='transparent')
    mv.add_isosurface(
        f, isolevel=-iso_level*2, color=0x1100cc, resolution=64, style='transparent')
    mv.add_isosurface(
        f, isolevel=-iso_level*4, color=0x3300aa, resolution=64, style='transparent')
    mv.add_isosurface(
        f, isolevel=-iso_level*6, color=0x550088, resolution=64, style='transparent')
    return mv

if __name__ == "__main__":
    print("Load me as a module please")
