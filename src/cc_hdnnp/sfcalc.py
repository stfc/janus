import numpy as np
import pandas as pd

class SymFuncCalculator():
    """
    Class for calculating symmetry functions based on parameters
    stored in an input.nn file.

    Parameters
    ----------
    filename: str
        input.nn filename containing parameters defining the symmetry function
    num_coords: int = 100
        The number of coordinates to be plotted. Default is 100
    """
    def __init__(
        self,
        filename: str,
        num_coords: int = 100,
    ):
        elements = []
        type = []
        eta = []
        r_shift = []
        r_cutoff = []
        lmbda = []
        zeta = []
        cutoff_type = 0
        with open(filename, "r") as file:
            for line in file.readlines():
                if (line[0] != '#'):
                    if ("cutoff_type" in line):
                        cutoff_type = float(line.split()[1])
                        alpha = float(line.split()[2])
                    if("symfunction_short" in line):
                        if(int(line.split()[2]) == 2):
                            elements.append([line.split()[1], line.split()[3]])
                            type.append(2)
                            eta.append(float(line.split()[4]))
                            r_shift.append(float(line.split()[5]))
                            r_cutoff.append(float(line.split()[6]))
                            lmbda.append( 0)
                            zeta.append(0)
                        if(int(line.split()[2]) == 3 or int(line.split()[2]) == 9):
                            elements.append([line.split()[1], line.split()[3],  line.split()[4]])
                            type.append(int(line.split()[2]))
                            eta.append(float(line.split()[5]))
                            lmbda.append(float(line.split()[6]))
                            zeta.append(float(line.split()[7]))
                            r_shift.append(float(line.split()[9]))
                            r_cutoff.append(float(line.split()[8]))
        self.params = pd.DataFrame({
            'elements': elements,
            'type': type,
            'eta': eta,
            'r_shift': r_shift,
            'r_cutoff': r_cutoff,
            'lambda': lmbda,
            'zeta': zeta,
            'alpha': alpha * np.ones_like(type),
            'cutoff_type': cutoff_type * np.ones_like(type),
        })
        self.r = np.linspace(self.params['r_cutoff'].max(), 0, num_coords)
        self.theta = np.linspace(0, 2 * np.pi, num_coords)


    def _cutoff_function(
        self,
        i: int,
        r: np.ndarray = None,
    ):
        """
        Evaluate the cutoff function for a set of r values.
        Currently, only n2p2's CT_POLY2 can be evaluated.

        Parameters
        ----------
        i: int
            Index of symmetry function to be evaluated
        r: np.ndarray = None,
            r values at which to evaluate the cutoff function. Default is None
        """
        r_c = self.params.iloc[i]['r_cutoff']
        a = self.params.iloc[i]['alpha']
        if r is None:
            r = self.r
        r_ci  = r_c * a
        x = (r - r_ci ) /  (r_c - r_ci )
        if self.params.iloc[i]['cutoff_type'] == 6:
            return (
                ((((15 - 6 * x) * x - 10) * (x ** 3) + 1) * (r < r_c) * (r >= r_ci))
                + ((r < r_ci )* (r >= 0))
            )
        else:
            return np.nan * np.ones_like(r)


    def radial_sym_func(
        self,
        i: int,
        r: np.ndarray = None,
    ):
        """
        Evaluate the radial component of symmetry functions.
        Currently, the functional form of type 2, 3 and 9 symmetry functions can
        be evaluated. Type 3 and 9 symmetry functions require multiple calls to
        this function for their complete radial dependency.

        Parameters
        ----------
        i: int
            Index of symmetry function to be evaluated
        r: np.ndarray = None,
            r values at which to evaluate the symmetry function. Default is None
        """
        eta = self.params.iloc[i]['eta']
        r_s = self.params.iloc[i]['r_shift']
        if r is None:
            r = self.r
        f_c = self._cutoff_function(i, r)
        if (
            self.params.iloc[i]['type'] == 2 or
            self.params.iloc[i]['type'] == 3 or
            self.params.iloc[i]['type'] == 9
        ):
            return np.exp(-eta * ((r - r_s)** 2)) * f_c
        else:
            return np.nan * np.ones_like(r)


    def angular_sym_func(
        self,
        i: int,
        theta: np.ndarray = None,
    ):
        """
        Evaluate the angular component of symmetry functions.
        Currently, the functional form of type 3 and 9 symmetry functions
        can be evaluated.

        Parameters
        ----------
        i: int
            Index of symmetry function to be evaluated
        theta: np.ndarray = None,
            theta values at which to evaluate the symmetry function. Default is None
        """
        zeta = self.params.iloc[i]['zeta']
        lmbda = self.params.iloc[i]['lambda']
        if theta is None:
            theta = self.theta

        if self.params.iloc[i]['type'] == 3 or self.params.iloc[i]['type'] == 9:
            return  (
                (2 ** (1 - zeta)) * ((1 + lmbda * np.cos(theta)) ** zeta)
            )
        else:
            return np.nan * np.ones_like(theta)


    def full_sym_func(
        self,
        i: int,
        r_ij: np.ndarray = None,
        r_jk: np.ndarray = 1.,
        theta: np.ndarray = None,
    ):
        """
        Evaluate the full symmetry functions. Currently, type 2, 3, and 9
        functions have been implemented.

        Parameters
        ----------
        i: int
            Index of symmetry function to be evaluated
        r_ij: np.ndarray = None,
            r_ij values at which to evaluate the symmetry function. Default is None
        r_jk: np.ndarray = None,
            r_jk values at which to evaluate the symmetry function. Default is 1.
        theta: np.ndarray = None,
            theta values at which to evaluate the symmetry function. Default is None
        """
        if theta is None:
            theta = self.theta
        if r_ij is None:
            r_ij = self.r

        if self.params.iloc[i]['type'] == 2:
            return (self.radial_sym_func(i, r_ij))

        elif self.params.iloc[i]['type'] == 3:
            r_ik = ((r_ij ** 2.) + (r_jk ** 2.) - (2. * r_ij * r_jk * np.cos(theta))) ** 0.5
            return (
                self.angular_sym_func(i, theta) * self.radial_sym_func(i, r_ij) *
                self.radial_sym_func(i, r_jk) * self.radial_sym_func(i, r_ik)
            )

        elif self.params.iloc[i]['type'] == 9:
            return (
                self.angular_sym_func(i, theta) * self.radial_sym_func(i, r_ij) *
                self.radial_sym_func(i, r_jk)
            )
        else:
            return np.nan * np.ones_like(r_ij)
