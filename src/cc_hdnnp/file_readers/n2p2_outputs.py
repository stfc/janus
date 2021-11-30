"""
Reads n2p2 train/test points/forces files for information used in the workflow.
"""
import numpy as np


def read_energies(filename: str) -> np.ndarray:
    """
    Read the energies predicted by the network from the n2p2 output file provided.

    Parameters
    ----------
    filename : str
        The file path of the file to read energies from. Should be in a recognisable
        format, namely the "trainpoints" file generated by the network training.

    Returns
    -------
    np.ndarray
        Two dimensional array with shape (N, 2) where N is the number of structures the
        network was evaluated for. Along the second dimension the first element is the
        index of the structure, and the second is the energy associated with it.
    """
    with open(filename) as f:
        # readline "pops" the first line so all indexes should decrease by 1
        line = f.readline().strip()
        if line.startswith("point"):
            energies = np.array(
                [line.strip().split()[2] for line in f.readlines()]
            ).astype(float)
            energies = np.dstack((np.arange(len(energies)), energies))[0]

        elif line.startswith("Conf."):
            energies = np.array(
                [np.array(line.strip().split())[[1, 3]] for line in f.readlines()]
            ).astype(float)
            energies = energies[:, 1] / energies[:, 0]
            energies = np.dstack((np.arange(len(energies)), energies))[0]

        elif line.startswith("###"):
            energies = np.array(
                [line.strip().split()[-1] for line in f.readlines()[11:]]
            ).astype(float)
            energies = np.dstack((np.arange(len(energies)), energies))[0]

        else:
            raise IOError("Unknown RuNNer format")

    return energies


def read_forces(filename: str) -> np.ndarray:
    """
    Read the forces predicted by the network from the n2p2 output file provided.

    Parameters
    ----------
    filename : str
        The file path of the file to read forces from. Should be in a recognisable
        format, namely the "trainforces" file generated by the network training.

    Returns
    -------
    np.ndarray
        Two dimensional array with shape (3N, 2) where N is the number of structures the
        network was evaluated for. Along the second dimension the first element is the
        index of the structure, and the second is a force associated with it in one of the
        cartesian directions. It is ordered so that the forces associated with index `i`
        appear consecutively, in xyz order:
        [... [i-1, fz(i-1)], [i, fx(i)], [i, fy(i)], [i, fz(i)], [i+1, fx(i+1)], ...]
    """
    with open(filename) as f:
        line = f.readline().strip()
        if line.startswith("point"):
            forces = np.array(
                [np.array(line.strip().split())[[0, 4]] for line in f if line.strip()]
            ).astype(float)
            forces[:, 0] -= 1

        elif line.startswith("Conf."):
            forces = np.array(
                [
                    np.array(line.strip().split())[[0, 5, 6, 7]]
                    for line in f
                    if line.strip()
                ]
            ).astype(float)
            forces = np.concatenate(
                (forces[:, [0, 1]], forces[:, [0, 2]], forces[:, [0, 3]])
            )
            forces[:, 0] -= 1

        elif line.startswith("###"):
            forces = []
            for line in f.readlines()[12:]:
                text = line.strip().split()
                forces.append([text[0], text[-1]])
            forces = np.array(forces).astype(float)

        else:
            raise IOError("Unknown RuNNer format")

    return forces
