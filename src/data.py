"""
Read data from file and convert it to other formats
"""

import re

from ase.io import read, write

class Data:
    """
    """

    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.trajectory = None

    def read_trajectory(self, file_in):
        """
        """
        format_in = 'dlp-history'
        self.trajectory = read(self.data_directory + file_in,
                               format=format_in, index=':')

    def write_xyz(self, file_out):
        """
        """
        format_out = 'extxyz'

        for i, config in enumerate(self.trajectory):
            config.wrap()
            write(self.data_directory + file_out.format(i),
                  config,
                  format=format_out,
                  columns=['symbols', 'positions'])

    def write_cp2k(self, file_name, file_xyz, n=None):
        """
        """
        with open(self.data_directory + file_name.format('template')) as f_template:
            template_text = f_template.read()
            if n is not None:
                n = min(n, len(self.trajectory))
            else:
                n = len(self.trajectory)

            for i in range(n):
                with open(self.data_directory + file_name.format('template'), 'w') as f:
                    f.write(template_text.format(self.data_directory + file_xyz.format(i), i))

    def write_batch(self, file_batch, n=None):
        """
        """
        with open(self.data_directory + file_batch.format('template')) as f_template:
            template_text = f_template.read()
            if n is not None:
                n = min(n, len(self.trajectory))
            else:
                n = len(self.trajectory)

            for i in range(n):
                with open(self.data_directory + file_batch.format(i), 'w') as f:
                    f.write(template_text.format(i))

    def write_n2p2(self):
        """
        begin
        comment <comment>
        lattice <ax> <ay> <az>
        lattice <bx> <by> <bz>
        lattice <cx> <cy> <cz>
        atom <x1> <y1> <z1> <e1> <c1> <n1> <fx1> <fy1> <fz1>
        atom <x2> <y2> <z2> <e2> <c2> <n2> <fx2> <fy2> <fz2>
        ...
        atom <xn> <yn> <zn> <en> <cn> <nn> <fxn> <fyn> <fzn>
        energy <energy>
        charge <charge>
        end
        """
        n = 1
        f_xyz = '/home/vol00/scarf860/cc_placement/m_cresol/data/config/extxyz_32_{}'
        f_log = '/home/vol00/scarf860/cc_placement/m_cresol/cp2k/100_32/cresol_32_{}.log'
        f_forces = '/home/vol00/scarf860/cc_placement/m_cresol/cp2k/100_32/m-cresol-forces_{}.xyz'
        f_input = '/home/vol00/scarf860/cc_placement/m_cresol/n2p2/input.data'

        text = ''
        for i in range(n):
            with open(f_xyz) as f:
                xyz_lines = f.readlines()
                n_atoms = int(xyz_lines[0].strip())
                header_list = xyz_lines[0].split('"')
                lattice_list = header_list[1].split()

            with open(f_forces) as f:
                force_lines = f.readlines()

            with open(f_log) as f:
                energy = None
                for line in f.readlines():
                    if re.search('^ ENERGY|'):
                        energy = line.split()[-1]
                
            if energy is None:
                raise ValueError('Energy not found in {}'.format(f_log))

            text += 'begin\n'
            text += 'comment config_index={}\n'.format(i)
            text += 'lattice {0} {1} {2}\n'.format(lattice_list[0], lattice_list[1], lattice_list[2])
            text += 'lattice {0} {1} {2}\n'.format(lattice_list[3], lattice_list[4], lattice_list[5])
            text += 'lattice {0} {1} {2}\n'.format(lattice_list[6], lattice_list[7], lattice_list[8])
            for j in range(n_atoms):
                atom_xyz = xyz_lines[j + 2].split()
                force = force_lines[j + 4].split()[-3:]
                text += 'atom {1} {2} {3} {0} 0.0 0.0 {4} {5} {6}\n'.format(*atom_xyz + force)

            text += 'energy {}\n'
            text += 'charge 0.0\n'
            text += 'end\n'

        with open(f_input, 'w') as f:
            f.write(text)
