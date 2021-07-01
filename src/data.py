"""
Read data from file and convert it to other formats so it can be used as input
for the next stage of the pipeline.

TODO set up the import properly so we don't have to manually run this:

import sys
sys.path.append('/home/vol00/scarf860/cc_placement/CC_HDNNP/src')

"""

from os.path import isfile
import re

from ase.io import read, write
from scipy.constants import physical_constants

from sfparamgen import SymFuncParamGenerator

class Data:
    """
    Holds information relevant to reading and writing data from and to file.

    Parameters
    ----------
    data_directory : str
        All file names passed to other function will be appended to
        `data_directory`
    """

    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.trajectory = None

    def read_trajectory(self, file_trajectory: str):
        """
        Reads `file_trajectory` and stores the trajectory object.

        Parameters
        ----------
        file_trajectory : str
            File containing a trajectory of atom configurations

        Examples
        --------
        from data import Data
        d = Data('/home/vol00/scarf860/cc_placement/CC_HDNNP/examples/m_cresol/')
        d.read_trajectory('m-cresol-npt-360K.history')
        """
        format_in = 'dlp-history'
        self.trajectory = read(self.data_directory + file_trajectory,
                               format=format_in, index=':')

    def write_xyz(self, file_xyz: str):
        """
        Writes a loaded trajectory to file as a series of xyz files.

        Parameters
        ----------
        file_xyz : str
            File name to write the xyz to. Will be formatted with the frame
            number, so should contain `'{}'` as part of the string.

        Examples
        --------
        d.write_xyz('xyz_360K/{}.xyz')
        """
        format_out = 'extxyz'

        for i, config in enumerate(self.trajectory):
            config.wrap()
            write(self.data_directory + file_xyz.format(i),
                  config,
                  format=format_out,
                  columns=['symbols', 'positions'])

    def _min_n_config(self, n_provided: int):
        """
        Utility function to ensure we don't attempt to read/write more frames
        than were in the trajectory originally.

        Parameters
        ----------
        n_provided : int
            Number of frames specified by the user.

        Returns
        -------
        int
            The minimum of `n_provided` and the length of `self.trajectory` if
            it exists.
        """
        if n_provided is not None and self.trajectory is not None:
            n_config = min(n_provided, len(self.trajectory))
        elif n_provided is not None:
            n_config = n_provided
        elif self.trajectory is not None:
            n_config = len(self.trajectory)
        else:
            n_config = 0

        return n_config

    def write_cp2k(self, file_batch: str, file_input: str, file_xyz: str,
                   n_config: int=None, **kwargs):
        """
        Writes .inp files and batch scripts for running cp2k from `n_config`
        .xyz files. Can set supported settings using `**kwargs`, in which case
        template file(s) will be formatted to contain the values provided.

        Parameters
        ----------
        file_batch : str
            File name to write the batch scripts to. Will be formatted with the
            frame number and any other `**kwargs`, so should contain '{}' as
            part of the string. There should already be a template of this file
            with 'template' instead of '{}' containing the details of the file
            that do not need formatting.
        file_input : str
            File name to write the cp2k input files to. Will be formatted with the
            frame number and any other `**kwargs`, so should contain '{}' as
            part of the string. There should already be a template of this file
            with 'template' instead of '{}' containing the details of the file
            that do not need formatting.
        file_xyz : str
            File name to read the xyz files from. Will be formatted with the
            frame number, so should contain '{}' as part of the string.
        n_config: int, optional
            The number of configuration frames to use. If the number provided
            is greater than the length of `self.trajectory`, or `n_config` is
            `None` then the length is used instead. Default is `None`.
        **kwargs:
          - cutoff: tuple of float
          - relcutoff: tuple of float

        Examples
        --------
from data import Data
d = Data('/home/vol00/scarf860/cc_placement/CC_HDNNP/examples/m_cresol/')
d.write_cp2k(file_batch='360K_frames/scripts/cp2k_batch_{}.bash',
                file_input='360K_frames/cp2k_input/cresol_{}.inp',
                file_xyz='xyz_360K/{}.xyz',
                n_config=101,
                cutoff=(600,),
                relcutoff=(60,))
        """
        with open(self.data_directory + file_input.format('template')) as f_template:
            input_template = f_template.read()

        with open(self.data_directory + file_batch.format('template')) as f_template:
            batch_template = f_template.read()

        n_config = self._min_n_config(n_config)
        file_id_template = 'n_{i}'

        if 'cutoff' in kwargs:
            file_id_template += '_cutoff_{cutoff}'
            cutoff_values = kwargs['cutoff']
        else:
            cutoff_values = [None]

        if 'relcutoff' in kwargs:
            file_id_template += '_relcutoff_{relcutoff}'
            relcutoff_values = kwargs['relcutoff']
        else:
            relcutoff_values = [None]

        for i in range(n_config):
            format_dict = {'i': i}
            with open(self.data_directory + file_xyz.format(i)) as f:
                header_line = f.readlines()[1]
                lattice_string = header_line.split('"')[1]
                lattice_list = lattice_string.split()
                format_dict['cell_x'] = ' '.join(lattice_list[0:3])
                format_dict['cell_y'] = ' '.join(lattice_list[3:6])
                format_dict['cell_z'] = ' '.join(lattice_list[6:9])

            for cutoff in cutoff_values:
                for relcutoff in relcutoff_values:
                    format_dict['cutoff'] = cutoff
                    format_dict['relcutoff'] = relcutoff
                    format_dict['file_xyz'] = self.data_directory + file_xyz.format(i)
                    file_id = file_id_template.format(**format_dict)
                    with open(self.data_directory + file_input.format(file_id), 'w') as f:
                        f.write(input_template.format(file_id=file_id, **format_dict))

                    with open(self.data_directory + file_batch.format(file_id), 'w') as f:
                        f.write(batch_template.format(file_id=file_id, **format_dict))

    def print_cp2k(self, file_output: str, n_config: int=None, **kwargs):
        """
        Print the final energy, time taken, and grid allocation for given cp2k
        settings. Formatted for a .md table.

        Parameters
        ----------
        file_output : str
            File name to read the cp2k output from. Will be formatted with the
            frame number, so should contain '{}' as part of the string.
        n_config: int, optional
            The number of configuration frames to use. If the number provided
            is greater than the length of `self.trajectory`, or `n_config` is
            `None` then the length is used instead. Default is `None`.
        **kwargs:
          - cutoff: tuple of float
          - relcutoff: tuple of float

        Examples
        --------
        from data import Data
        d = Data('/home/vol00/scarf860/cc_placement/CC_HDNNP/examples/m_cresol/')
        d.print_cp2k(file_output='cutoff/cp2k_output/cresol_{}.log', n_config=1,
                     cutoff=(400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
                             1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
                             2100, 2200, 2300, 2400))
        """
        n_config = self._min_n_config(n_config)
        file_id_template = 'n_{i}'

        msg = '| '
        if 'cutoff' in kwargs:
            file_id_template += '_cutoff_{cutoff}'
            cutoff_values = kwargs['cutoff']
            msg += '  {cutoff} | '
        else:
            cutoff_values = [None]

        if 'relcutoff' in kwargs:
            file_id_template += '_relcutoff_{relcutoff}'
            relcutoff_values = kwargs['relcutoff']
            msg += '  {relcutoff} | '
        else:
            relcutoff_values = [None]

        for i in range(n_config):
            for cutoff in cutoff_values:
                for relcutoff in relcutoff_values:
                    format_dict = {'i': i,
                                   'cutoff': cutoff,
                                   'relcutoff': relcutoff}

                    file_id = file_id_template.format(**format_dict)
                    with open(self.data_directory + file_output.format(file_id)) as f:
                        energy = None
                        m_grid = []
                        steps = None
                        total_time = None
                        for line in f.readlines():
                            if re.search('^ ENERGY\|', line):
                                energy = line.split()[-1]
                            elif re.search('^ count for grid', line):
                                m_grid.append(line.split()[4])
                            elif re.search('^  outer SCF loop converged in', line):
                                steps = line.split()[-2]
                            elif re.search('^ CP2K  ', line):
                                total_time = line.split()[-2]

                        try:
                            time_per_step = round(float(total_time) / int(steps), 1)
                        except TypeError:
                            time_per_step = None
                        msg_out = msg.format(**format_dict)
                        msg_out += '        1 | {energy} | {time_per_step} |    {total_time} | '.format(energy=energy,
                                                                                                    time_per_step=time_per_step,
                                                                                                    total_time=total_time,
                                                                                                    **format_dict)
                        msg_out += ' | '.join(m_grid)
                        print(msg_out + ' |')

    def write_n2p2_data(self, file_log: str, file_forces: str, file_xyz: str,
                        file_input: str, n_config: int=None):
        """
        Reads xyz and cp2k output data, and writes it to file as n2p2 input
        data in the following format:

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

        If `file_input` already exists, then it will not be overwritten but
        appended to. This allows multiple directories of xyz and cp2k output to
        be combined into one n2p2 input file.

        Parameters
        ----------
        file_log : str
            File name to read the cp2k output from. Will be formatted with the
            frame number, so should contain '{}' as part of the string.
        file_forces : str
            File name to read the cp2k forces from. Will be formatted with the
            frame number, so should contain '{}' as part of the string.
        file_xyz : str
            File name to read the xyz files from. Will be formatted with the
            frame number, so should contain '{}' as part of the string.
        file_input : str
            File name to write the n2p2 data to.
        n_config: int, optional
            The number of configuration frames to use. If the number provided
            is greater than the length of `self.trajectory`, or `n_config` is
            `None` then the length is used instead. Default is `None`.

        Examples
        --------
from data import Data
d = Data('/home/vol00/scarf860/cc_placement/CC_HDNNP/examples/m_cresol/')
d.write_n2p2_data(file_log='nvt_frames/cp2k_output/cresol_n_{}_cutoff_600_relcutoff_60.log',
                  file_forces='nvt_frames/cp2k_output/cresol_n_{}_cutoff_600_relcutoff_60-forces-1_0.xyz',
                  file_xyz='xyz/extxyz_32_{}',
                  file_input='all_frames/n2p2/input.data',
                  n_config=101)
d.write_n2p2_data(file_log='320K_frames/cp2k_output/cresol_n_{}_cutoff_600_relcutoff_60.log',
                  file_forces='320K_frames/cp2k_output/cresol_n_{}_cutoff_600_relcutoff_60-forces-1_0.xyz',
                  file_xyz='xyz_320K/{}.xyz',
                  file_input='all_frames/n2p2/input.data',
                  n_config=101)
d.write_n2p2_data(file_log='340K_frames/cp2k_output/cresol_n_{}_cutoff_600_relcutoff_60.log',
                  file_forces='340K_frames/cp2k_output/cresol_n_{}_cutoff_600_relcutoff_60-forces-1_0.xyz',
                  file_xyz='xyz_340K/{}.xyz',
                  file_input='all_frames/n2p2/input.data',
                  n_config=101)
d.write_n2p2_data(file_log='360K_frames/cp2k_output/cresol_n_{}_cutoff_600_relcutoff_60.log',
                  file_forces='360K_frames/cp2k_output/cresol_n_{}_cutoff_600_relcutoff_60-forces-1_0.xyz',
                  file_xyz='xyz_360K/{}.xyz',
                  file_input='all_frames/n2p2/input.data',
                  n_config=101)
        """
        text = ''
        n = self._min_n_config(n_config)
        for i in range(n):
            with open(self.data_directory + file_xyz.format(i)) as f:
                xyz_lines = f.readlines()
                n_atoms = int(xyz_lines[0].strip())
                header_list = xyz_lines[1].split('"')
                lattice_list = header_list[1].split()
                for j, lattice in enumerate(lattice_list):
                    lattice_list[j] = float(lattice) * 1e-10 / physical_constants['Bohr radius'][0]

            with open(self.data_directory + file_forces.format(i)) as f:
                force_lines = f.readlines()

            with open(self.data_directory + file_log.format(i)) as f:
                energy = None
                lines = f.readlines()
                for j, line in enumerate(lines):
                    if re.search('^ ENERGY\|', line):
                        energy = line.split()[-1]
                    if re.search('Hirshfeld Charges', line):
                        charge_lines = lines[j+3: j+3+n_atoms]
                        total_charge = lines[j+3+n_atoms+1].split()[-1]
                
            if energy is None:
                raise ValueError('Energy not found in {}'.format(file_log))

            text += 'begin\n'
            text += 'comment config_index={} units=Hartree and Bohr radii\n'.format(i)
            text += 'lattice {0} {1} {2}\n'.format(lattice_list[0], lattice_list[1], lattice_list[2])
            text += 'lattice {0} {1} {2}\n'.format(lattice_list[3], lattice_list[4], lattice_list[5])
            text += 'lattice {0} {1} {2}\n'.format(lattice_list[6], lattice_list[7], lattice_list[8])
            for j in range(n_atoms):
                atom_xyz = xyz_lines[j + 2].split()
                for k, position in enumerate(atom_xyz[1:], 1):
                    atom_xyz[k] = float(position) * 1e-10 / physical_constants['Bohr radius'][0]

                force = force_lines[j + 4].split()[-3:]
                charge = charge_lines[j].split()[-1]
                text += 'atom {1} {2} {3} {0} {4} 0.0 {5} {6} {7}\n'.format(*atom_xyz + [charge] + force)

            text += 'energy {}\n'.format(energy)
            text += 'charge {}\n'.format(total_charge)
            text += 'end\n'
        
        if isfile(self.data_directory + file_input):
            with open(self.data_directory + file_input, 'a') as f:
                f.write(text)
        else:
            with open(self.data_directory + file_input, 'w') as f:
                f.write(text)

    def write_n2p2_nn(self, file_template: str, elements: list, r_cutoff: float,
                      type: str, rule: str, n_pairs: int, mode: str,
                      zetas: list=[], r_lower: float=None, r_upper: float=None,
                      file_nn: str='input.nn'):
        """
        Based on `file_template`, write the input.nn file for n2p2 with
        symmetry functions generated using the provided arguments.

        Parameters
        ----------
        file_template : str
            File name to read the cp2k output from. Will be formatted with the
            frame number, so should contain '{}' as part of the string.
        elements: list of str
            List of the elements present in the system, expressed using their
            chemical symbol.
        r_cutoff: float
            The cutoff distance for the symmetry functions.
        type: {'radial', 'angular_narrow', 'angular_wide', 'weighted_radial', 'weighted_angular'}
            The type of symmetry function to generate.
        rule: {'imbalzano2018', 'gastegger2018'}
            The ruleset used to determine how to chose values for r_shift and eta.
        n_pairs: int
            The number of symmetry functions to generate. Specifically,
            `n_pairs` values for eta and r_shift are generated.
        mode: {'center', 'shift'}
            Whether the symmetry functions are centred or are shifted relative
            to the central atom.
        zetas: list of int, optional
            Not used for radial functions. Default is `[]`.
        r_lower: float, optional
            Not used for the 'imbalzano2018' ruleset. For 'gastegger2018', this
            sets either the minimum r_shift value or the maximum eta value for
            modes 'shift' and 'center' respectively. Default is `None`.
        r_upper: float, optional
            Not used for the 'imbalzano2018' ruleset. For 'gastegger2018', this
            sets either the maximum r_shift value or the minimum eta value for
            modes 'shift' and 'center' respectively. Default is `None`.
        file_nn: str, optional
            The file to write the output to. If the file already exists, then
            it is appended to with the new symmetry functions. If it does not,
            then it is created and the text from `file_template` is written to
            it before the symmetry functions are written.

        Examples
        --------
import sys
sys.path.append('/home/vol00/scarf860/cc_placement/CC_HDNNP/src')
from data import Data
d = Data('/home/vol00/scarf860/cc_placement/CC_HDNNP/examples/m_cresol/all_frames/n2p2/')

d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='radial',
                rule='imbalzano2018',
                mode='center',
                n_pairs=5)
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_narrow',
                rule='imbalzano2018',
                mode='center',
                n_pairs=5,
                zetas=[1])
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_wide',
                rule='imbalzano2018',
                mode='center',
                n_pairs=5,
                zetas=[1])
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='radial',
                rule='imbalzano2018',
                mode='shift',
                n_pairs=5)
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_narrow',
                rule='imbalzano2018',
                mode='shift',
                n_pairs=5,
                zetas=[1])
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_wide',
                rule='imbalzano2018',
                mode='shift',
                n_pairs=5,
                zetas=[1])


d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='imbalzano_center/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='radial',
                rule='imbalzano2018',
                mode='center',
                n_pairs=10)
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='imbalzano_center/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_narrow',
                rule='imbalzano2018',
                mode='center',
                n_pairs=10,
                zetas=[1, 4, 16])
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='imbalzano_center/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_wide',
                rule='imbalzano2018',
                mode='center',
                n_pairs=10,
                zetas=[1, 4, 16])

d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='imbalzano_shift/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='radial',
                rule='imbalzano2018',
                mode='shift',
                n_pairs=10)
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='imbalzano_shift/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_narrow',
                rule='imbalzano2018',
                mode='shift',
                n_pairs=10,
                zetas=[1, 4, 16])
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='imbalzano_shift/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_wide',
                rule='imbalzano2018',
                mode='shift',
                n_pairs=10,
                zetas=[1, 4, 16])

d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='gastegger_center/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='radial',
                rule='gastegger2018',
                mode='center',
                n_pairs=10,
                r_lower=1.0)
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='gastegger_center/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_narrow',
                rule='gastegger2018',
                mode='center',
                n_pairs=10,
                zetas=[1],
                r_lower=1.0)
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='gastegger_center/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_wide',
                rule='gastegger2018',
                mode='center',
                n_pairs=10,
                zetas=[1],
                r_lower=1.0)

d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='gastegger_shift/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='radial',
                rule='gastegger2018',
                mode='shift',
                n_pairs=10,
                r_lower=1.0)
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='gastegger_shift/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_narrow',
                rule='gastegger2018',
                mode='shift',
                n_pairs=10,
                zetas=[1],
                r_lower=1.0)
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='gastegger_shift/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='angular_wide',
                rule='gastegger2018',
                mode='shift',
                n_pairs=10,
                zetas=[1],
                r_lower=1.0)


d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='gastegger_center_weighted/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='weighted_radial',
                rule='gastegger2018',
                mode='center',
                n_pairs=10,
                r_lower=1.0)
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='gastegger_center_weighted/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='weighted_angular',
                rule='gastegger2018',
                mode='center',
                n_pairs=10,
                zetas=[1],
                r_lower=1.0)

d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='gastegger_shift_weighted/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='weighted_radial',
                rule='gastegger2018',
                mode='shift',
                n_pairs=10,
                r_lower=1.0)
d.write_n2p2_nn(file_template='input.nn.template',
                file_nn='gastegger_shift_weighted/input.nn',
                elements=['H', 'C', 'O'],
                r_cutoff=12.0,
                type='weighted_angular',
                rule='gastegger2018',
                mode='shift',
                n_pairs=10,
                zetas=[1],
                r_lower=1.0)



        """
        generator = SymFuncParamGenerator(elements=elements, r_cutoff = r_cutoff)
        generator.symfunc_type = type
        generator.zetas = zetas

        generator.generate_radial_params(rule=rule, mode=mode, nb_param_pairs=n_pairs,
                                         r_lower=r_lower, r_upper=r_upper)
        
        if isfile(self.data_directory + file_nn):
            with open(self.data_directory + file_nn, 'a') as f:
                generator.write_settings_overview(fileobj=f)
                generator.write_parameter_strings(fileobj=f)
        else:
            with open(self.data_directory + file_template) as f:
                template_text = f.read()
            with open(self.data_directory + file_nn, 'w') as f:
                f.write(template_text)
                generator.write_settings_overview(fileobj=f)
                generator.write_parameter_strings(fileobj=f)
