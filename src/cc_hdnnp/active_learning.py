from copy import deepcopy
from os import DirEntry, listdir
from os.path import isdir, isfile
import subprocess
from typing import Dict, List
import warnings

import numpy as np

from data import Data

# TODO combine all unit conversions
# Set a float to define the conversion factor from Bohr radius to Angstrom.
# Recommendation: 0.529177210903 (CODATA 2018).
Bohr2Ang = 0.529177210903
# Set a float to define the conversion factor from Hartree to electronvolt.
# Recommendation: 27.211386245988 (CODATA 2018).
Hartree2eV = 27.211386245988

class ActiveLearning:
    """
    """

    def __init__(self,
                 data_controller: Data,
                 n2p2_directories: List[str],
                 integrators: List[str]=['npt'],
                 pressures: List[float]=[1.],
                 N_steps: int=200000,
                 barostat_option: str='tri',
                 atom_style: str='atomic',
                 dump_lammpstrj: int=200,
                 min_timestep_separation_interpolation: List[int]=[200],
                 element_types: List[str]=['H', 'C', 'O'],
                 masses: List[float]=[1.00794, 12.011, 15.99491462],
                 max_len_joblist: int=0,
                 comment_name_keyword: str='comment structure',
                 structure_selection: List[List[int]] = [[0, 1]],
                 timestep: float=0.0005,
                 runner_cutoff: float=12.0,
                 periodic: bool=True,
                 # TODO combine with other element specification, create a class for it?
                 d_mins: List[Dict[str, List[float]]] = [{'H':[0.8, 0.8, 0.8], 'C':[0.8, 0.8], 'O':[0.8]}],
                 min_timestep_separation_extrapolation: List[int] = [20],
                 timestep_separation_interpolation_checks: List[int] = [10000],
                 delta_E: List[float] = [0.0001],
                 delta_F: List[float] = [0.01],
                 all_extrapolated_structures: List[bool] = [True],
                 exceptions: list = [None],
                 max_extrapolated_structures: List[int] = [50],
                 max_interpolated_structures_per_simulation: List[int] = [4],
                 tolerances: List[float] = [0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                 initial_tolerance: int = 5):
        """
        """

        self.structure_names = data_controller.structure_names

        for integrator in integrators:
            if integrator!='nve' and integrator!='nvt' and integrator!='npt':
                raise ValueError('Integrator {0} is not implemented.'.format(integrator))
        self.integrators = integrators

        if not list(pressures) and 'npt' in integrators:
            raise ValueError('Integrator npt requires to specify at least one value for pressure.')
        self.pressures = pressures

        if barostat_option!='tri' and barostat_option!='aniso' and barostat_option!='iso':
            raise ValueError('Barostat option {0} is not implemented in RuNNerActiveLearn_1.py.'.format(barostat_option))
        self.barostat_option = barostat_option

        if atom_style!='atomic' and atom_style!='full':
            raise ValueError('Atom style {0} is not implemented in RuNNerActiveLearn_1.py.'.format(atom_style))
        self.atom_style = atom_style

        if N_steps%dump_lammpstrj!=0:
            raise ValueError('N_steps has to be a multiple of dump_lammpstrj ({0}!=N*{1}).'.format(N_steps, dump_lammpstrj))
        self.N_steps = N_steps

        if dump_lammpstrj<np.array(min_timestep_separation_interpolation).min():
            raise ValueError('The extrapolation free structures would be stored only every {0}th time step, but the minimum time step separation of interpolated structures is set to {1} time steps.'.format(dump_lammpstrj, np.array(min_timestep_separation_interpolation).min()))
        self.dump_lammpstrj = dump_lammpstrj
        self.min_timestep_separation_interpolation = min_timestep_separation_interpolation

        if len(element_types)!=len(masses):
            raise ValueError('The number of given element types is not equal to the number of given masses ({0}!={1}).'.format(len(element_types), len(masses)))
        self.masses = masses

        if not max_len_joblist>=0:
            raise ValueError('The maximal length of the job list has to be set to 0 (which means infinity) or a positive integer number.'.format(max_len_joblist))
        self.max_len_joblist = max_len_joblist

        if (comment_name_keyword==None and self.structure_names!=None) or (comment_name_keyword!=None and self.structure_names==None):
            raise ValueError('If comment_name_keyword or structure_names is set to None the other one has to be set to None as well.')
        self.comment_name_keyword = comment_name_keyword   

        if self.structure_names==None:
            if not (1==len(d_mins)==
                    len(min_timestep_separation_extrapolation)==
                    len(timestep_separation_interpolation_checks)==
                    len(min_timestep_separation_interpolation)==
                    len(delta_E)==len(delta_F)==len(structure_selection)==
                    len(all_extrapolated_structures)==len(exceptions)):
                raise ValueError('As no structure names are given, exactly one setting for structure_selection is required.')
            if structure_selection[0][0]<0 or structure_selection[0][1]<1:
                raise ValueError('The settings of structure_selection are not '
                                 'reasonable (every {0}th structure starting '
                                 'with the {0}th one).'
                                 ''.format(structure_selection[0][1],
                                           structure_selection[0][0]))
        else:
            if isinstance(self.structure_names, list):
                for structure_name in self.structure_names:
                    if structure_name==None:
                        raise TypeError('Individual structure names cannot be set to None. You have to specify an array of structure names or use structure_names = None.')
            else:
                raise TypeError('`structure_names` has to be set to None or a '
                                '`list` of structure names, but was an '
                                'instance of `{}`.'.format(type(self.structure_names)))   

            n_structure_names = len(self.structure_names)    
            if len(structure_selection)==1:
                if structure_selection[0][0]<0 or structure_selection[0][1]<1:
                    raise ValueError('The settings of structure_selection are not reasonable (every {0}th structure starting with the {0}th one).'.format(structure_selection[0][1], structure_selection[0][0]))
                structure_selection = [deepcopy(structure_selection[0]) for i in range(n_structure_names)]
            elif len(structure_selection)!=n_structure_names:
                raise ValueError('Structure name dependent settings for structure_selection are not given for every structure name or there are too many settings for the given structure names. Also there is not given one value which could be used for all structures names.')
            else:
                for i in range(n_structure_names):
                    if structure_selection[i][0]<0 or structure_selection[i][1]<1:
                        raise ValueError('The settings of structure_selection are not reasonable (every {0}th structure starting with the {0}th one).'.format(structure_selection[i][1], structure_selection[i][0]))

            if len(d_mins)==1:
                d_mins *= n_structure_names
            elif len(d_mins)!=n_structure_names:
                raise ValueError('Structure name dependent settings for d_mins are not given for every structure name or there are too many settings for the given structure names. Also there is not given one value which could be used for all structures names.')

            if len(min_timestep_separation_extrapolation)==1:
                min_timestep_separation_extrapolation *= n_structure_names
            elif len(min_timestep_separation_extrapolation)!=n_structure_names:
                raise ValueError('Structure name dependent settings for min_timestep_separation_extrapolation are not given for every structure name or there are too many settings for the given structure names. Also there is not given one value which could be used for all structures names.')

            if len(timestep_separation_interpolation_checks)==1:
                timestep_separation_interpolation_checks *= n_structure_names
            elif len(timestep_separation_interpolation_checks)!=n_structure_names:
                raise ValueError('Structure name dependent settings for timestep_separation_interpolation_checks are not given for every structure name or there are too many settings for the given structure names. Also there is not given one value which could be used for all structures names.')

            if len(min_timestep_separation_interpolation)==1:
                min_timestep_separation_interpolation *= n_structure_names
            elif len(min_timestep_separation_interpolation)!=n_structure_names:
                raise ValueError('Structure name dependent settings for min_timestep_separation_interpolation are not given for every structure name or there are too many settings for the given structure names. Also there is not given one value which could be used for all structures names.')

            if len(delta_E)==1:
                delta_E *= n_structure_names
            elif len(delta_E)!=n_structure_names:
                raise ValueError('Structure name dependent settings for delta_E are not given for every structure name or there are too many settings for the given structure names. Also there is not given one value which could be used for all structures names.')
            
            if len(delta_F)==1:
                delta_F *= n_structure_names
            elif len(delta_F)!=n_structure_names:
                raise ValueError('Structure name dependent settings for delta_F are not given for every structure name or there are too many settings for the given structure names. Also there is not given one value which could be used for all structures names.')
            
            if len(all_extrapolated_structures)==1:
                all_extrapolated_structures *= n_structure_names
            elif len(all_extrapolated_structures)!=n_structure_names:
                raise ValueError('Structure name dependent settings for all_extrapolated_structures are not given for every structure name or there are too many settings for the given structure names. Also there is not given one value which could be used for all structures names.')
            
            if len(max_extrapolated_structures)==1:
                max_extrapolated_structures *= n_structure_names
            elif len(max_extrapolated_structures)!=n_structure_names:
                raise ValueError('Structure name dependent settings for max_extrapolated_structures are not given for every structure name or there are too many settings for the given structure names. Also there is not given one value which could be used for all structures names.')
            
            if len(max_interpolated_structures_per_simulation)==1:
                max_interpolated_structures_per_simulation *= n_structure_names
            elif len(max_interpolated_structures_per_simulation)!=n_structure_names:
                raise ValueError('Structure name dependent settings for max_interpolated_structures_per_simulation are not given for every structure name or there are too many settings for the given structure names. Also there is not given one value which could be used for all structures names.')
            
            if len(exceptions)==1:
                exceptions *= n_structure_names
            elif len(exceptions)!=n_structure_names:
                raise ValueError('Structure name dependent settings for exceptions are not given for every structure name or there are too many settings for the given structure names. Also there is not given one value which could be used for all structures names.')
            
            if not (np.array(max_extrapolated_structures)>=0).all():
                raise ValueError('The value of max_extrapolated_structures has to be an integer equal or higher than 0.')
            

        if (np.array(timestep_separation_interpolation_checks)*5>=N_steps).any():
            raise ValueError('The time step separation between two interpolation checks has to be smaller than a fifth of the number of MD steps.')
            
        if (np.array(timestep_separation_interpolation_checks)<np.array(min_timestep_separation_interpolation)).any():
            raise ValueError('The time step separation between two interpolation checks is set to a smaller value than the minimal time step separation between two interpolated structures.')
            
        if initial_tolerance<=1:
            raise ValueError('The value of initial_tolerance has to be higher than 1.')
            
        if len(tolerances)<=initial_tolerance:
            raise ValueError('There are not enough tolerance values as initial_tolerance results in an index error.')
            

        if timestep>0.01:
            print('WARNING: Very large timestep of {0} ps.'.format(timestep))
        self.timestep = timestep

        self.active_learning_directory = data_controller.active_learning_directory
        self.n2p2_directories = n2p2_directories
        self.runner_cutoff = runner_cutoff
        self.periodic = periodic
        self.structure_selection = structure_selection
        self.delta_E = delta_E
        self.delta_F = delta_F
        self.all_extrapolated_structures = all_extrapolated_structures
        self.max_extrapolated_structures = max_extrapolated_structures
        self.max_interpolated_structures_per_simulation = max_interpolated_structures_per_simulation
        self.timestep_separation_interpolation_checks = timestep_separation_interpolation_checks
        self.exceptions = exceptions
        self.tolerances = tolerances
        self.min_timestep_separation_extrapolation = min_timestep_separation_extrapolation
        self.min_timestep_separation_interpolation = min_timestep_separation_interpolation
        self.initial_tolerance = initial_tolerance
        self.element_types = element_types
        self.d_mins = d_mins

        self.lattices = []
        self.elements = []
        self.charges = []
        self.statistics = []
        self.names = []
        self.positions = []
        self.selection = None


    def read_input_data(self, comment_name_separator: str='-', comment_name_index: int=2):
        '''
        '''
        names = []
        lattices = []
        elements = []
        xyzs = []
        qs = []

        with open(self.n2p2_directories[0] + '/input.data') as f:
            # TODO assert that the data used for both networks is the same
            for line in f.readlines():
                line = line.strip()
                if line.startswith('atom'):
                    line = line.split()
                    elements[-1].append(line[4])
                    xyzs[-1].append([line[1], line[2], line[3]])
                    qs[-1].append(line[5])
                elif line.startswith('lattice'):
                    lattices[-1].append(line.split()[1:4])
                elif line.startswith('begin'):
                    lattices.append([])
                    elements.append([])
                    xyzs.append([])
                    qs.append([])
                elif line.startswith('end'):
                    if not elements[-1]:
                        raise ValueError('For some of the structures the definition of the atoms is incomplete or missing.')
                    xyzs[-1] = np.array(xyzs[-1]).astype(float)*Bohr2Ang
                    qs[-1] = np.array(qs[-1]).astype(float)
                    if self.periodic:
                        if len(lattices[-1])==3:
                            lattices[-1] = np.array(lattices[-1]).astype(float)*Bohr2Ang
                        else:
                            raise ValueError('The periodic keyword is set to True but for some of the structures the definition of the lattice is incomplete or missing.')
                    else:
                        if lattices[-1]:
                            raise ValueError('The periodic keyword is set to False but for some of the structures a definition of a lattice exists.')
                        else:
                            lattices[-1] = np.array([[xyzs[-1][:,0].max()-xyzs[-1][:,0].min()+2*self.runner_cutoff*Bohr2Ang, 0.0, 0.0], [0.0, xyzs[-1][:,1].max()-xyzs[-1][:,1].min()+2*self.runner_cutoff*Bohr2Ang, 0.0], [0.0, 0.0, xyzs[-1][:,2].max()-xyzs[-1][:,2].min()+2*self.runner_cutoff*Bohr2Ang]])
                else:
                    if self.comment_name_keyword!=None:
                        if line.startswith(self.comment_name_keyword):
                            names.append(line.split()[comment_name_index].split(comment_name_separator)[0])

        names = np.array(names)
        lattices = np.array(lattices)
        elements = np.array(elements)
        xyzs = np.array(xyzs)
        qs = np.array(qs)

        return names, lattices, elements, xyzs, qs


    def write_input_lammps(self, path, seed, temperature, pressure, integrator):
        '''
        '''
        runner_cutoff = round(self.runner_cutoff*Bohr2Ang, 12)
        cflength = round(1.0/Bohr2Ang, 12)
        cfenergy = round(1.0/Hartree2eV, 15)
        elements_string = ''
        for element_type in self.element_types:
            elements_string += element_type+' '

        input_lammps = 'variable temperature equal {0}\n'.format(float(temperature))
        if integrator=='npt':
            input_lammps += 'variable pressure equal {0}\n'.format(float(pressure))
        input_lammps += 'variable N_steps equal {0}\n'.format(self.N_steps)\
                        + 'variable seed equal {0}\n\n'.format(seed)
        input_lammps += 'units metal\n'\
                        + 'boundary p p p\n'\
                        + 'atom_style {0}\n'.format(self.atom_style)\
                        + 'read_data structure.lammps\n'\
                        + 'pair_style nnp dir RuNNer showew yes resetew no maxew 750 showewsum 0 cflength {0} cfenergy {1}\n'.format(cflength, cfenergy)\
                        + 'pair_coeff * * {0}\n'.format(runner_cutoff)\
                        + 'timestep {0}\n'.format(self.timestep)
        if integrator=='nve':
            input_lammps += 'fix int all nve\n'
        elif integrator=='nvt':
            input_lammps += 'fix int all nvt temp ${{temperature}} ${{temperature}} {0}\n'.format(self.timestep*100)
        elif integrator=='npt':
            input_lammps += 'fix int all npt temp ${{temperature}} ${{temperature}} {0} {1} ${{pressure}} ${{pressure}} {2} fixedpoint 0.0 0.0 0.0\n'.format(self.timestep*100, self.barostat_option, self.timestep*1000)
        input_lammps += 'thermo 1\n'\
                        + 'variable thermo equal 0\n'\
                        + 'thermo_style custom v_thermo step time temp epair etotal fmax fnorm press cella cellb cellc cellalpha cellbeta cellgamma density\n'\
                        + 'thermo_modify format line "thermo %8d %10.4f %8.3f %15.5f %15.5f %9.4f %9.4f %9.2f %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f %8.5f"\n'
        if self.periodic:
            if self.atom_style=='atomic':
                input_lammps += 'dump lammpstrj all custom 1 structure.lammpstrj id element x y z\n'
            elif self.atom_style=='full':
                input_lammps += 'dump lammpstrj all custom 1 structure.lammpstrj id element x y z q\n'
            input_lammps += 'dump_modify lammpstrj pbc yes sort id element {0}\n'.format(elements_string[:-1])
        else:
            if self.atom_style=='atomic':
                input_lammps += 'dump lammpstrj all custom 1 structure.lammpstrj id element xu yu zu\n'
            elif self.atom_style=='full':
                input_lammps += 'dump lammpstrj all custom 1 structure.lammpstrj id element xu yu zu q\n'
            input_lammps += 'dump_modify lammpstrj pbc no sort id element {0}\n'.format(elements_string[:-1])
        input_lammps += 'velocity all create ${temperature} ${seed}\n\n'

        with open(path+'/input.lammps', 'w') as f:
            f.write(input_lammps)

        subprocess.Popen('cat {0}/simulation.lammps >> {1}/input.lammps'.format(self.active_learning_directory, path), shell=True)


    def write_structure_lammps(self, path, lattice, element, xyz, q):
        '''
        '''
        lattice_lammps = self.transform_lattice(lattice)

        structure_lammps = 'RuNNerActiveLearn\n\n'\
                        + '{0} atoms\n\n'.format(len(element))\
                        + '{0} atom types\n\n'.format(len(self.element_types))\
                        + '0.0 {0} xlo xhi\n'.format(round(lattice_lammps[0], 5))\
                        + '0.0 {0} ylo yhi\n'.format(round(lattice_lammps[1], 5))\
                        + '0.0 {0} zlo zhi\n'.format(round(lattice_lammps[2], 5))
        if self.barostat_option=='tri':
            structure_lammps += '{0} {1} {2} xy xz yz\n'.format(round(lattice_lammps[3], 5), round(lattice_lammps[4],5), round(lattice_lammps[5], 5))
        structure_lammps += '\nMasses\n\n'
        for i in range(len(self.masses)):
            structure_lammps += '{0} {1}\n'.format(i+1, self.masses[i])
        structure_lammps += '\nAtoms\n\n'

        with open(path+'/structure.lammps', 'w') as f:
            f.write(structure_lammps)
            if self.atom_style=='atomic':
                for i in range(len(element)):
                    f.write('{0:4d} {1} {2:9.5f} {3:9.5f} {4:9.5f}\n'.format(i+1, self.element_types.index(element[i])+1, xyz[i][0], xyz[i][1], xyz[i][2]))
            elif self.atom_style=='full':
                for i in range(len(element)):
                    f.write('{0:4d} 1 {1} {2:6.3f} {3:9.5f} {4:9.5f} {5:9.5f}\n'.format(i+1, self.element_types.index(element[i])+1, round(q[i], 3), round(xyz[i][0], 5), round(xyz[i][1], 5), round(xyz[i][2], 5)))


    def transform_lattice(self, lattice):
        '''
        '''
        a = np.linalg.norm(lattice[0])
        b = np.linalg.norm(lattice[1])
        c = np.linalg.norm(lattice[2])
        cos_alpha = np.dot(lattice[1], lattice[2])/b/c
        cos_beta = np.dot(lattice[0], lattice[2])/a/c
        cos_gamma = np.dot(lattice[0], lattice[1])/a/b
        xy = b*cos_gamma
        lx = a
        xz = c*cos_beta
        ly = np.sqrt(b**2-xy**2)
        yz = (b*c*cos_alpha-xy*xz)/ly
        lz = np.sqrt(c**2-xz**2-yz**2)

        return [lx, ly, lz, xy, xz, yz]


    def write_lammps(self, temperatures: range=range(300, 400), seed: int=1):
        """
        """
        mode1_directory = self.active_learning_directory + '/mode1'
        if isdir(mode1_directory):
            raise IOError('Path mode1 already exists. Please remove old directory first if you would like to recreate it.')
        subprocess.Popen('mkdir ' + mode1_directory, shell=True).wait()

        # TODO allow non-default arguments
        names_all, lattices_all, elements_all, xyzs_all, qs_all = self.read_input_data()

        if self.max_len_joblist==0:
            joblist_name = self.active_learning_directory + '/joblist_mode1.dat'
            with open(joblist_name, 'w') as f:
                f.write('')
        if self.structure_names==None:
            structure_names = [None]
        else:
            structure_names = self.structure_names
        pressures_npt = self.pressures
        n_simulations = 0
        n_previous_simulations = 0
        counter = 0

        for i in range(len(structure_names)):
            if structure_names[i]==None:
                names = names_all
                lattices = lattices_all
                elements = elements_all
                xyzs = xyzs_all
                qs = qs_all
            else:
                try:
                    names = names_all[names_all==structure_names[i]]
                except IndexError as e:
                    raise IndexError('structure_names: {0}\nnames_all: {1}'.format(structure_names, names_all))
                lattices = lattices_all[names_all==structure_names[i]]
                elements = elements_all[names_all==structure_names[i]]
                xyzs = xyzs_all[names_all==structure_names[i]]
                qs = qs_all[names_all==structure_names[i]]
                print('Structure name: {0}'.format(structure_names[i]))
            self.structure_selection[i][0] = self.structure_selection[i][0]%self.structure_selection[i][1]
            print('Starting from the {0}th structure every {1}th structure of the '
                'input.data file is used.'.format(self.structure_selection[i][0],
                                                    self.structure_selection[i][1]))
            n_structures = len(lattices)
            n_npt = int(np.array([1 for j in self.integrators if j=='npt']).sum())
            repetitions = max(1,
                            int(float(n_structures)/2/len(self.integrators)/len(temperatures)/((len(self.pressures)-1)
                                *n_npt/len(self.integrators)+1)/self.structure_selection[i][1]))
            print('The given variations of the settings are repeated {0} times'.format(repetitions))

            for x in range(repetitions):
                for j in (0, 1):
                    HDNNP = str(j+1)
                    for integrator in self.integrators:
                        for temperature in temperatures:
                            if integrator!='npt':
                                pressures = [0]
                            else:
                                pressures = pressures_npt
                            for pressure in pressures:
                                if n_structures//self.structure_selection[i][1]<=counter:
                                    n_simulations += counter
                                    counter = 0
                                    print('WARNING: The structures of the input.data file are used more than once.')
                                    if self.structure_selection[i][1]>1:
                                        self.structure_selection[i][0] = (self.structure_selection[i][0]+1)%self.structure_selection[i][1]
                                        print('Try to avoid this by start from the {0}th structure and using again every {1}th structure.'
                                            ''.format(self.structure_selection[i][0], self.structure_selection[i][1]))
                                selection = counter * self.structure_selection[i][1] + self.structure_selection[i][0]
                                path = ''
                                if self.comment_name_keyword!=None:
                                    try:
                                        path += names[selection] + '_'
                                    except IndexError as e:
                                        msg = '`names`={0} does not not have entries for the `selection`={1}'.format(names, selection)
                                        raise IndexError(msg) from e
                                
                                if integrator=='npt':
                                    path += integrator+'_hdnnp'+HDNNP+'_t'+str(temperature)+'_p'+str(pressure)+'_'+str(seed)
                                else:
                                    path += integrator+'_hdnnp'+HDNNP+'_t'+str(temperature)+'_'+str(seed)

                                mode1_path = mode1_directory +'/'+path
                                if isdir(mode1_path):
                                    raise IOError('Path {0} already exists. Please remove old directories first if you would like to recreate them.'.format(mode1_path))
                                subprocess.Popen('mkdir '+ mode1_path, shell=True).wait()
                                self.write_input_lammps(mode1_path, seed, temperature, pressure, integrator)
                                self.write_structure_lammps(mode1_path, lattices[selection], elements[selection], xyzs[selection], qs[selection])
                                subprocess.Popen('mkdir '+mode1_path+'/RuNNer', shell=True).wait()
                                # TODO only copy the minimum needed since we're re-using the n2p2 directory outright
                                # TODO handle the absence of weights.XXX.data files gracefully
                                # subprocess.Popen('cp -i ' + self.n2p2_directories[j] + '/* '+ mode1_path +'/RuNNer', shell=True)
                                subprocess.Popen('cp -i ' + self.n2p2_directories[j] + '/input.nn '+ mode1_path +'/RuNNer/input.nn', shell=True)
                                subprocess.Popen('cp -i ' + self.n2p2_directories[j] + '/scaling.data '+ mode1_path +'/RuNNer/scaling.data', shell=True)
                                subprocess.Popen('cp -i ' + self.n2p2_directories[j] + '/weights.*.data '+ mode1_path +'/RuNNer', shell=True)
                                if self.max_len_joblist!=0 and (n_simulations+counter)%self.max_len_joblist==0:
                                    joblist_name = self.active_learning_directory + '/joblist_mode1_'+str((n_simulations+counter)//self.max_len_joblist+1)+'.dat'
                                    with open(joblist_name, 'w') as f:
                                        f.write('')
                                with open(joblist_name, 'a') as f:
                                    f.write('{0}\n'.format(path))
                                seed += 1
                                counter += 1

            if structure_names[i]!=None:
                n_simulations += counter
                counter = 0
                print('Input was generated for {0} simulations.'.format(n_simulations-n_previous_simulations))
                n_previous_simulations = n_simulations

        if structure_names[0]==None:
            n_simulations += counter
            print('Input was generated for {0} simulations.'.format(n_simulations))


    def _read_lammps_log(self, dump_lammpstrj, directory):
        '''
        '''
        with open(directory + '/log.lammps') as f:
            data = [line for line in f.readlines()]
        counter = 0
        n_lines = len(data)
        while counter<n_lines and not data[counter].startswith('**********'):
            counter += 1

        extrapolation = False
        i = counter
        while i<n_lines and not data[i].startswith('### NNP EXTRAPOLATION WARNING ###'):
            i += 1
        if i<n_lines:
            extrapolation = True
        i -= 1
        while i>counter and not data[i].startswith('thermo'):
            i -= 1
        if extrapolation:
            extrapolation_free_lines = i
            if i>counter:
                extrapolation_free_timesteps = int(data[i].split()[1])
            else:
                extrapolation_free_timesteps = -1
        else:
            extrapolation_free_lines = -1
            extrapolation_free_timesteps = int(data[i].split()[1])

        data = [int(line.split()[1]) if line.startswith('thermo') else -1 for line in data[counter:] if line.startswith('thermo') or line.startswith('### NNP EXTRAPOLATION WARNING ###')]
        timesteps = np.unique(np.array([data[i] for i in range(1,len(data)) if data[i]!=-1 and (data[i]%dump_lammpstrj==0 or data[i-1]==-1)]))

        return timesteps, extrapolation_free_lines, extrapolation_free_timesteps


    def _read_lammpstrj(self, timesteps, directory):
        '''
        '''
        structures = []
        i = 0
        n_timesteps = len(timesteps)
        with open(directory + '/structure.lammpstrj') as f:
            line = f.readline()
            while line and i<n_timesteps:
                while not line.startswith('ITEM: TIMESTEP') and line:
                    line = f.readline()
                line = f.readline()
                if timesteps[i]==int(line.strip()):
                    structures.append('ITEM: TIMESTEP\n')
                    while not line.startswith('ITEM: TIMESTEP') and line:
                        structures.append(line)
                        line = f.readline()
                    i += 1

        i = 1
        n_lines = len(structures)
        while i<n_lines and not structures[i].startswith('ITEM: TIMESTEP'):
            i += 1
        structure_lines = i 

        return structures, structure_lines


    def _write_lammpstrj(self, structures, directory):
        '''
        '''
        with open(directory + '/structure.lammpstrj', 'w') as f:
            for line in structures:
                f.write(line)


    def _write_extrapolation(self, extrapolation_free_timesteps, extrapolation_free_lines, dump_lammpstrj, structure_lines, last_timestep, directory):
        '''
        '''
        with open(directory + '/extrapolation.dat', 'w') as f:
            f.write('extrapolation_free_initial_time_steps: {0}\nlines_before_first_extrapolation: {1}\ntimesteps_between_non_extrapolated_structures: {2}\nlines_per_structure: {3}\nlast_timestep: {4}'.format(extrapolation_free_timesteps, extrapolation_free_lines, dump_lammpstrj, structure_lines, last_timestep)) 


    def prepare_lammps_trajectory(self):
        """
        """
        for path in listdir(self.active_learning_directory + '/mode1'):
            directory = self.active_learning_directory + '/mode1/' + path
            timesteps, extrapolation_free_lines, extrapolation_free_timesteps = self._read_lammps_log(self.dump_lammpstrj, directory=directory)
            structures, structure_lines = self._read_lammpstrj(timesteps, directory=directory)
            self._write_lammpstrj(structures, directory=directory)
            self._write_extrapolation(extrapolation_free_timesteps, extrapolation_free_lines, self.dump_lammpstrj, structure_lines, timesteps[-1], directory=directory)


    def _get_paths(self, structure_name):
        '''
        '''
        if structure_name!='':
            try:
                cmd = 'ls {0}/mode1 | grep {1}_ | grep -e _nve_hdnnp -e _nvt_hdnnp -e _npt_hdnnp'.format(self.active_learning_directory, structure_name)
                paths = str(subprocess.check_output(cmd,
                                                    stderr=subprocess.STDOUT, shell=True).decode()).strip().split('\n')
            except subprocess.CalledProcessError:
                raise IOError('Simulations with the structure name {0} were not found.'.format(structure_name))
                
        else:
            paths = str(subprocess.check_output('ls {}/mode1 | grep -e nve_hdnnp -e nvt_hdnnp -e npt_hdnnp'.format(self.active_learning_directory),
                                                stderr=subprocess.STDOUT, shell=True).decode()).strip().split('\n')

        finished = []
        for i in range(len(paths)):
            if isfile(self.active_learning_directory + '/mode1/'+paths[i]+'/extrapolation.dat'):
                finished.append(i)
            else:
                print('Simulation {0} is not finished.'.format(paths[i]))
        paths = np.array(paths)[finished]
        if len(paths)==0:
            raise ValueError('None of the {0} simulations finished.'.format(structure_name))
            

        return paths


    def _read_extrapolation(self, path):
        '''
        '''
        with open('{0}/mode1/{1}/extrapolation.dat'.format(self.active_learning_directory, path)) as f:
            extrapolation_data = np.array([line.strip().split() for line in f.readlines()])[:,1].astype(int)

        return extrapolation_data


    def _read_log_format(self, path):
        '''
        '''
        with open('{0}/mode1/{1}/log.lammps'.format(self.active_learning_directory, path)) as f:
            data = [line for line in f.readlines()]
        counter = 0
        n_lines = len(data)
        while counter<n_lines and not data[counter].startswith('**********'):
            counter += 1
        if counter<n_lines:
            if data[counter+2].startswith('   NNP LIBRARY v2.0.0'):
                extrapolation_format = 'v2.0.0'
            elif (data[counter+5].startswith('n²p² version      : v2.1.1') or 
                  data[counter+5].startswith('n²p² version  (from git): v2.1.4')):
                extrapolation_format = 'v2.1.1'
            else:
                raise IOError('n2p2 extrapolation warning format cannot be identified in the file {0}/log.lammps. Known formats are corresponding to n2p2 v2.0.0 and v2.1.1.'.format(path))
                
        else:
            raise IOError('n2p2 extrapolation warning format cannot be identified in the file {0}/log.lammps. Known formats are corresponding to n2p2 v2.0.0 and v2.1.1.'.format(path))
            

        return extrapolation_format


    def _read_log(self, path, extrapolation_data, extrapolation_format):
        '''
        '''
        if extrapolation_data[1]!=-1:
            with open('{0}/mode1/{1}/log.lammps'.format(self.active_learning_directory, path)) as f:
                data = [line.strip() for line in f.readlines()][extrapolation_data[1]:-1]

            if extrapolation_format=='v2.0.0':
                data = np.array([[float(line.split()[1])]+[np.nan,np.nan,np.nan,np.nan,np.nan] if line.startswith('thermo') else [np.nan]+list(np.array(line.split())[[12,14,16,8,10]].astype(float)) for line in data if line.startswith('thermo') or line.startswith('### NNP')])
            elif extrapolation_format=='v2.1.1':
                data = np.array([[float(line.split()[1])]+[np.nan,np.nan,np.nan,np.nan,np.nan] if line.startswith('thermo') else [np.nan]+list(np.array(line.split())[[16,18,20,8,12]].astype(float)) for line in data if line.startswith('thermo') or line.startswith('### NNP')])

            if np.isnan(data[-1,0]):
                data = data[:-np.argmax(np.isfinite(data[:,0][::-1]))]

            if np.isnan(data[0,0]):
                print('WARNING: Extrapolation occurred already in the first time step in {0}.'.format(path))
                data = np.concatenate((np.array([[-1.0]+[np.nan,np.nan,np.nan,np.nan,np.nan]]),data), axis=0)
            extrapolation = np.absolute((data[:,1]-data[:,2])/(data[:,3]-data[:,2])-0.5)-0.5

            for i in range(1,len(extrapolation)):
                if np.isfinite(extrapolation[i]) and np.isfinite(extrapolation[i-1]):
                    extrapolation[i] += extrapolation[i-1]

            extrapolation_indices = []
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                for tolerance in self.tolerances:
                    extrapolation_indices.append(np.argmax(extrapolation>tolerance))
            extrapolation_timestep = []
            extrapolation_value = []
            extrapolation_statistic = []
            for i in range(len(self.tolerances)):
                if extrapolation_indices[i]>0:
                    j = 1
                    while np.isnan(data[extrapolation_indices[i]+j,0]):
                        j += 1
                    extrapolation_timestep.append(int(data[extrapolation_indices[i]+j,0]))
                    extrapolation_value.append(extrapolation[extrapolation_indices[i]+j-1])
                    extrapolation_statistic.append([])
                    j -= 1
                    extrapolation_statistic[-1].append(data[extrapolation_indices[i]+j,[1,4,5]])
                    extrapolation_statistic[-1][-1][0] = (data[extrapolation_indices[i]+j,1]-data[extrapolation_indices[i]+j,2])/(data[extrapolation_indices[i]+j,3]-data[extrapolation_indices[i]+j,2])-0.5
                    if extrapolation_statistic[-1][-1][0]<0:
                        extrapolation_statistic[-1][-1][0] += 0.5
                    else:
                        extrapolation_statistic[-1][-1][0] -= 0.5
                    j -= 1
                    while np.isnan(data[extrapolation_indices[i]+j,0]):
                        extrapolation_statistic[-1].append(data[extrapolation_indices[i]+j,[1,4,5]])
                        extrapolation_statistic[-1][-1][0] = (data[extrapolation_indices[i]+j,1]-data[extrapolation_indices[i]+j,2])/(data[extrapolation_indices[i]+j,3]-data[extrapolation_indices[i]+j,2])-0.5
                        if extrapolation_statistic[-1][-1][0]<0:
                            extrapolation_statistic[-1][-1][0] += 0.5
                        else:
                            extrapolation_statistic[-1][-1][0] -= 0.5
                        j -= 1
                else:
                    extrapolation_timestep.append(-1)
                    extrapolation_value.append(0)
                    extrapolation_statistic.append([None])
        else:
            extrapolation_timestep = len(self.tolerances)*[-1]
            extrapolation_value = len(self.tolerances)*[0]
            extrapolation_statistic = len(self.tolerances)*[None]

        return extrapolation_timestep, extrapolation_value, extrapolation_statistic


    def _get_timesteps(self, extrapolation_timesteps, extrapolation_values, extrapolation_data, index: int):
        '''
        '''
        min_fraction = 0.001
        n_tolerances = len(self.tolerances)
        n_small = 2
        small = n_small-1
        structure_extrapolation = self.min_timestep_separation_extrapolation[index]
        structure_interpolation = self.min_timestep_separation_interpolation[index]
        structure_checks = self.timestep_separation_interpolation_checks[index]
        if len(extrapolation_timesteps[:,small][extrapolation_timesteps[:,small]>=structure_extrapolation])<min_fraction*len(extrapolation_timesteps[:,small]):
            print('Only less than 0.1% of the simulations show an extrapolation if a tolerance of {0} is employed (the initial {1} time steps are neglected). The tolerance value is reduced to {2}.'.format(self.tolerances[small], structure_extrapolation, self.tolerances[small-1]))
            small -= 1
        if not (extrapolation_timesteps[:,small][extrapolation_timesteps[:,small]>=0]).any():
            print('There are no small extrapolations.')
            tolerance_indices = len(extrapolation_timesteps)*[-1]
        else:
            n_simulations_extrapolation = len(extrapolation_timesteps[:,small][extrapolation_timesteps[:,small]>=0])
            n_simulations = len(extrapolation_timesteps[:,small])
            print('Small extrapolations are present in {0} of {1} simulations ({2}%).'.format(n_simulations_extrapolation, n_simulations, round(100.0*n_simulations_extrapolation/n_simulations,2)))
            extrapolation_values_reduced = extrapolation_values[extrapolation_timesteps[:,small]==-1]
            mean_small = np.mean(extrapolation_values_reduced[:,small])
            std_small = np.std(extrapolation_values_reduced[:,small])
            criterium = mean_small+std_small+max(0,self.tolerances[small]-mean_small+std_small)
            while criterium>self.tolerances[self.initial_tolerance] and self.initial_tolerance<n_tolerances:
                self.initial_tolerance += 1
            while (not (extrapolation_timesteps[:,self.initial_tolerance][extrapolation_timesteps[:,self.initial_tolerance]>=structure_extrapolation]).any()
                    and self.initial_tolerance>n_small):
                print('There are no large extrapolations for a tolerance of {0} (the initial {1} time steps are neglected). The tolerance value is reduced to {2}.'
                        ''.format(self.tolerances[self.initial_tolerance], structure_extrapolation, self.tolerances[self.initial_tolerance-1]))
                self.initial_tolerance -= 1
            if self.initial_tolerance==n_small:
                if not (extrapolation_timesteps[:,self.initial_tolerance][extrapolation_timesteps[:,self.initial_tolerance]>=0]).any():
                    print('There are no large extrapolations.')
            extra_steps = (extrapolation_timesteps[:,n_small:].T-extrapolation_timesteps[:,small]).T
            extra_steps[extra_steps<0] = structure_extrapolation+1
            extra_steps_reduced = extra_steps[extrapolation_timesteps[:,small]!=-1]
            tolerance_indices = self.initial_tolerance*np.ones(len(extra_steps), dtype=int)
            tolerance_indices[extrapolation_timesteps[:,small]==-1] = -1
            tolerance_indices_reduced = tolerance_indices[extrapolation_timesteps[:,small]!=-1]
            for i in range(self.initial_tolerance-n_small, n_tolerances-n_small):
                tolerance_indices_reduced[extra_steps_reduced[:,i]<structure_extrapolation] += 1
            tolerance_indices_reduced[tolerance_indices_reduced>=n_tolerances] = -1
            tolerance_indices[extrapolation_timesteps[:,small]!=-1] = tolerance_indices_reduced
            tolerance_indices[tolerance_indices>=small] -= n_small

        selected_timesteps = []
        smalls = small*np.ones(len(extrapolation_timesteps), dtype=int)
        min_interpolated_structure_checks = 3
        for i in range(len(extrapolation_timesteps)):
            if extrapolation_timesteps[i][small]<0 and extrapolation_data[i][4]!=self.N_steps:
                print('WARNING: A simulation ended due to too many extrapolations but no one of these was larger than the tolerance of {0}. If this message is printed several times you should consider to reduce the first and second entry of tolerances.'.format(self.tolerances[small]))
                if small>0 and extrapolation_timesteps[i][small-1]>=structure_extrapolation:
                    smalls[i] = small-1
                    print('With the reduction of the tolerance to {0} an extrapolated structure could be found in this case.'.format(self.tolerances[smalls[i]]))
            if extrapolation_timesteps[i][smalls[i]]>=0:
                if extrapolation_timesteps[i][smalls[i]]>(min_interpolated_structure_checks+2)*structure_checks:
                    selected_timesteps.append(list(range(2*structure_checks, extrapolation_timesteps[i][smalls[i]]-structure_checks+1, structure_checks))+[extrapolation_timesteps[i][smalls[i]], deepcopy(extrapolation_timesteps[i][n_small:])])
                else:
                    small_timestep_separation_interpolation_checks = ((extrapolation_timesteps[i][smalls[i]]//(min_interpolated_structure_checks+2))//extrapolation_data[i][2])*extrapolation_data[i][2]
                    n_interpolation_checks = min_interpolated_structure_checks
                    while small_timestep_separation_interpolation_checks<structure_interpolation and n_interpolation_checks>1:
                        n_interpolation_checks -= 1
                        small_timestep_separation_interpolation_checks = (extrapolation_timesteps[i][smalls[i]]//(n_interpolation_checks+2)//extrapolation_data[i][2])*extrapolation_data[i][2]
                    if small_timestep_separation_interpolation_checks>structure_interpolation:
                        selected_timesteps.append([j*small_timestep_separation_interpolation_checks for j in range(2,n_interpolation_checks+2)]+[extrapolation_timesteps[i][smalls[i]], deepcopy(extrapolation_timesteps[i][n_small:])])
                    else:
                        selected_timesteps.append([extrapolation_timesteps[i][smalls[i]], deepcopy(extrapolation_timesteps[i][n_small:])])
            else:
                if extrapolation_data[i][4]>(min_interpolated_structure_checks+2)*structure_checks:
                    selected_timesteps.append(list(range(2*structure_checks, extrapolation_data[i][4]+1, structure_checks))+[-1, (n_tolerances-n_small)*[-1]])
                else:
                    small_timestep_separation_interpolation_checks = ((extrapolation_data[i][4]//(min_interpolated_structure_checks+2))//extrapolation_data[i][2])*extrapolation_data[i][2]
                    n_interpolation_checks = min_interpolated_structure_checks
                    while small_timestep_separation_interpolation_checks<structure_interpolation and n_interpolation_checks>1:
                        n_interpolation_checks -= 1
                        small_timestep_separation_interpolation_checks = (extrapolation_data[i][4]//(n_interpolation_checks+2)//extrapolation_data[i][2])*extrapolation_data[i][2]
                    if small_timestep_separation_interpolation_checks>structure_interpolation:
                        selected_timesteps.append([j*small_timestep_separation_interpolation_checks for j in range(2,n_interpolation_checks+2)]+[(extrapolation_data[i][4]//extrapolation_data[i][2])*extrapolation_data[i][2], -1, (n_tolerances-n_small)*[-1]])
                        print('Included the last regularly dumped structure of the simulation as it ended due to too many extrapolations.')
                    else:
                        if (extrapolation_data[i][4]//extrapolation_data[i][2])*extrapolation_data[i][2]>=structure_extrapolation:
                            selected_timesteps.append([(extrapolation_data[i][4]//extrapolation_data[i][2])*extrapolation_data[i][2], -1, (n_tolerances-n_small)*[-1]])
                            print('Included the last regularly dumped structure of the simulation as it ended due to too many extrapolations.')
                        else:
                            selected_timesteps.append([-1, (n_tolerances-n_small)*[-1]])

        return selected_timesteps, tolerance_indices, smalls, n_small


    def _get_structure(self, data):
        '''
        '''
        lat = np.array([data[5].split(), data[6].split(), data[7].split()]).astype(float)
        if data[4].startswith('ITEM: BOX BOUNDS xy xz yz pp pp pp'):
            lx = lat[0][1]-lat[0][0]-np.array([0.0,lat[0][2],lat[1][2],lat[0][2]+lat[1][2]]).max()+np.array([0.0,lat[0][2],lat[1][2],lat[0][2]+lat[1][2]]).min()
            ly = lat[1][1]-lat[1][0]-np.array([0.0,lat[2][2]]).max()+np.array([0.0,lat[2][2]]).min()
            lz = lat[2][1]-lat[2][0]
            lattice = [[lx, 0.0, 0.0], [lat[0][2], ly, 0.0], [lat[1][2], lat[2][2], lz]]
        else:
            lattice = [[(lat[0][1]-lat[0][0]), 0.0, 0.0], [0.0, (lat[1][1]-lat[1][0]), 0.0], [0.0, 0.0, (lat[2][1]-lat[2][0])]]

        atom_style = 'atomic'
        if data[8].startswith('ITEM: ATOMS id element x y z q') or data[8].startswith('ITEM: ATOMS id element xu yu zu q'):
            atom_style = 'full'
        data = np.array([line.split() for line in data[9:]])
        element = deepcopy(data[:,1])
        position = deepcopy(data[:,2:5]).astype(float)
        if atom_style=='full':
            charge = deepcopy(data[:,5]).astype(float)
        else:
            charge = np.zeros(len(element))

        return lattice, element, position, charge


    def _check_nearest_neighbours(self, lat, pos_i, pos_j, ii, d_min):
        '''
        '''
        if len(pos_i)==0 or len(pos_j)==0:
            return True, -1

        if pos_i.ndim==1:
            pos_i = np.array([pos_i])
        if pos_j.ndim==1:
            pos_j = np.array([pos_j])

        pos = np.array(deepcopy(pos_j))
        pos = np.concatenate((pos,
                                np.dstack((pos[:,0]-lat[0][0]-lat[1][0]-lat[2][0], pos[:,1]-lat[0][1]-lat[1][1]-lat[2][1], pos[:,2]-lat[0][2]-lat[1][2]-lat[2][2]))[0],
                                np.dstack((pos[:,0]-lat[0][0]-lat[1][0]          , pos[:,1]-lat[0][1]-lat[1][1]          , pos[:,2]-lat[0][2]-lat[1][2]          ))[0],
                                np.dstack((pos[:,0]-lat[0][0]-lat[1][0]+lat[2][0], pos[:,1]-lat[0][1]-lat[1][1]+lat[2][1], pos[:,2]-lat[0][2]-lat[1][2]+lat[2][2]))[0],
                                np.dstack((pos[:,0]-lat[0][0]          -lat[2][0], pos[:,1]-lat[0][1]          -lat[2][1], pos[:,2]-lat[0][2]          -lat[2][2]))[0],
                                np.dstack((pos[:,0]-lat[0][0]                    , pos[:,1]-lat[0][1]                    , pos[:,2]-lat[0][2]                    ))[0],
                                np.dstack((pos[:,0]-lat[0][0]          +lat[2][0], pos[:,1]-lat[0][1]          +lat[2][1], pos[:,2]-lat[0][2]          +lat[2][2]))[0],
                                np.dstack((pos[:,0]-lat[0][0]+lat[1][0]-lat[2][0], pos[:,1]-lat[0][1]+lat[1][1]-lat[2][1], pos[:,2]-lat[0][2]+lat[1][2]-lat[2][2]))[0],
                                np.dstack((pos[:,0]-lat[0][0]+lat[1][0]          , pos[:,1]-lat[0][1]+lat[1][1]          , pos[:,2]-lat[0][2]+lat[1][2]          ))[0],
                                np.dstack((pos[:,0]-lat[0][0]+lat[1][0]+lat[2][0], pos[:,1]-lat[0][1]+lat[1][1]+lat[2][1], pos[:,2]-lat[0][2]+lat[1][2]+lat[2][2]))[0],
                                np.dstack((pos[:,0]          -lat[1][0]-lat[2][0], pos[:,1]          -lat[1][1]-lat[2][1], pos[:,2]          -lat[1][2]-lat[2][2]))[0],
                                np.dstack((pos[:,0]          -lat[1][0]          , pos[:,1]          -lat[1][1]          , pos[:,2]          -lat[1][2]          ))[0],
                                np.dstack((pos[:,0]          -lat[1][0]+lat[2][0], pos[:,1]          -lat[1][1]+lat[2][1], pos[:,2]          -lat[1][2]+lat[2][2]))[0],
                                np.dstack((pos[:,0]                    -lat[2][0], pos[:,1]                    -lat[2][1], pos[:,2]                    -lat[2][2]))[0],
                                np.dstack((pos[:,0]                    +lat[2][0], pos[:,1]                    +lat[2][1], pos[:,2]                    +lat[2][2]))[0],
                                np.dstack((pos[:,0]          +lat[1][0]-lat[2][0], pos[:,1]          +lat[1][1]-lat[2][1], pos[:,2]          +lat[1][2]-lat[2][2]))[0],
                                np.dstack((pos[:,0]          +lat[1][0]          , pos[:,1]          +lat[1][1]          , pos[:,2]          +lat[1][2]          ))[0],
                                np.dstack((pos[:,0]          +lat[1][0]+lat[2][0], pos[:,1]          +lat[1][1]+lat[2][1], pos[:,2]          +lat[1][2]+lat[2][2]))[0],
                                np.dstack((pos[:,0]+lat[0][0]-lat[1][0]-lat[2][0], pos[:,1]+lat[0][1]-lat[1][1]-lat[2][1], pos[:,2]+lat[0][2]-lat[1][2]-lat[2][2]))[0],
                                np.dstack((pos[:,0]+lat[0][0]-lat[1][0]          , pos[:,1]+lat[0][1]-lat[1][1]          , pos[:,2]+lat[0][2]-lat[1][2]          ))[0],
                                np.dstack((pos[:,0]+lat[0][0]-lat[1][0]+lat[2][0], pos[:,1]+lat[0][1]-lat[1][1]+lat[2][1], pos[:,2]+lat[0][2]-lat[1][2]+lat[2][2]))[0],
                                np.dstack((pos[:,0]+lat[0][0]          -lat[2][0], pos[:,1]+lat[0][1]          -lat[2][1], pos[:,2]+lat[0][2]          -lat[2][2]))[0],
                                np.dstack((pos[:,0]+lat[0][0]                    , pos[:,1]+lat[0][1]                    , pos[:,2]+lat[0][2]                    ))[0],
                                np.dstack((pos[:,0]+lat[0][0]          +lat[2][0], pos[:,1]+lat[0][1]          +lat[2][1], pos[:,2]+lat[0][2]          +lat[2][2]))[0],
                                np.dstack((pos[:,0]+lat[0][0]+lat[1][0]-lat[2][0], pos[:,1]+lat[0][1]+lat[1][1]-lat[2][1], pos[:,2]+lat[0][2]+lat[1][2]-lat[2][2]))[0],
                                np.dstack((pos[:,0]+lat[0][0]+lat[1][0]          , pos[:,1]+lat[0][1]+lat[1][1]          , pos[:,2]+lat[0][2]+lat[1][2]          ))[0],
                                np.dstack((pos[:,0]+lat[0][0]+lat[1][0]+lat[2][0], pos[:,1]+lat[0][1]+lat[1][1]+lat[2][1], pos[:,2]+lat[0][2]+lat[1][2]+lat[2][2]))[0]), axis=0)

        if ii:
            select = 1
        else:
            select = 0

        for p in pos_i:
            d = np.dstack((pos[:,0]-p[0], pos[:,1]-p[1], pos[:,2]-p[2]))[0]
            d = np.sqrt(d[:,0]**2+d[:,1]**2+d[:,2]**2)
            d = d[d.argsort()[select]]
            if d < d_min:
                return False, d

        return True, -1


    def _check_structure(self, lattice, element, position, d_mins, path, timestep):
        '''
        '''
        N = len(self.element_types)
        for i in range(N):
            for j in range(i,N):
                accepted, d = self._check_nearest_neighbours(lattice, position[element==self.element_types[i]], position[element==self.element_types[j]], i==j, d_mins[self.element_types[i]][j-i])
                if not accepted:
                    print('Too small interatomic distance in {0}_s{1}: {2}-{3}: {4} Ang'.format(path, timestep, self.element_types[i], self.element_types[j], d))
                    return False

        return True


    def _read_structure(self, data, path, timestep, d_mins):
        '''
        '''
        lattice, element, position, charge = self._get_structure(data)
        accepted = self._check_structure(lattice, element, position, d_mins, path, timestep)
        if accepted:
            self.names.append(path+'_s'+str(timestep))
            self.lattices.append(lattice)
            self.elements.append(element)
            self.positions.append(position)
            self.charges.append(charge)

        return accepted


    def _read_structures(self, 
                         path,
                         extrapolation_data,
                         selected_timestep,
                         n_small,
                         small,
                         tolerance_index,
                         extrapolation_statistic,
                         element2index,
                         index: int):
        '''
        '''
        structure_extrapolation = self.min_timestep_separation_extrapolation[index]
        d_min = self.d_mins[index]
        with open('{0}/mode1/{1}/structure.lammpstrj'.format(self.active_learning_directory, path)) as f:
            data = [line.strip() for line in f.readlines()]

        n_interpolation_checks = len(selected_timestep)-2
        if n_interpolation_checks>0:
            for i in range(n_interpolation_checks):
                if selected_timestep[i]>=0:
                    if selected_timestep[i]!=int(data[3]):
                        start = data.index(str(selected_timestep[i]))-1
                    else:
                        start = 1
                        while selected_timestep[i]!=int(data[start]):
                            start += extrapolation_data[3]
                        start -= 1
                    end = start+extrapolation_data[3]
                    accepted = self._read_structure(data[start:end], path, selected_timestep[i], d_mins=d_min)
                    if accepted:
                        self.statistics.append([])
                    data = data[end:]

        if selected_timestep[-2]>=structure_extrapolation:
            if selected_timestep[-2]!=int(data[3]):
                start = data.index(str(selected_timestep[-2]))-1
            else:
                start = 1
                while selected_timestep[-2]!=int(data[start]):
                    start += extrapolation_data[3]
                start -= 1
            end = start+extrapolation_data[3]
            accepted = self._read_structure(data[start:end], path, selected_timestep[-2], d_mins=d_min)
            if accepted:
                extrapolation_statistic[small] = np.array(extrapolation_statistic[small])
                extrapolation_statistic[small][:,1] = np.array([element2index[self.elements[-1][extrapolation_statistic[small][i,1].astype(int)-1]] for i in range(len(extrapolation_statistic[small]))])
                extrapolation_statistic[small] = extrapolation_statistic[small][extrapolation_statistic[small][:,0].argsort()]
                extrapolation_statistic[small] = extrapolation_statistic[small][extrapolation_statistic[small][:,2].argsort(kind='mergesort')]
                extrapolation_statistic[small] = extrapolation_statistic[small][extrapolation_statistic[small][:,1].argsort(kind='mergesort')]
                self.statistics.append(['small', str(list(np.array(self.element_types)[extrapolation_statistic[small][:,1].astype(int)])).strip('[]').replace("'",''), str(list(extrapolation_statistic[small][:,2].astype(int)+1)).strip('[]'), str([round(j, 5) for j in extrapolation_statistic[small][:,0]]).strip('[]')])

        accepted = False
        while not accepted and tolerance_index>=0:
            if selected_timestep[-1][tolerance_index]>=structure_extrapolation:
                if selected_timestep[-1][tolerance_index]!=int(data[3]):
                    start = data.index(str(selected_timestep[-1][tolerance_index]))-1
                else:
                    start = 1
                    while selected_timestep[-1][tolerance_index]!=int(data[start]):
                        start += extrapolation_data[3]
                    start -= 1
                end = start+extrapolation_data[3]
                accepted = self._read_structure(data[start:end], path, selected_timestep[-1][tolerance_index], d_mins=d_min)
            else:
                tolerance_index = -1
            if not accepted:
                tolerance_index -= 1
                if selected_timestep[-1][tolerance_index]-structure_extrapolation<selected_timestep[-2]:
                    tolerance_index = -1
        if accepted:
            extrapolation_statistic[tolerance_index+n_small] = np.array(extrapolation_statistic[tolerance_index+n_small])
            extrapolation_statistic[tolerance_index+n_small][:,1] = np.array([element2index[self.elements[-1][extrapolation_statistic[tolerance_index+n_small][i,1].astype(int)-1]] for i in range(len(extrapolation_statistic[tolerance_index+n_small]))])
            extrapolation_statistic[tolerance_index+n_small] = extrapolation_statistic[tolerance_index+n_small][extrapolation_statistic[tolerance_index+n_small][:,0].argsort()]
            extrapolation_statistic[tolerance_index+n_small] = extrapolation_statistic[tolerance_index+n_small][extrapolation_statistic[tolerance_index+n_small][:,2].argsort(kind='mergesort')]
            extrapolation_statistic[tolerance_index+n_small] = extrapolation_statistic[tolerance_index+n_small][extrapolation_statistic[tolerance_index+n_small][:,1].argsort(kind='mergesort')]
            self.statistics.append(['large', str(list(np.array(self.element_types)[extrapolation_statistic[tolerance_index+n_small][:,1].astype(int)])).strip('[]').replace("'",''), str(list(extrapolation_statistic[tolerance_index+n_small][:,2].astype(int)+1)).strip('[]'), str([round(j, 5) for j in extrapolation_statistic[tolerance_index+n_small][:,0]]).strip('[]')])

        return tolerance_index


    def _write_data(self,
                    names,
                    lattices,
                    elements,
                    positions,
                    charges,
                    file_name: str,
                    mode: str):
        '''
        '''
        with open(file_name, mode) as f:
            # Make sure we have a trailing newline if appending to file
            if mode == 'a+':
                if f.readlines()[-1].strip() == '':
                    f.write('\n')

            for i in range(len(names)):
                f.write('begin\ncomment file {0}\n'.format(names[i]))
                if list(self.statistics[i]):
                    f.write('comment statistics {0}\ncomment statistics {1}\ncomment statistics {2}\ncomment statistics {3}\n'.format(self.statistics[i][0], self.statistics[i][1], self.statistics[i][2], self.statistics[i][3]))
                if self.periodic:
                    f.write('lattice {0:>9.5f} {1:>9.5f} {2:>9.5f}\nlattice {3:>9.5f} {4:>9.5f} {5:>9.5f}\nlattice {6:>9.5f} {7:>9.5f} {8:>9.5f}\n'.format(round(lattices[i][0][0]/Bohr2Ang,5), round(lattices[i][0][1]/Bohr2Ang,5), round(lattices[i][0][2]/Bohr2Ang,5), round(lattices[i][1][0]/Bohr2Ang,5), round(lattices[i][1][1]/Bohr2Ang,5), round(lattices[i][1][2]/Bohr2Ang,5), round(lattices[i][2][0]/Bohr2Ang,5), round(lattices[i][2][1]/Bohr2Ang,5), round(lattices[i][2][2]/Bohr2Ang,5)))
                for j in range(len(elements[i])):
                    f.write('atom {0:>9.5f} {1:>9.5f} {2:>9.5f} {3:2} {4:>6.3f} 0.0 0.0 0.0 0.0\n'.format(round(positions[i][j][0]/Bohr2Ang,5), round(positions[i][j][1]/Bohr2Ang,5), round(positions[i][j][2]/Bohr2Ang,5), elements[i][j], charges[i][j]))
                f.write('energy 0.0\ncharge 0.0\nend\n')


    def _print_reliability(self, extrapolation_timesteps, smalls, tolerance_indices, paths):
        '''
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            small_extrapolation_timesteps = np.diagonal(extrapolation_timesteps.T[smalls])
            extrapolation_timesteps_reduced = small_extrapolation_timesteps[small_extrapolation_timesteps!=-1]
            paths_reduced = paths[small_extrapolation_timesteps!=-1]
            median_small = np.median(extrapolation_timesteps_reduced)
            if np.isfinite(median_small):
                median_small = int(round(median_small,0))
                median_small_1 = np.median(extrapolation_timesteps_reduced[np.flatnonzero(np.core.defchararray.find(paths_reduced,'_hdnnp1_')!=-1)])
                if np.isfinite(median_small_1):
                    median_small_1 = int(round(median_small_1,0))
                median_small_2 = np.median(extrapolation_timesteps_reduced[np.flatnonzero(np.core.defchararray.find(paths_reduced,'_hdnnp2_')!=-1)])
                if np.isfinite(median_small_2):
                    median_small_2 = int(round(median_small_2,0))
                print('The median number of time steps to a small extrapolation is {0} (HDNNP_1: {1}, HDNNP_2: {2}).'.format(median_small, median_small_1, median_small_2))
            extra_steps = np.diagonal(extrapolation_timesteps.T[tolerance_indices])[tolerance_indices>=0]-small_extrapolation_timesteps[tolerance_indices>=0]
            if not np.isscalar(extra_steps):
                paths_reduced = paths[tolerance_indices>=0]
                median_extra_steps = np.median(extra_steps)
                if np.isfinite(median_extra_steps):
                    median_extra_steps = int(round(median_extra_steps,0))
                    median_extra_steps_1 = np.median(extra_steps[np.flatnonzero(np.core.defchararray.find(paths_reduced,'_hdnnp1_')!=-1)])
                    if np.isfinite(median_extra_steps_1):
                        median_extra_steps_1 = int(round(median_extra_steps_1,0))
                    median_extra_steps_2 = np.median(extra_steps[np.flatnonzero(np.core.defchararray.find(paths_reduced,'_hdnnp2_')!=-1)])
                    if np.isfinite(median_extra_steps_2):
                        median_extra_steps_2 = int(round(median_extra_steps_2,0))
                    print('The median number of time steps between the first and second selected extrapolated structure is {0} (HDNNP_1: {1}, HDNNP_2: {2}).'.format(median_extra_steps, median_extra_steps_1, median_extra_steps_2))


    def _read_data(self):
        '''
        '''
        names = []
        lattices = []
        elements = []
        positions = []
        charges = []
        statistics = []
        with open('input.data-new') as f:
            for line in f.readlines():
                if line.startswith('atom'):
                    line = line.split()
                    elements[-1].append(line[4])
                    positions[-1].append(line[1:4])
                    charges[-1].append(line[5])
                elif line.startswith('lattice'):
                    lattices[-1].append(line.strip().split()[1:4])
                elif line.startswith('comment file'):
                    names.append(line.strip().split()[2])
                elif line.startswith('comment statistics'):
                    statistics[-1].append(line.strip()[19:])
                elif line.startswith('begin'):
                    lattices.append([])
                    elements.append([])
                    positions.append([])
                    charges.append([])
                    statistics.append([])
                elif line.startswith('end'):
                    lattices[-1] = np.array(lattices[-1]).astype(float)*Bohr2Ang
                    elements[-1] = np.array(elements[-1])
                    positions[-1] = np.array(positions[-1]).astype(float)*Bohr2Ang
                    charges[-1] = np.array(charges[-1]).astype(float)
        names = np.array(names)
        lattices = np.array(lattices)
        elements = np.array(elements)
        positions = np.array(positions)
        charges = np.array(charges)
        statistics = np.array(statistics)

        return names, lattices, elements, positions, charges, statistics


    def _print_performance(self, n_calculations):
        '''
        '''
        time = []
        for input_name in ['/mode2/HDNNP_1/mode_2.out', '/mode2/HDNNP_2/mode_2.out']:
            with open(self.active_learning_directory + input_name) as f:
                file_time = [line.strip().split() for line in f.readlines() if line.startswith('TIMING Training loop finished:')]
                if len(file_time) > 0 and len(file_time[0]) > 1:
                    time.append(float(file_time[0][-2]))
        unit = ['s', 's']
        for i in range(2):
            if time[i]>=60.0:
                time[i] /= 60.0
                unit[i] = 'min'
                if time[i]>=60.0:
                    time[i] /= 60.0
                    unit[i] = 'h'
        print('\nTime to calculate {0} structures using RuNNer: HDNNP_1: {1} {2}, HDNNP_2: {3} {4}.\n'.format(n_calculations, round(time[0],2), unit[0], round(time[1],2), unit[1]))


    def _read_energies(self, input_name):
        '''
        '''
        with open(input_name) as f:
            # readline "pops" the first line so all indexes should decrease by 1
            line = f.readline().strip()
            if line.startswith('point'):
                energies = np.array([line.strip().split()[2] for line in f.readlines()]).astype(float)
                energies = np.dstack((np.arange(len(energies)),energies))[0]
            elif line.startswith('Conf.'):
                energies = np.array([np.array(line.strip().split())[[1,3]] for line in f.readlines()]).astype(float)
                energies = energies[:,1]/energies[:,0]
                energies = np.dstack((np.arange(len(energies)),energies))[0]
            elif line.startswith('###'):
                energies = np.array([line.strip().split()[-1] for line in f.readlines()[11:]]).astype(float)
                energies = np.dstack((np.arange(len(energies)),energies))[0]
            else:
                raise IOError('Unknown RuNNer format')
                

        return energies


    def _read_forces(self, input_name):
        '''
        '''
        with open(input_name) as f:
            line = f.readline().strip()
            if line.startswith('point'):
                forces = np.array([np.array(line.strip().split())[[0,4]] for line in f if line.strip()]).astype(float)
                forces[:,0] -= 1
            elif line.startswith('Conf.'):
                forces = np.array([np.array(line.strip().split())[[0,5,6,7]] for line in f if line.strip()]).astype(float)
                forces = np.concatenate((forces[:,[0,1]], forces[:,[0,2]], forces[:,[0,3]]))
                forces[:,0] -= 1
            elif line.startswith('###'):
                forces = []
                for line in f.readlines()[12:]:
                    text = line.strip().split()
                    forces.append([text[0], text[-1]])
                forces = np.array(forces).astype(float)
            else:
                raise IOError('Unknown RuNNer format')
                

        return forces


    def _reduce_selection(self, selection, max_interpolated_structures_per_simulation, structure_name_index, steps, indices):
        '''
        '''
        steps = np.array(steps)
        steps_difference = steps[1:]-steps[:-1]
        min_separation = steps_difference.min()
        if min_separation<self.timestep_separation_interpolation_checks[structure_name_index]:
            selection = selection[selection!=indices[1]]
        else:
            n_steps = len(steps)
            min_timestep_separation_interpolation_checks = (n_steps//max_interpolated_structures_per_simulation+1)*self.timestep_separation_interpolation_checks[structure_name_index]
            j = 1
            while j<n_steps-1:
                if steps_difference[j]<=min_timestep_separation_interpolation_checks:
                    selection = selection[selection!=indices[j]]
                    j += 1
                j += 1

        return selection


    def _improve_selection(self,
                           selection,
                           statistics,
                           names,
                           all_extrapolated_structures,
                           max_extrapolated_structures,
                           max_interpolated_structures_per_simulation,
                           exceptions,
                           structure_name_indices):
        '''
        '''
        current_name = None
        steps = []
        indices = []
        for i in range(len(names)):
            if list(statistics[i]):
                if all_extrapolated_structures[structure_name_indices[i]]:
                    selection = np.append(selection, i)
            elif i in selection:
                name = names[i].split('_s')
                if current_name==name[0]:
                    steps.append(int(name[1]))
                    indices.append(i)
                else:
                    if len(steps)>2:
                        selection = self._reduce_selection(selection, max_interpolated_structures_per_simulation[structure_name_indices[indices[0]]], structure_name_indices[indices[0]], self.timestep_separation_interpolation_checks, steps, indices)
                    current_name = name[0]
                    steps = [int(name[1])]
                    indices = [i]
        if len(steps)>2:
            selection = self._reduce_selection(selection, max_interpolated_structures_per_simulation[structure_name_indices[indices[0]]], structure_name_indices[indices[0]], self.timestep_separation_interpolation_checks, steps, indices)
        selection = np.unique(selection)

        if any(max_extrapolated_structures) or any(exceptions):
            statistics_reduced = statistics[selection]
            structure_name_indices_reduced = structure_name_indices[selection]
            structure_name_indices_reduced = np.array([structure_name_indices_reduced[i] for i in range(len(structure_name_indices_reduced)) if list(statistics_reduced[i]) and statistics_reduced[i][0]=='small']).astype(str)
            if list(structure_name_indices_reduced):
                statistics_reduced = np.array([i for i in statistics_reduced if list(i) and i[0]=='small'])
                statistics_reduced = np.core.defchararray.add(np.core.defchararray.add(np.core.defchararray.add(np.core.defchararray.add(structure_name_indices_reduced, ';'), statistics_reduced[:,1]), ';'), statistics_reduced[:,2])
                statistics_unique = np.unique(statistics_reduced)
                counts = {}
                for i in statistics_unique:
                    counts[i] = 0
                for i in statistics_reduced:
                    counts[i] += 1
                exception_list = {}

                if any(max_extrapolated_structures):
                    for i in statistics_unique:
                        structure_index = int(i.split(';')[0])
                        if max_extrapolated_structures[structure_index]!=0:
                            if counts[i]>max_extrapolated_structures[structure_index]:
                                exception_list[i] = np.concatenate((np.ones(max_extrapolated_structures[structure_index], dtype=int), np.zeros(counts[i]-max_extrapolated_structures[structure_index], dtype=int)))
                                np.random.shuffle(exception_list[i])
                                print('The extrapolation [\'{0}\', \'{1}\'] occurred {2} times.'.format(i.split(';')[1], i.split(';')[2], counts[i]))

                if any(exceptions):
                    exceptions_unique = []
                    for i in range(len(exceptions)):
                        if exceptions[i]!=None:
                            for j in range(len(exceptions[i])):
                                exceptions_unique.append([str(i)+';'+exceptions[i][j][0]+';'+exceptions[i][j][1], exceptions[i][j][2]])
                    counts_keys = counts.keys()
                    for i in exceptions_unique:
                        if i[0] in counts_keys:
                            keep = int(round(i[1]*counts[i[0]]))
                            exception_list[i[0]] = np.concatenate((np.ones(keep, dtype=int), np.zeros(counts[i[0]]-keep, dtype=int)))

                exception_list_keys = exception_list.keys()
                if list(exception_list_keys):
                    structure_name_indices_reduced = structure_name_indices[selection]
                    statistics_reduced = np.array(list(statistics[selection]))
                    for i in range(len(selection)):
                        if list(statistics_reduced[i]) and statistics_reduced[i][0]=='small':
                            key = str(structure_name_indices_reduced[i])+';'+statistics_reduced[i][1]+';'+statistics_reduced[i][2]
                            if key in exception_list_keys:
                                if exception_list[key][-1]==0:
                                    selection[i] = -1
                                exception_list[key] = np.delete(exception_list[key], -1, 0)
                    selection = np.unique(selection)
                    if selection[0] == -1:
                        selection = selection[1:]

        return selection


    def _print_statistics(self, selection, statistics, names, structure_names):
        '''
        '''
        if structure_names!=None and len(structure_names)>1:
            for structure_name in structure_names:
                print('Structure: {0}'.format(structure_name))
                n_extrapolations = int(np.array([1 for name in names[selection] if name.split('_')[0]==structure_name]).sum())
                print('{0} missing structures were identified.'.format(n_extrapolations))
                statistics_reduced = np.array([statistics[selection][i][0] for i in range(len(statistics[selection])) if names[selection][i].split('_')[0]==structure_name and list(statistics[selection][i])])
                if len(statistics_reduced)>0:
                    n_small_extrapolations = int(np.array([1 for i in statistics_reduced if i=='small']).sum())
                    n_large_extrapolations = int(np.array([1 for i in statistics_reduced if i=='large']).sum())
                    print('{0} missing structures originate from small extrapolations.\n{1} missing structures originate from large extrapolations.'.format(n_small_extrapolations, n_large_extrapolations))
        else:
            print('{0} missing structures were identified.'.format(len(selection)))
            statistics_reduced = np.array([i[0] for i in statistics[selection] if list(i)])
            if len(statistics_reduced)>0:
                n_small_extrapolations = int(np.array([1 for i in statistics_reduced if i=='small']).sum())
                n_large_extrapolations = int(np.array([1 for i in statistics_reduced if i=='large']).sum())
                print('{0} missing structures originate from small extrapolations.\n{1} missing structures originate from large extrapolations.'.format(n_small_extrapolations, n_large_extrapolations))
        statistics = np.array([i for i in statistics[selection] if list(i)])
        if list(statistics):
            self._analyse_extrapolation_statistics(statistics)


    def _analyse_extrapolation_statistics(self, statistics):
        '''
        '''
        elements = []
        for line in statistics[:,1]:
            if ', ' in line:
                elements.extend(line.split(', '))
            else:
                elements.append(line)
        elements = np.array(elements)
        symmetry_functions = []
        for line in statistics[:,2]:
            if ', ' in line:
                symmetry_functions.extend(line.split(', '))
            else:
                symmetry_functions.append(line)
        symmetry_functions = np.array(symmetry_functions).astype(int)
        values = []
        for line in statistics[:,3]:
            if ', ' in line:
                values.extend(line.split(', '))
            else:
                values.append(line)
        values = np.array(values).astype(float)
        element_list = np.unique(elements)
        for e in element_list:
            symfunc = symmetry_functions[elements==e]
            symfunc_list = np.unique(symfunc)
            with open(self.active_learning_directory + '/extrapolation_statistics_'+e+'.dat', 'w') as f:
                for s in symfunc_list:
                    val = values[elements==e][symfunc==s]
                    for v in val:
                        f.write('{0} {1}\n'.format(s, v))


    def prepare_data_new(self):
        """
        """
        # TODO add in an overwrite check?
        self.lattices = []
        self.elements = []
        self.charges = []
        self.statistics = []
        self.names = []
        self.positions = []
        if self.structure_names==None:
            n_structure_names = 1
        else:
            n_structure_names = len(self.structure_names)
        for i in range(n_structure_names):
            if self.structure_names==None:
                paths = self._get_paths('')
            else:
                print('Structure: {0}'.format(self.structure_names[i]))
                paths = self._get_paths(self.structure_names[i])
            extrapolation_data = []
            extrapolation_timesteps = []
            extrapolation_values = []
            extrapolation_statistics = []
            extrapolation_format = self._read_log_format(paths[0])
            for path in paths:
                extrapolation_data.append(self._read_extrapolation(path))
                extrapolation_timestep, extrapolation_value, extrapolation_statistic = self._read_log(path, extrapolation_data[-1], extrapolation_format)
                extrapolation_timesteps.append(extrapolation_timestep)
                extrapolation_values.append(extrapolation_value)
                extrapolation_statistics.append(extrapolation_statistic)
            extrapolation_timesteps = np.array(extrapolation_timesteps).astype(int)
            extrapolation_values = np.array(extrapolation_values)
            selected_timesteps, tolerance_indices, smalls, n_small = self._get_timesteps(extrapolation_timesteps,
                                                                                         extrapolation_values,
                                                                                         extrapolation_data,
                                                                                         index=i)
            element2index = {}
            for j in range(len(self.element_types)):
                element2index[self.element_types[j]] = j
            for j in range(len(paths)):
                tolerance_indices[j] = self._read_structures(paths[j],
                                                             extrapolation_data[j],
                                                             selected_timesteps[j],
                                                             n_small,
                                                             smalls[j],
                                                             tolerance_indices[j],
                                                             extrapolation_statistics[j],
                                                             element2index,
                                                             index=i)
            self._print_reliability(extrapolation_timesteps, smalls, tolerance_indices, paths)
        self.names = np.array(self.names)
        self.lattices = np.array(self.lattices)
        self.elements = np.array(self.elements)
        self.positions = np.array(self.positions)
        self.charges = np.array(self.charges)
        self.statistics = np.array(self.statistics)
        print("Writing {} names to input.data-new".format(len(self.names)))
        self._write_data(self.names,
                         self.lattices,
                         self.elements,
                         self.positions,
                         self.charges,
                         file_name=self.active_learning_directory + '/input.data-new',
                         mode='w')


    def prepare_data_add(self):
        """
        """
        if not isdir(self.active_learning_directory + '/mode2'):
            raise IOError('`mode2` directory not found.')

        self._print_performance(len(self.names))
        if self.structure_names==None:
            structure_name_indices = np.zeros(len(self.names), dtype=int)
        else:
            # TODO This is tricky, because name has _ in it but so does my comment structure name
            # structure_name_indices = np.array([self.structure_names.index(name.split('_')[0]+'_'+name.split('_')[1]) for name in self.names])
            structure_name_indices = np.array([self.structure_names.index(name.split('_')[0]) for name in self.names])
        energies_1 = self._read_energies(self.active_learning_directory + '/mode2/HDNNP_1/trainpoints.000000.out')
        energies_2 = self._read_energies(self.active_learning_directory + '/mode2/HDNNP_2/trainpoints.000000.out')
        forces_1 = self._read_forces(self.active_learning_directory + '/mode2/HDNNP_1/trainforces.000000.out')
        forces_2 = self._read_forces(self.active_learning_directory + '/mode2/HDNNP_2/trainforces.000000.out')
        dE = np.array([self.delta_E[structure_name_index] for structure_name_index in structure_name_indices])
        dF = np.array([self.delta_F[structure_name_indices[i]] for i in range(len(structure_name_indices)) for j in range(3*len(self.positions[i]))])
        energies = energies_1[np.absolute(energies_2[:,1]-energies_1[:,1])>dE,0]
        forces = forces_1[np.absolute(forces_2[:,1]-forces_1[:,1])>dF,0]
        self.selection = np.unique(np.concatenate((energies, forces)).astype(int))
        self.selection = self._improve_selection(self.selection,
                                                 self.statistics,
                                                 self.names, 
                                                 self.all_extrapolated_structures,
                                                 self.max_extrapolated_structures,
                                                 self.max_interpolated_structures_per_simulation,
                                                 self.exceptions,
                                                 structure_name_indices)
        self._write_data(self.names[self.selection],
                         self.lattices[self.selection],
                         self.elements[self.selection],
                         self.positions[self.selection],
                         self.charges[self.selection],
                         file_name=self.active_learning_directory + '/input.data-add',
                         mode='w')
        self._print_statistics(self.selection, self.statistics, self.names, self.structure_names)


    def combine_data_add(self):
        """
        """
        for directory in self.n2p2_directories:
            self._write_data(self.names[self.selection],
                             self.lattices[self.selection],
                             self.elements[self.selection],
                             self.positions[self.selection],
                             self.charges[self.selection],
                             file_name=directory + '/input.data',
                             mode='a+')
