import numpy as np
import sys
import scipy.constants as sc

def F_EC(mks, volume, T):

	"""
		calculates Einstein crystal free energy
		mks is an array or list with each row being: [atom LAMMPS type, number of atoms of this type, type mass, type force constant, type msd]	
		volume is the equilibrium volume in Å^3
		T is temperature in K
	"""

	kb_kcal = sc.value('Boltzmann constant in eV/K') * 23.06054 # kb in kcal / mol K
	kb = sc.value('Boltzmann constant') # kb in SI base units
	mu = sc.value('atomic mass constant') # atomic mass constant
	hbar = sc.value('Planck constant over 2 pi') # hbar in SI base units
	Av = sc.value('Avogadro constant') # Avagadro's number

	# calculate the EC free energy

	F_harm = 0.0
	natoms = 0
	total_mass = 0.0
	com_correction_data = []

	for ty, num_type, mass, force_k, msd in mks:
			
		natoms += num_type

		total_mass += mass * num_type
		com_correction_data.append((num_type, mass, force_k)) # this is list data used to compute COM correction below
			
		force_k_SI = force_k * 4184 * (1.0/Av) * (1.0e20) # convert force_k to SI units [J/m^2]
		omega = np.sqrt((force_k_SI)/(mass*mu)) # frequency in Hz
		F_harm += 3 * num_type * kb_kcal * T * np.log((hbar*omega)/(kb*T)) # sum over each atom, the summand in the same within each type, output in kcal/mol
		
	# calculate the fixed COM correction

	mass_term = 0.0
	total_mass_squared = total_mass**2

	for num_type, mass, force_k in com_correction_data:

		mass_term += (num_type * (mass**2))/(total_mass_squared * force_k)

	F_cmco = (1.5 * kb_kcal * T) * np.log((2 * np.pi * kb_kcal * T) * mass_term) + (kb_kcal * T * np.log(1.0/volume))

	return natoms, F_harm, F_cmco

def integrate_switching(forward, backward):

	"""
		integrates the forward and backward switching output 
		the lambda value should be axis 1
		dH/dlamdba should be axis 0
	"""

	I_forw = np.trapz(y=forward[:,0], x=forward[:,1])
	I_back = np.trapz(y=backward[:,0], x=backward[:,1])

	Q = -(I_forw + I_back) / 2.0
	W =  (I_forw - I_back) / 2.0

	return W, Q, I_forw, I_back

if __name__ == '__main__':

	### arguments
	forward = np.genfromtxt(sys.argv[1], delimiter = '')
	backward = np.genfromtxt(sys.argv[2], delimiter = '')
	force_k_data = np.genfromtxt(sys.argv[3], delimiter = '')
	T = float(sys.argv[4])
	volume = float(sys.argv[5])
	
	### check to make sure full forward and backward integrations finished properly, ofcourse there can be other problems
	if len(forward) != len(backward):
		raise ValueError('forward and backward switching lengths are not equal')
	
	### calculate free energy
	natoms, FE_harm, F_CM = F_EC(force_k_data, volume, T)
	natoms = float(natoms)
	W, Q, I_F, I_B = integrate_switching(forward, backward)
	FE_T = W + FE_harm # add the EC free energy to the work calculated (integration)
	
	### print results
	fs = '{:>10} {:>15.8f} {:>15.8f} {:>15.8f} {:>15.8f} {:>15.8f} {:>15.8f} {:>15.8f}'
	print('{:>10} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}'.format('N', 'FE_T', 'W_T', 'FE_harm', 'W_F', 'W_B', 'Q', 'F_CM'))
	print(fs.format(natoms, FE_T/natoms, W/natoms, FE_harm/natoms, I_F/natoms, I_B/natoms, Q/natoms, F_CM))
