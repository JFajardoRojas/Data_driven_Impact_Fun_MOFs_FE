### Simulation Guide

1. Run the lammps input file in.FL_path 	| This generates the required outputs.

2. Run the python code FL_path_FE.py with temperature as first argument, and the volume in Å^3 as the second argument. 
The volume is obtained in one of the outputs in the previous step.
Example:
	python FL_path_FE.py 300.0 `cat volume.dat`

Using the bash python.sl, the rest of the arguments are hardcoded.

3. The output FL_FE.txt should be: Small differences can occurred due to differences in seed numbers.

         N            FE_T             W_T         FE_harm             W_F             W_B               Q            F_CM
    8192.0      4.79579444      5.29296008     -0.49716563      5.22805374     -5.35786642      0.06490634    -16.53793123

### Lammps extra requirements
Lammps built with module USER-MISC
This is required for UFF4MOF Fourier angle terms, and ti/spring fix.

### Python requirements
As stated in the SI.
