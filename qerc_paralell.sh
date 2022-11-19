#!/bin/bash

# for N=10 can use about 3.5% of memory on this workstation
njobs=25

source $HOME/venv/qutip_qist/bin/activate

# Export so child processes can see variables
export exec_folder=$HOME/GitHub/qerc
export filename_root=/scratch/NemotoU/henry/qerc/data/
export model='ising'
export dt=0.5
export tf=10
export hsize_initial=500
export hsize_final=6000
export hsize_step=500

# Reservoir and elm of output
qerc() {
  alpha=$1
  N=$2
  g=$3

  outfile_evolve=evolve_${model}_N_${N}_g_${g}_alpha_${alpha}.out
  outfile_elm=elm_${model}_N_${N}_g_${g}_alpha_${alpha}.out

  python3 ${exec_folder}/mnist_evolve.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -dt ${dt} -tf ${tf} \
	  &> ${outfile_evolve}

  python3 ${exec_folder}/mnist_observe.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -delete_qu 'True' \
	  &>> ${outfile_evolve}
  
  python3 ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -hsize_initial ${hsize_initial} -hsize_final ${hsize_final} -hsize_step ${hsize_step} \
	  -filename_root ${filename_root} -model ${model} -node_type 'rho_diag' \
	  &> ${outfile_elm}

  python3 ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -node_type 'rho_diag' -activation 'identity' \
	  &>> ${outfile_elm}
    
  python3 ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -hsize_initial ${hsize_initial} -hsize_final ${hsize_final} -hsize_step ${hsize_step} \
	  -filename_root ${filename_root} -model ${model} -node_type 'psi' \
	  &>> ${outfile_elm}

  python3 ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -node_type 'psi' -activation 'identity' -pinv 'numpy' \
	  &>> ${outfile_elm}
    
  python3 ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -hsize_initial ${hsize_initial} -hsize_final ${hsize_final} -hsize_step ${hsize_step} \
	  -filename_root ${filename_root} -model ${model} -node_type 'corr' \
	  &>> ${outfile_elm}

  python3 ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -node_type 'corr' -activation 'identity' -pinv 'numpy' \
	  &>> ${outfile_elm}
}
export -f qerc

# Run in parallel (indexed as alpha, N, g) using GNU parallel
#parallel -j${njobs} --dry-run qerc ::: 1.51 10000 ::: $(seq 5 10) ::: $(seq 0 0.1 1.5)
parallel -j${njobs} qerc ::: 1.51 10000 ::: $(seq 5 10) ::: $(seq 0 0.1 1.5)
