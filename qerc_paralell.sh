#!/bin/bash

N_arr=$(seq 5 11)
#g_arr=$(seq 0.91 0.01 1.09)
g_arr=$(seq 0 0.1 1.5)
alpha_arr=(1.51 10000)

# for N=10 can use about 3.5% of memory on this workstation
#njobs=25
njobs=32

source $HOME/venv/qutip_qist/bin/activate

# Export so child processes can see variables
export exec_folder=$HOME/GitHub/qerc
export filename_root=/scratch/NemotoU/henry/qerc/data/
export model='ising'
export dt=0.5
export tf=5
export hsize_initial=500
export hsize_final=4000
export hsize_step=500

# Reservoir
evolve() {
  alpha=$1
  N=$2
  g=$3

  outfile_evolve=evolve_${model}_N_${N}_g_${g}_alpha_${alpha}.out

  python3 -u ${exec_folder}/mnist_evolve.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -dt ${dt} -tf ${tf} \
	  &> ${outfile_evolve}

  python3 -u ${exec_folder}/mnist_observe.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -delete_qu 'True' \
	  &>> ${outfile_evolve}
}
export -f evolve

# Different kinds of elm
elm() {  
  alpha=$1
  N=$2
  g=$3
  
  outfile_elm=elm_${model}_N_${N}_g_${g}_alpha_${alpha}.out
  
  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -hsize_initial ${hsize_initial} -hsize_final ${hsize_final} -hsize_step ${hsize_step} \
	  -filename_root ${filename_root} -model ${model} -node_type 'rho_diag' \
	  &> ${outfile_elm}

  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -node_type 'rho_diag' -activation 'identity' \
	  &>> ${outfile_elm}
    
  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -hsize_initial ${hsize_initial} -hsize_final ${hsize_final} -hsize_step ${hsize_step} \
	  -filename_root ${filename_root} -model ${model} -node_type 'psi' \
	  &>> ${outfile_elm}

  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -node_type 'psi' -activation 'identity' \
	  &>> ${outfile_elm}
    
  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -hsize_initial ${hsize_initial} -hsize_final ${hsize_final} -hsize_step ${hsize_step} \
	  -filename_root ${filename_root} -model ${model} -node_type 'corr' \
	  &>> ${outfile_elm}

  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -node_type 'corr' -activation 'identity' \
	  &>> ${outfile_elm}
}
export -f elm

perceptron() { 
  alpha=$1
  N=$2
  g=$3

  outfile_per=perceptron_${model}_N_${N}_g_${g}_alpha_${alpha}.out

  python3 -u ${exec_folder}/mnist_perceptron.py -N ${N} -g ${g} -alpha ${alpha} \
          -filename_root ${filename_root} -model ${model} -node_type 'rho_diag' \
          &> ${outfile_per}

  python3 -u ${exec_folder}/mnist_perceptron.py -N ${N} -g ${g} -alpha ${alpha} \ 
          -filename_root ${filename_root} -model ${model} -node_type 'psi' \
          &>> ${outfile_per}

  python3 -u ${exec_folder}/mnist_perceptron.py -N ${N} -g ${g} -alpha ${alpha} \
          -filename_root ${filename_root} -model ${model} -node_type 'corr' \
          &>> ${outfile_per}
}
export -f perceptron

# Run in parallel (indexed as alpha, N, g) using GNU parallel
parallel -j${njobs} evolve ::: ${alpha_arr} ::: ${N_arr} ::: ${g_arr}

parallel -j${njobs} elm ::: ${alpha_arr} ::: ${N_arr} ::: ${g_arr}

parallel -j${njobs} perceptron ::: ${alpha_arr} ::: ${N_arr} ::: ${g_arr}
