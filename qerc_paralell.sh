#!/bin/bash

N_arr=$(seq 5 11)
#g_arr=$(seq 0.90 0.01 1.1)
#g_arr=$(seq 0 0.1 2)
g_arr=$(seq 0 0.5 5)
alpha_arr=(1.51 10000)

# for N=10 can use about 3.5% of memory on this workstation
njobs=20
#njobs=32

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

encode() {
  N=$1
  
  outfile=encode_N_${N}.out
  
  python3 -u ${exec_folder}/mnist_encode.py -N ${N} \
	  -filename_root ${filename_root} \
	  &> ${outfile}
}
export -f encode

evolve() {
  alpha=$1
  N=$2
  g=$3

  outfile=evolve_${model}_N_${N}_g_${g}_alpha_${alpha}.out

  python3 -u ${exec_folder}/mnist_evolve.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -dt ${dt} -tf ${tf} \
	  &> ${outfile}

  python3 -u ${exec_folder}/mnist_observe.py -N ${N} -g ${g} -alpha ${alpha} \
	  -filename_root ${filename_root} -model ${model} -delete_qu 'True' \
	  &>> ${outfile}
}
export -f evolve

elm() {  
  alpha=$1
  N=$2
  g=$3
  
  outfile=elm_${model}_N_${N}_g_${g}_alpha_${alpha}.out
  
  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -hsize_initial ${hsize_initial} -hsize_final ${hsize_final} -hsize_step ${hsize_step} \
	  -filename_root ${filename_root} -model ${model} -node_type 'rho_diag' \
	  &> ${outfile}

    
  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -hsize_initial ${hsize_initial} -hsize_final ${hsize_final} -hsize_step ${hsize_step} \
	  -filename_root ${filename_root} -model ${model} -node_type 'psi' \
	  &>> ${outfile}

    
  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
	  -hsize_initial ${hsize_initial} -hsize_final ${hsize_final} -hsize_step ${hsize_step} \
	  -filename_root ${filename_root} -model ${model} -node_type 'corr' \
	  &>> ${outfile}
}
export -f elm

identity() {
  alpha=$1
  N=$2
  g=$3
  
  outfile=identity_${model}_N_${N}_g_${g}_alpha_${alpha}.out
  
  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
          -activation 'identity' -standardize 'True' -pinv 'numpy' \
	  -filename_root ${filename_root} -model ${model} -node_type 'rho_diag' \
	  &>> ${outfile}

  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
          -activation 'identity' -standardize 'True' -pinv 'numpy' \
	  -filename_root ${filename_root} -model ${model} -node_type 'psi' \
	  &>> ${outfile}
  
  python3 -u ${exec_folder}/mnist_elm.py -N ${N} -g ${g} -alpha ${alpha} \
          -activation 'identity' -standardize 'True' -pinv 'numpy' \
	  -filename_root ${filename_root} -model ${model} -node_type 'corr' \
	  &>> ${outfile}
}
export -f identity

perceptron() { 
  alpha=$1
  N=$2
  g=$3

  outfile=perceptron_${model}_N_${N}_g_${g}_alpha_${alpha}.out

  python3 -u ${exec_folder}/mnist_perceptron.py -N ${N} -g ${g} -alpha ${alpha} \
          -filename_root ${filename_root} -model ${model} -node_type 'rho_diag' \
          &> ${outfile}

  python3 -u ${exec_folder}/mnist_perceptron.py -N ${N} -g ${g} -alpha ${alpha} \
          -filename_root ${filename_root} -model ${model} -node_type 'psi' \
          &>> ${outfile}

  python3 -u ${exec_folder}/mnist_perceptron.py -N ${N} -g ${g} -alpha ${alpha} \
          -filename_root ${filename_root} -model ${model} -node_type 'corr' \
          &>> ${outfile}
}
export -f perceptron

# Run in parallel (indexed as alpha, N, g) using GNU parallel
parallel -j${njobs} --memsuspend 2G encode ::: "${N_arr[@]}"

parallel -j${njobs} --memsuspend 2G evolve ::: "${alpha_arr[@]}" ::: "${N_arr[@]}" ::: "${g_arr[@]}"

parallel -j${njobs} --memsuspend 2G perceptron ::: "${alpha_arr[@]}" ::: "${N_arr[@]}" ::: "${g_arr[@]}"

parallel -j${njobs} --memsuspend 2G identity ::: "${alpha_arr[@]}" ::: "${N_arr[@]}" ::: "${g_arr[@]}"

parallel -j${njobs} --memsuspend 2G elm ::: "${alpha_arr[@]}" ::: "${N_arr[@]}" ::: "${g_arr[@]}"