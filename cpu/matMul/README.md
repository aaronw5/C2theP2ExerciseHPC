
## Matrix Multiplication on CPU
### Install packages:
```
python3 -m venv ~/c2p2

. ~/c2p2/bin/activate

pip install mpi4py

pip install numba

pip install numpy

```
### Run matrix multiplication of 500 x 500 matrix without parallelization.

edit `runMatMulCPU.sh` to run on 1 node using vim, gromacs... by setting the value after select to 1: `#PBS -l select=1:system=crux`

Run the script: `qsub runMatMulCPU.sh`

Go into logs and look at the 
