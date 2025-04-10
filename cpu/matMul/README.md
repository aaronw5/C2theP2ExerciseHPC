
##Using PBS  
```qsub <file> submits a batch job
qstat -u $USEr check the status of your jobs
qstat` check the status of ALL jobs on the system
qdel <job number> deletes job
```
## Matrix Multiplication on CPU
### Install packages:
make logs folder: mkdir logs

```
python3 -m venv ~/c2p2

. ~/c2p2/bin/activate

pip install mpi4py

pip install numba

pip install numpy

```

### Run matrix multiplication matrix without parallelization.

edit `runMatMulCPU.sh` to run on 1 node using vim, gromacs... by setting the value after select to 1: `#PBS -l select=1:system=crux`

Run the script: `qsub runMatMulCPU.sh`

Go into logs and look at the `matMul.out` output. Notice how the output is printed multiple times. Why is that?

Now, run it on 4 nodes. Do you expect it to run faster? slower? or the same?

Look at output logs and compare. 


### Run Matrix multiplication with parallelization across CPU nodes

edit `runMatMulMPI.sh` to run on 1 node and submit the job using `qsub runMatMulMPI.sh`
