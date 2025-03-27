# EAS6995

Files for the EAS 6995 course at Cornell University, covering Deep Learning in the Earth Sciences.
This repository, in addition to the course work itself, contains the custom matplotlib stylesheet I use in my figures for reproducability.

### NCAR HPC

1. To `ssh` into Derecho and Casper, I set up my `.ssh/config` so that I can bypass having to type `ssh your_username@derecho.hpc.ucar.edu` and just do `ssh derecho`. After establishing a connection, I just need to enter my password and confirm the Duo push notification. I connect to remote servers using VSCode, so that I can have my interactive environment available.

2. I created a simple PBS shell script to run a Python file, which looks like

```
#!/bin/bash
#PBS -N hello_ncar
#PBS -A UCOR0090
#PBS -j oe
#PBS -k eod
#PBS -q casper
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=1:mpiprocs=1

### Set temp to scratch
export TMPDIR=${SCRATCH}/${USER}/temp && mkdir -p $TMPDIR

cd /glade/u/home/aidenz/Documents/EAS6995/hello_ncar

### specify desired module environment
module purge
module load conda
conda activate /glade/u/apps/opt/conda/envs/npl-2025a

### Compile & Run MPI Program
python /glade/u/home/aidenz/Documents/EAS6995/hello_ncar/hello.py
```

__IMPORTANT:__ To run on derecho, change `#PBS -q casper` to `#PBS -q main`.

This can be run using `qsub <script name>`. The status of all active runs can be checked using `qstat -u <username>` and all ended scripts `qstat -x -u <username>`.

3. Transfering files can be done using `scp` or `sftp`. An example to go from local to derecho would be to run `scp local_file <username>@derecho.hpc.ucar.edu:/path/to/hpc/folder`.

4. To run an interactive job on casper, do `qsub -I -l select=1:ncpus=1:mem=20GB:ngpus=1 -q casper@casper-pbs -l walltime=00:10:00 -A <ACCOUNT #>` and for derecho `qsub -I -l select=1:ncpus=1:mem=20GB:ngpus=1 -q main@desched1 -l walltime=00:10:00 -A <ACCOUNT #>`