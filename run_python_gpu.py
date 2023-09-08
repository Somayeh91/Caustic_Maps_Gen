import argparse
import os
import time
import subprocess

date = time.strftime("%y-%m-%d-%H-%M-%S")

""" COURTESY OF LINDSEY KWOK & MAX NEWMAN
Usage: python make_remote_jupyter_lab_session.py <optional partition; default is p_km1243_1> <full directory path where jupyter lab is launched; default is /projects/f_km1243_1/> <conda environment name> <port on amarel to use; default is 8888> 

"""


def parse_options():
    """Function to handle options speficied at command line      
    """
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('-partition', dest='partition', action='store', default='skylake',
                        help='which partition to run job on')
    parser.add_argument('-directory', action='store', default='/home/skhakpas/codes',
                        help='Specify the directory the in which jupyter lab should launch.')
    parser.add_argument('-shell_name', action='store', default='~/.bash_profile',
                        help='Shell where conda environment path is defined.')
    parser.add_argument('-conda_env', action='store', default='base',
                        help='Name of conda environment the jupyter lab should run in.')
    # parser.add_argument('-r_port', action='store', default='8888', help='Amarel (remote) port to use.')
    parser.add_argument('-python_file', action='store', default='', help='Specify python file to run.')
    parser.add_argument('-mem', action='store', default='40000', help='Needed memory.')
    parser.add_argument('-pythonFileArguments', action='store', default='', help='Arguments for the python file you are'
                                                                                 'running.')
    parser.add_argument('-n_gpu', action='store', default=1, help='Number of requested gpus')
    parser.add_argument('-n_hrs', action='store', default=100, help='Number of hours gpus')

    # Parses through the arguments and saves them within the keyword args
    args = parser.parse_args()
    return args


args = parse_options()
partition = args.partition
email = subprocess.check_output('echo $USER', shell=True).decode('ascii')[:-1]
jobname = f'gpu_{time.strftime("%m-%d-%H-%M")}'
outfilename = f'python_{date}'
cwd = args.directory
shell_name = args.shell_name
conda_env = args.conda_env
python_file = args.python_file
mem = args.mem
arguments = args.pythonFileArguments
n_gpu = args.n_gpu
n_hrs = args.n_hrs
# rport = args.r_port
script = """#!/bin/bash

#SBATCH --partition={partition}       # Partition (job queue)
#SBATCH --job-name={jobname}          # Assign an short name to your job
#SBATCH --nodes=1                     # Number of nodes you require
#SBATCH --ntasks=1                    # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1             # Cores per task (>1 if multithread tasks)
#SBATCH --mem={mem}                 # Real memory (RAM) required (MB)
#SBATCH --time={n_hrs}:00:00               # Total run time limit (HH:MM:SS)
#SBATCH --output={outfilename}.out    # STDOUT output file
#SBATCH --error={outfilename}.err     # STDERR output file (optional)
#SBATCH --export=ALL                  # Export you current env to the job env
#SBATCH --mail-type=END
#SBATCH --mail-user=somayeh.khakpash@gmail.com
#SBATCH --gres=gpu:{n_gpu}                  # Number of GPUs
cd {cwd}
source {shell_name}
module load gcc/11.3.0
module load openmpi/4.1.4
module load python/3.10.4
module load numpy/1.22.3-scipy-bundle-2022.05
module load matplotlib/3.5.2 
module load pandas/1.4.2-scipy-bundle-2022.05
module load tensorflow/2.11.0-cuda-11.7.0
module load cuda/12.0.0
module load tqdm/4.64.0
module load scipy/1.8.1-scipy-bundle-2022.05 
. ~/maps/bin/activate
python {python_file} -date={date} {arguments}
""".format(partition=partition, python_file=python_file, date=date, arguments=arguments, mem=mem,
           n_gpu=n_gpu, n_hrs=n_hrs, jobname=jobname, outfilename=outfilename,
           email=email, cwd=cwd, shell_name=shell_name, conda_env=conda_env)

# write the slurm script to file
slurm_script_file = 'slurm_' + outfilename
with open(slurm_script_file, 'w') as f:
    f.write(script)
# and now submit the job
os.system('sbatch {}'.format(slurm_script_file))
