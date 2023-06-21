import argparse
import os
import time
import subprocess

date = time.strftime("%y-%m-%d-%H-%M-%S")
# os.mkdir('./results/' + str(date))

"""COURTESY OF LINDSEY KWOK & MAX NEWMAN Usage: python make_remote_jupyter_lab_session.py <optional partition; 
default is p_km1243_1> <full directory path where jupyter lab is launched; default is /projects/f_km1243_1/> <conda 
environment name> <port on amarel to use; default is 8888> Example: python make_remote_jupyter_lab_session.py 
-partition=main -directory=/scratch/dr919 -conda_env=base -r_port=9999 


After running this script, run the terminal command 'squeue -u <username>' where <username> is your amarel username. 
Locate the node that the session is currently on (i.e. hal0164). Open a new terminal window and type the following to 
create a tunnel between amarel and your local computer por: ssh -L 8888:slepner016:9999 dr919@amarel.rutgers.edu 

The SSH command with the optional argument -L specified first the local port, in the example 8888, the remote host 
name (slepner016), and the remote port (9999) specified when you run this script (with r_port). 

Finally, in your local browser window, type 'localhost:<localport>'. 

Notes: (1) If you already have a local jupyter session running, chances are it's running on the port 8888. Instead, 
when you run the SSH command, use 8889, or 9999. (2) You may need to enter a token when you first launch the jupyter 
session in your browser. On amarel, locate the file: jupyter_<data submitted>.err where the <date_submitted> is 
formatted as year_month_day. In this file you should find a line with a long token code. Paste this into your local 
browser where indicated. In the future, you can set a password rather than having to locate a new token each time. If 
you are having trouble setting the password, return to the initial amarel window where you ran the script and running 
the command: jupyter server password <desired_password>. Then return to the local jupyter session in your browser and 
enter the password (you may need to scancel the job and re-run everything after the password has been set.) """


def parse_options():
    """Function to handle options speficied at command line      
    """
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('-partition', dest='partition', action='store', default='skylake',
                        help='which partition to run job on')
    parser.add_argument('-directory', action='store', default='/home/skhakpash/',
                        help='Specify the directory the in which jupyter lab should launch.')
    parser.add_argument('-output/directory', action='store', default='/home/skhakpash/',
                        help='Specify the directory the in which jupyter lab should launch.')
    parser.add_argument('-shell_name', action='store', default='~/.bash_profile',
                        help='Shell where conda environment path is defined.')
    parser.add_argument('-conda_env', action='store', default='base',
                        help='Name of conda environment the jupyter lab should run in.')
    parser.add_argument('-python_file', action='store', default='', help='Specify python file to run.')
    parser.add_argument('-time', action='store', default='40:00:00', help='Time')
    parser.add_argument('-mem', action='store', default='32000', help='Needed memory.')
    # Parses through the arguments and saves them within the keyword args
    arguments = parser.parse_args()
    return arguments


args = parse_options()
partition = args.partition
email = subprocess.check_output('echo $USER', shell=True).decode('ascii')[:-1]
jobname = f'python_{email}'
outfilename = f'python_{date}'
mem = args.mem
time = args.time
cwd = args.directory
shell_name = args.shell_name
conda_env = args.conda_env
python_file = args.python_file
output_direc = cwd + 'results/'+ str(date) + '/'
# rport = args.r_port
script = """#!/bin/bash

#SBATCH --partition={partition}       # Partition (job queue)
#SBATCH --job-name={jobname}          # Assign an short name to your job
#SBATCH --nodes=1                     # Number of nodes you require
#SBATCH --ntasks=1                    # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1             # Cores per task (>1 if multithread tasks)
#SBATCH --mem={mem}                   # Real memory (RAM) required (MB)
#SBATCH --time={time}               # Total run time limit (HH:MM:SS)
#SBATCH --output={outfilename}.out    # STDOUT output file
#SBATCH --error={outfilename}.err     # STDERR output file (optional)
#SBATCH --export=ALL                  # Export you current env to the job env
#SBATCH --mail-type=END
#SBATCH --mail-user=somayeh.khakpash@gmail.com

cd {cwd}
source {shell_name}
module load gcc/11.3.0
module load openmpi/4.1.4
module load python/3.10.4
module load numpy/1.22.3-scipy-bundle-2022.05
module load matplotlib/3.5.2 
module load pandas/1.4.2-scipy-bundle-2022.05
module load tqdm/4.64.0
module load scipy/1.8.1-scipy-bundle-2022.05 
. ~/maps/bin/activate

python {python_file}
""".format(partition=partition, python_file=python_file, jobname=jobname, mem=mem, time=time,
           outfilename=output_direc+outfilename,
           cwd=cwd, shell_name=shell_name)

# write the slurm script to file
slurm_script_file = 'slurm_' + outfilename
with open(slurm_script_file, 'w') as f:
    f.write(script)
# and now submit the job
os.system('sbatch {}'.format(slurm_script_file))
