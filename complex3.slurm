#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --array=1-5
#SBATCH --job-name="cgp_config_3_run"
#SBATCH --output=config_3_run.out
#SBATCH --mail-user=jonora@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load Anaconda3/2020.07
pip install numpy --user
pip install pyyaml --user
pip install pygame --user
pip install gym --user
python /cluster/home/jonora/CGP_Neuron_Masters/runfile.py config_complex_3 "/cluster/work/jonora/CGP_Neuron_masters/logfiles" $SLURM_ARRAY_TASK_ID

uname -a