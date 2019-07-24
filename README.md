# Installing and Running PyTorch on the HPC Clusters

Here we install PyTorch on the GPU clusters, using SOL at Lehigh Univ as an example.

## Clone the repo

Log in to a head node on one of the GPU clusters. Then clone the repo using:

```
git clone https://github.com/sha256feng/install_pytorch.git
```

This will create a folder called `install_pytorch` which contains the files needed to follow this tutorial.

## Make a conda environment and install

Next we create a conda environment that includes pytorch and its dependencies (note that you may consider replacing the environment name "pytorch_tutorial" with something more specific to your work):

If you have anaconda3 installed in the local environment: 
```
conda update conda # This proves to be important, some conda version had problems installing pytorch
conda create --name torch_env pytorch torchvision cudatoolkit=9.0 -c pytorch
```

Or you can opt to use Anaconda module on SOL:
```
module load anaconda3
conda create --name torch_env pytorch torchvision cudatoolkit=9.0 -c pytorch
```


While we have a newer version of the CUDA toolkit installed on the HPC clusters, PyTorch recommends version 9.

Once the command above completes, as long as you have the Anaconda installed or  `anaconda3` module loaded (current session only,
you'll note that we load it in the Slurm script `mnist.slurm`),
you'll have access to `conda` and can use it to access the Python virtual environment you just created.

Activate the conda environment:

```
conda activate pytorch_tutorial
```

Let's make sure our installation can find the GPU by launching an interactive session on one of the compute nodes:

On SOL cluster, managers recommend using "srun" rather than "salloc". So instead of using
```
salloc -t 00:05:00 --gres=gpu:1 # We don't use this. Princeton Univ's cluster uses this.
```
We use 
```
srun -t 2:00:00 -p im2080-gpu --nodes=1 --ntasks-per-node=2 --gres=gpu:1 --pty /bin/bash
```

When your allocation is granted, you'll be moved to a compute node. Execute the following command on the compute node to test for GPU support:

```
conda activate torch_env # since we move to a new node
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If the output is "True" and "GETX 2080 Ti" (on partition 'im2080') then your installation of PyTorch is GPU-enabled. Type `exit` to return to the head node.


## Running the example

The compute nodes do not have internet access so we must obtain the data in advance. Run the `mnist_download.py` script from the `install_pytorch` directory on the head node:

```
python mnist_download.py
python minist_classification.py
```

Now that you have the data, you can schedule the job using the following command:

```
sbatch mnist.slurm
```

This will request one GPU, 5 minutes of computing time, and queue the job. You should receive a job number and can check if your job is running or queued
via `squeue -u <your-username>`.

Once the job runs, you'll have a `slurm-xxxxx.out` file in the `install_pytorch` directory. This log file contains both Slurm and PyTorch messages.

## Examining GPU utilization

To see how effectively your job is using the GPU, immediately after submiting the job run the following command:

```
squeue -u <your-username>
```

The rightmost column labeled "NODELIST(REASON)" gives the name of the node where your job is running. SSH to this node:

```
ssh sol-exxx
```

Once on the compute node run `watch -n 1 gpustat`. This will show you a percentage value indicating how effectively your code is using the GPU. The memory allocated to the GPU is also available. For this specific example you will see that only about 10% of the GPU cores are utilized. Given that a CNN is being trained on small images (i.e., 28x28 pixels) this is not surprising. You should repeat this analysis with your actual research script to ensure that your GPUs are nearly fully utilized.

Type `Ctrl+C` to exit the `watch` screen. Type `exit` to return to the head node.

## More examples

More PyTorch example scripts are found here:
```
https://github.com/pytorch/examples
```

This folder was adapted from Princeton University's "install_pytorch" tutorial. 
