import subprocess
import itertools
import os
import time

# This is design for python 2.6
if __name__ == "__main__":

    mem = 48                        # memory in Gb
    num_cpus = 10                   # number of cpu cores to use
    sh_hours = 32                    # time in hour
    mem_per_gpu = 1             # the number of memory the job will use per gpu
    task_name = "test_hmm"            # name of the task
    conda_path = "/rigel/home/lw2827/miniconda3/bin"
    env_name = "ptc"
    execution_path = "/rigel/home/lw2827/SocialBehaviors"
    py_script_path = "/rigel/home/lw2827/SocialBehaviors/SocialBehaviorptc/runner_hmm.py"

    sh_name = "runner_hmm.sh"

    params_dict = {}

    # need to modify the job name further
    params_dict["job_name"] = ["0930_hmm"]

    params_dict["train_model"] = [True]
    params_dict["k"] = [10]
    params_dict["downsample_n"] = [2]
    params_dict["video_clip"] = [[0,5]]
    params_dict["n_x"] = [8]
    params_dict["n_y"] = [8]
    params_dict["list_of_num_iters"] = [[20,10]]
    params_dict["list_of_lr"] = [[0.1,0.05]]
    params_dict["sample_t"] = [90000]
    params_dict["pbar_update_interval"] = [1]



    # --------------------- parameters part ends --------------------- #
    param_keys = list(params_dict.keys())
    param_values = list(params_dict.values())
    param_vals_permutation = list(itertools.product(*param_values))

    for i, param_vals in enumerate(param_vals_permutation):
        args = ""
        arg_dict = {}
        for param_name, param_val in zip(param_keys, param_vals):
            # flag arguments
            if param_name == "train_model" or param_name == "load_model":
                if param_val:
                    args += " --{}".format(param_name)
                continue
            if isinstance(param_val, list):
                param_val = ",".join([str(x) for x in param_val])
            arg_dict[param_name] = param_val
            
            if param_name == "job_name":
                continue

            args += " --{0}={1} ".format(param_name, param_val)

        arg_dict["job_name"] += "/v{}{}_K{}_{}by{}".format(arg_dict["video_clip"][0], arg_dict["video_clip"][2],
                                                           arg_dict["k"], arg_dict["n_x"], arg_dict["n_y"])

        args += " --job_name={0}".format(arg_dict["job_name"])

        # create shell script
        with open(sh_name, "w") as f:
            f.write("#!/bin/sh\n")
            f.write("#SBATCH --account=stats\n")
            f.write("SBATCH --job-name={}\n".format(task_name))
            f.write("SBATCH -c {}".format(num_cpus))
            f.write("SBATCH --time={}:00:00".format(sh_hours))
            f.write("SBATCH --mem-per-cpu={}gb".format(mem_per_gpu))

            f.write("cd {0}\n".format(conda_path))
            f.write("source activate {0}\n".format(env_name))
            f.write("cd {0}\n".format(execution_path))
            f.write("python {0} {1}\n".format(py_script_path, args))
            f.write("cd {0}\n".format(conda_path))
            f.write("source deactivate")

        # execute the shell script
        #subprocess.Popen("sbatch {0}".format(sh_name), shell=True)
        #time.sleep(5)
