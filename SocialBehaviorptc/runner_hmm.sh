#!/bin/sh
#SBATCH --account=stats
SBATCH --job-name=test_hmm
SBATCH -c 10SBATCH --time=32:00:00SBATCH --mem-per-cpu=1gbcd /rigel/home/lw2827/miniconda3/bin
source activate ptc
cd /rigel/home/lw2827/SocialBehaviors
python /rigel/home/lw2827/SocialBehaviors/SocialBehaviorptc/runner_hmm.py  --train_model --k=10  --downsample_n=2  --video_clip=0,5  --n_x=8  --n_y=8  --list_of_num_iters=20,10  --list_of_lr=0.1,0.05  --sample_t=90000  --pbar_update_interval=1  --job_name=0930_hmm/v05_K10_8by8
cd /rigel/home/lw2827/miniconda3/bin
source deactivate