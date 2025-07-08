#!/bin/bash
#SBATCH --job-name=h4_v2_sanity
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a30:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1-00:00:00

mkdir -p logs

export RUST_BACKTRACE=FULL 
export RUST_LOG=trace 

cargo build --release --bin study
srun ./target/release/study ref4_p1
#mv /tmp/* /scratch/ajn6-amg/
