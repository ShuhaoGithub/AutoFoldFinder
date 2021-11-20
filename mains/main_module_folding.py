import os, sys, glob, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

import tensorflow as tf
import numpy as np
import pandas as pd
# from data_loader.data_generator import DataGenerator
from prokit.models.background_model import BackgroundModel
from prokit.models.trrosetta_models import TrRosettaModelKL, TrRosettaModelPred
from prokit.trainers.background_trainer import BackgroundTrainer
from prokit.trainers.trrosetta_trainers import TrRosettaTrainer
from prokit.optimizers.MCMC_optimizer import MCMCOptimizer
from prokit.utils.config import process_exp_config
from prokit.utils.dirs import create_dirs
from prokit.utils.logger import *
from prokit.utils.args import get_fold_args, get_args_setup
from prokit.utils.protein_utils import *
from prokit.generators.protein_sequence import generateOneRandomSequence, getBackgroundAAComposition, aa2idx
from prokit.folding.trRosetta import trRosetta 
import traceback


def run_folding():
    # parameters
    try:
        config, config_dict = get_args_setup()
        config = process_exp_config(config)
        os.environ["debug"] = str(config.debug)
    except:
        xprint("missing or invalid arguments")
        xprint(traceback.format_exc())
        exit(0)

    # Create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.output_dir])
    curNum = len(glob.glob(f"{config.output_dir}/trajectory_*.csv"))

    for count in range(curNum):
        output_csv = os.path.join(config.output_dir, f"trajectory_{count}.csv")
        output_fasta = os.path.join(config.output_dir, f"final_seq_{count}.fasta")
        output_npz = os.path.join(config.output_dir, f"final_seq_{count}.npz")
        output_model = os.path.join(config.output_dir, f"final_seq_{count}_model.pdb")

        if not os.path.exists(output_model):
            config.seq_fasta_path = output_fasta
            config.contact_path =  output_npz
            config.pred_model_path =  output_model

            xprint(json.dumps(config, indent= 4))
            run_folding_with_pyrosetta(config, config_dict)

@timeit
def run_folding_with_pyrosetta(inputConfig, inputConfigDict):
    with open(inputConfig.folding_params_file, "r") as f:
        fold_params = json.load(f)
    fold_params["PCUT"] = inputConfig.folding_pcut
    fold_params["USE_ORIENT"] = inputConfig.folding_use_orient

    fold_args = get_fold_args(inputConfigDict)
    fold_args.NPZ= inputConfig.contact_path
    fold_args.FASTA = inputConfig.seq_fasta_path
    fold_args.OUT = inputConfig.pred_model_path
    fold_args.wdir = os.path.join(inputConfig.output_dir, "temp")
    
    xprint(json.dumps(fold_args, indent= 4))

    os.makedirs(fold_args.wdir, exist_ok=True)
    trRosetta.run_folding(fold_args, fold_params)

if __name__ == "__main__":
    run_folding()