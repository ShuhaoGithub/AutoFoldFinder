import os, sys, glob, json
import traceback
# import warnings
# warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

import tensorflow as tf
import numpy as np
import pandas as pd

# from data_loader.data_generator import DataGenerator
from prokit.models.background_model import BackgroundModel
from prokit.models.trrosetta_models_advance import TrRosettaModelKL_close, TrRosettaModelPred
from prokit.trainers.background_trainer import BackgroundTrainer
from prokit.trainers.trrosetta_trainers import TrRosettaTrainer
from prokit.optimizers.MCMC_optimizer_advance import MCMCOptimizer
from prokit.utils.config import process_exp_config
from prokit.utils.dirs import create_dirs
from prokit.utils.logger import *
from prokit.utils.args import get_fold_args, get_args_setup
from prokit.utils.protein_utils import *
from prokit.generators.protein_sequence import generateOneRandomSequence, getBackgroundAAComposition, aa2idx
from prokit.folding.trRosetta import trRosetta 

def main():
    # parameters
    try:
        config, config_dict = get_args_setup()
        config = process_exp_config(config)
        os.environ["debug"] = str(config.debug)
        if config.debug == "True":
            config.optimize_schedule = [0.1, 10, 2.0, 5000]
    except:
        xprint("missing or invalid arguments")
        xprint(traceback.format_exc())
        exit(0)

    # Create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.output_dir, config.temp_dir])
    count = len(glob.glob(f"{config.output_dir}/trajectory_*.csv"))
    output_csv = os.path.join(config.output_dir, f"trajectory_{count}.csv")
    output_fasta = os.path.join(config.output_dir, f"final_seq_{count}.fasta")
    output_npz = os.path.join(config.output_dir, f"final_seq_{count}.npz")
    output_model = os.path.join(config.output_dir, f"final_seq_{count}_model.pdb")
    
    # add output file into config parameters
    config.temp_map_file = os.path.join(config.temp_dir, "query.map")
    config.seq_fasta_path = output_fasta
    config.contact_path =  output_npz
    config.pred_model_path =  output_model
    
    xprint("# Current parameter")
    xprint(json.dumps(config, indent= 4))
    xprint("#" * 50)
    config.len_seq = 121

    # Set gpu
    xprint("# GPU settings...")
    gpuConfig = tf.compat.v1.ConfigProto(
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    )

    # Step 1: get background from a model
    xprint("# Step 1. Calculating background model...")
    backgroundMaps = getBackground(config, gpuConfig)
    dprint(backgroundMaps)
    xprint("Done!\n")



    # Step 2: generate a random sequence with valid amino acids
    xprint("# Step 2. Generating a random sequence...")
    randSeq, randIdxArray, validAAs  = initializeRandomSequence(config)
    xprint("Done!\n")

    

    # Step 3. run opimization for this random sequence
    xprint("# Step 3. Run optimization...")
    config.valid_aminoacids = validAAs
    config.background_dist_dict = backgroundMaps
    config.npz_list = ['6H5H_novel_JCIM.npz']
    config.weight_list = [1,1,0.5,1,0.5,1]
    trajectory, finalSeq = run_optimizer_with_trained_model(config, gpuConfig, randIdxArray)
    
    if output_csv != "":
        df = pd.DataFrame(trajectory, columns = ['step', 'sequence', 'score'])
        df.to_csv(output_csv, index = None)
        xprint("The trajectory file is saved to", output_csv)

    if output_fasta != "":
        with open(output_fasta,'w') as f:
            f.write(">final_seq\n%s\n"%(finalSeq))
        xprint("The fasta file is saved to", output_fasta)
    xprint("Done!\n")
    

    # Step 4. Run contact prediction for the final sequence
    xprint("# Step 4. Predicting contancts...")
    contactMaps = run_contact_prediction(config, gpuConfig)
    if output_npz !="":
        np.savez_compressed(output_npz, dist=contactMaps['pd'], omega=contactMaps['po'], theta=contactMaps['pt'], phi=contactMaps['pp'])
        xprint("The npz file is saved to", output_npz)
    xprint("Done!\n")


    
    # Step 5. Run folding for the final sequence and predicted maps
    xprint("# Step 5. Run folding...")
    run_folding_with_pyrosetta(config, config_dict)
    xprint("The predicted model file is saved to", output_model)
    xprint("Finished!\n")



@timeit
def getBackground(inputConfig, gpuConfig):
    # create tensorflow session
    tf.reset_default_graph()
    with tf.compat.v1.Session(config = gpuConfig) as sess:
        data = inputConfig.len_seq
        model = BackgroundModel(inputConfig)
        logger = Logger(sess, inputConfig)
        trainer = BackgroundTrainer(sess, model, data, inputConfig, logger)
        backgroundMaps = trainer.predict_ensemble()
        for key in backgroundMaps.keys():
            print(key, ":" , backgroundMaps[key].shape)

    return backgroundMaps

def initializeRandomSequence(config):
    # generate one random sequence 
    rand_seq, rand_idx_list, aa_valid = generateOneRandomSequence(config.len_seq, removeAA=config.rm_aa)
    print(f"Initialize a random sequence: {rand_seq} -> {len(rand_seq)}")
    return rand_seq, rand_idx_list, aa_valid

@timeit
def run_optimizer_with_trained_model(inputConfig, gpuConfig, seqIdxArrayData):
    tf.reset_default_graph()
    with tf.compat.v1.Session(config = gpuConfig) as sess:
        model = TrRosettaModelKL(inputConfig)

        # create tensorboard logger
        logger = Logger(sess, inputConfig)
        # create optimizer and pass all the previous components to it
        optimizer = MCMCOptimizer(sess, model, inputConfig, logger)
        trajectory, finalSeq = optimizer.optimize_epoch(seqIdxArrayData)
        sess.close()

    return trajectory, finalSeq

@timeit
def run_contact_prediction(inputConfig, gpuConfig):
    # create tensorflow session
    tf.reset_default_graph()
    with tf.compat.v1.Session(config = gpuConfig) as sess:
        a3mData = parse_a3m(inputConfig.seq_fasta_path)
        model = TrRosettaModelPred(inputConfig)
        logger = Logger(sess, inputConfig)
        trainer = TrRosettaTrainer(sess, model, a3mData, inputConfig, logger)

        contactMaps = trainer.predict_ensemble(a3mData)
        dprint(contactMaps['pd'])
        sess.close()
    return contactMaps


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
    

if __name__ == '__main__':
    main()
