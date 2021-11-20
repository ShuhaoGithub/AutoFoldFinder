import os, sys, glob
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
from prokit.utils.logger import Logger
from prokit.utils.args import get_args_setup
from prokit.utils.protein_utils import *
from prokit.generators.protein_sequence import generateOneRandomSequence, getBackgroundAAComposition, aa2idx

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        config = get_args_setup()
        config = process_exp_config(config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.output_dir])
    count = len(glob.glob(f"{config.output_dir}/trajectory_*.csv"))
    output_csv = os.path.join(config.output_dir, f"trajectory_{count}.csv")
    output_fasta = os.path.join(config.output_dir, f"final_seq_{count}.fasta")
    output_npz = os.path.join(config.output_dir, f"final_seq_{count}.npz")

    # 
    gpuConfig = tf.compat.v1.ConfigProto(
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    )
    # get background
    backgroundMaps = getBackground(config, gpuConfig)

    # generate one random sequence 
    randSeq, randIdxArray, validAAs = generateOneRandomSequence(config.len_seq, removeAA=config.rm_aa)
    print("The initial sequence:", randSeq, "->", len(randSeq))

    # test
    # randIdxArray= np.array(aa2idx("PFGGQDMRIFISTMIVAMPEVPIKNVRMKGIKHYFAHGIFFFIIGMNRIH")).reshape(1, -1)


    config.valid_aminoacids = validAAs
    config.background_dist_dict = backgroundMaps
    trajectory, finalSeq = run_optimizer_with_trained_model(config, gpuConfig, randIdxArray)
       
    # 3. save results
    if output_csv != "":
        df = pd.DataFrame(trajectory, columns = ['step', 'sequence', 'score'])
        df.to_csv(output_csv, index = None)
        print("The trajectory file is saved to", output_csv)


    # 4. save sequence fasta and predict npz
    if output_fasta != "":
        with open(output_fasta,'w') as f:
            f.write(">final_seq\n%s\n"%(finalSeq))
        print("The fasta file is saved to", output_fasta)
        config.seq_fasta_path = output_fasta


        contactMaps = run_prediction(config, gpuConfig)
        # save distograms & anglegrams
        if output_npz !="":
            np.savez_compressed(output_npz, dist=contactMaps['pd'], omega=contactMaps['po'], theta=contactMaps['pt'], phi=contactMaps['pp'])
            print("The npz file is saved to", output_npz)


def getBackground(inputConfig, gpuConfig):
    # create tensorflow session
    with tf.compat.v1.Session(config = gpuConfig) as sess:
        # create your data generator
        data = inputConfig.len_seq
        
        # create an instance of the model you want
        model = BackgroundModel(inputConfig)

        # create tensorboard logger
        logger = Logger(sess, inputConfig)
        # create trainer and pass all the previous components to it
        trainer = BackgroundTrainer(sess, model, data, inputConfig, logger)

        # here you train your model
        backgroundMaps = trainer.predict_ensemble()

        for key in backgroundMaps.keys():
            print(key, ":" , backgroundMaps[key].shape)
    

    return backgroundMaps


def initRandomSequenceArray(config):
    # generate one random sequence 
    rand_seq, rand_idx_list, aa_valid = generateOneRandomSequence(config.len_seq, removeAA=config.rm_aa)
    print(f"Initialize a random sequence: {rand_seq} -> {len(rand_seq)}")

    return np.array(rand_idx_list).reshape(1, -1)


def run_optimizer_with_trained_model(inputConfig, gpuConfig, seqIdxArrayData):
    
    with tf.compat.v1.Session(config = gpuConfig) as sess:
        model = TrRosettaModelKL(inputConfig)

        # create tensorboard logger
        logger = Logger(sess, inputConfig)
        
        # create optimizer and pass all the previous components to it
        optimizer = MCMCOptimizer(sess, model, inputConfig, logger)
        trajectory, finalSeq = optimizer.optimize_epoch(seqIdxArrayData)

    return trajectory, finalSeq

def run_prediction(inputConfig, gpuConfig):
    # create tensorflow session
    with tf.compat.v1.Session(config = gpuConfig) as sess:
        # create your data generator

        a3mData = parse_a3m(inputConfig.seq_fasta_path)
        # create an instance of the model you want
        model = TrRosettaModelPred(inputConfig)

        # create tensorboard logger
        logger = Logger(sess, inputConfig)
        # create trainer and pass all the previous components to it
        trainer = TrRosettaTrainer(sess, model, a3mData, inputConfig, logger)

        # here you train your model
        contactMaps = trainer.predict_ensemble(a3mData)

    return contactMaps

if __name__ == '__main__':
    main()
