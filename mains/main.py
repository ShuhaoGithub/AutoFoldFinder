
import logging
from prokit.Generator import ProteinSequence
import os

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="parameters for prokit")
    parser.add_argument("--num_seq", type= int, default= 1000, help="TODO")
    parser.add_argument("--len_seq", type= int, default= 50, help="TODO")
    parser.add_argument("--output_prefix", type= str, default= "data/randomSequences_v1/randSeq", help="TODO")

    # TODO: some problems
    parser.add_argument("--background_model", type=str, required=False, default="data/ckpt/background/bkgr2019_05",
                        help="path to background network weights")

    parser.add_argument("--trRosetta_model",  type=str, required=False, default="data/ckpt/model2019_07",
                        help="path to trRosetta network weights")

    parser.add_argument("--rm_aa", type=str, default="CW",
                        help="disable specific amino acids from being sampled (ex: 'C' or 'W,Y,F')")
    parser.add_argument("-d", "--debug", action="store_true", default= False, help="TODO")
    args = parser.parse_args()
    return args


def getBackground(kwargs):
    lenSeq = kwargs["len_seq"]
    modelDir = kwargs["background_model"]

    return 



def runGenerateProSequences(kwargs):
    numSeq = kwargs["num_seq"]
    lenSeq = kwargs["len_seq"]
    outputPrefix = kwargs["output_prefix"]
    rmAA = kwargs["rm_aa"]

    logging.info("Step 1: Generating protein files...")
    os.makedirs(outputPrefix.rsplit("/", 1)[0], exist_ok= True)
    ProteinSequence.batchGenerateFasta(numSeq, lenSeq, outputPrefix,removeAA=rmAA)
    logging.info("Done")

def runMultiSeqAlignment(kwargs):
    outputPrefix = kwargs["output_prefix"]
    numSeq = kwargs["num_seq"]
    return

def runOptimization(kwargs):

    return

if __name__ == '__main__':

    args=get_args()
    logformat = "%(asctime)s %(levelname)s: %(message)s"
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=logformat)
    else:
        logging.basicConfig(level=logging.INFO, format=logformat)


    kwargs = vars(args)
    runGenerateProSequences(kwargs)
    # runMSA(kwargs)
    # runStep2(args)
