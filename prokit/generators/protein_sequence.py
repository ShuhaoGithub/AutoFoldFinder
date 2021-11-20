import numpy as np
import os
import logging


# characters to integers
def aa2idx(seq):
    # convert letters into numbers
    abc = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    idx = np.array(list(seq), dtype='|S1').view(np.uint8)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > 20] = 20
    return idx

# integers back to sequence:
def idx2aa(idx):
    abc=np.array(list("ARNDCQEGHILKMFPSTWYV"))
    return("".join(list(abc[idx])))


# generate one random sequence
def generateOneRandomSequence(lenSeq = 50, removeAA = "CW"):
    aa_valid = np.arange(20)
    aa_skip = aa2idx(removeAA)
    aa_valid = np.setdiff1d(aa_valid, aa_skip)
    rand_idx = np.random.choice(aa_valid, lenSeq)
    rand_seq = idx2aa(rand_idx)
#     rand_seq = 'VDNAYPFMNLYITLEINIDDSPKMNYYNEYRTDFRMHYQLYDGTHRGKVR'
#     rand_idx = aa2idx(rand_seq)
    return rand_seq, rand_idx.reshape(1, -1), aa_valid


def batchGenerateFasta(numSeq, lenSeq, outputPrefix, removeAA="CW"):
    """Batch generate fasta files for random sequences
    """
    basename= os.path.basename(outputPrefix)
    for i in range(numSeq):
        with open(f"{outputPrefix}_{i}.fasta", "w") as f:
            f.write(f">{basename}_{i}\n")
            f.write(generateOneRandomSequence(lenSeq, removeAA=removeAA)[0] + "\n")
        logging.debug(f"Success to create the file: {outputPrefix}_{i}.fasta")


def getBackgroundAAComposition():
    return np.array([0.07892653, 0.04979037, 0.0451488 , 0.0603382 , 0.01261332,
                    0.03783883, 0.06592534, 0.07122109, 0.02324815, 0.05647807,
                    0.09311339, 0.05980368, 0.02072943, 0.04145316, 0.04631926,
                    0.06123779, 0.0547427 , 0.01489194, 0.03705282, 0.0691271])


# mutate one protein sequence with random points and its random valid characters
def mutateProteinSequence(seqArray, aaValidArray, mutNum =1):
    seqArray = seqArray.reshape(1, -1)
    randMutIdx = np.random.choice(np.arange(seqArray.shape[1]), mutNum)
    newSeqArray = np.copy(seqArray)
    newSeqArray[0,randMutIdx] = np.random.choice(aaValidArray, mutNum)
    return newSeqArray
