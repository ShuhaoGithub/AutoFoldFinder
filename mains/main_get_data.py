
import os, glob, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

from prokit.utils.logger import *
from prokit.utils.args import get_database_args
# from prokit.utils.parallel import mapper
from multiprocessing import Pool
import requests


def main(args):
    parallel_fetch_pdbs_with_range(args)
    xprint("Task is completed.")

def parallel_fetch_pdbs_with_range(args):
    lowbound, upbound =  args["seq_range"]
    dictList =[]
    seqDict = _load_fasta_file(args)
    for key in seqDict.keys():
        curLen = len(seqDict[key])
        if (curLen >= lowbound) and  (curLen <= upbound):
            dictList.append({"scopid": key, "pdbdir": args["out_pdb_dir"]})
    # print(dictList)
    # exit(0)
    xprint(f"Current number of seqs: {len(dictList)}")
    pool = Pool(args["core_num"])
    res = pool.map(_wget_one_pdb, dictList)
    pool.close()
    pool.join()

    return 

def _wget_one_pdb(kwargs):
    scopID, pdbDir = kwargs["scopid"], kwargs["pdbdir"]
    url = f"https://scop.berkeley.edu/astral/pdbstyle/ver=2.07&id={scopID}&output=txt"
    if (os.path.getsize(f"{pdbDir}/{scopID}.pdb") == 0) or (not os.path.exists(f"{pdbDir}/{scopID}.pdb")):
        cmdStr = f"curl --location --request GET '{url}' > {pdbDir}/{scopID}.pdb"
        xprint(cmdStr)
        os.system(cmdStr)


def _load_fasta_file(args):
    seqDict={}
    with open(args["seq_file"], "r") as f:
        for line in f:
            row = line.strip()
            if row[:1]== ">":
                curkey = row[1:8]
                seqDict[curkey] = ""
            else:
                seqDict[curkey] += row
    return seqDict
        
    

if __name__ == '__main__':
    args = get_database_args()
    main(args)



