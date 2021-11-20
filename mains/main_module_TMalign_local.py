import os, glob, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

from prokit.alignment.tmalignWrapper import compare_one_against_all
from prokit.utils.args import get_local_tmalign_args
from prokit.utils.logger import *
import traceback, io
import pandas as pd
from multiprocessing import Pool


def main(args):
    startNum = int(args["exp_start"])

    if args["core_num"]== 1 :
        jsonList = calc_all_models_to_scop(args["exp_num"], args["scop_dir"], start=startNum)
    else:
        jsonList = parallel_calc_all_models_to_scop(args["exp_num"], args["scop_dir"], coreNum=args["core_num"], start=startNum)

    endNum = len(jsonList) + startNum
    expNum = args["exp_num"]
    args["out_csv_path"] = f"../data/experiments/{expNum}/{expNum}-summary-{startNum}_{endNum}.csv" 
    save_to_csv(jsonList, args["out_csv_path"])
    
    
def save_to_csv(jsonList, csvFile):
    dfResult = pd.DataFrame.from_dict(jsonList)
    keyList = ["queryPath", "refPath", "tmScore", "seqIdentity", "rmsd"]
    dfResult.to_csv(csvFile, columns= keyList, index=False)
    xprint(f"The file is saved to {csvFile}")


def calc_all_models_to_scop(expNum, refPDBDir, start=0):
    pdbFileList = glob.glob(f"../data/experiments/{expNum}/output/final_seq_*_model.pdb")
    jsonList = []
    for pdbModelID in range(start, len(pdbFileList)):
        queryFile = f"../data/experiments/{expNum}/output/final_seq_{pdbModelID}_model.pdb"
        jsonList.append(calc_one_model_to_scop((queryFile, refPDBDir)))
    return jsonList


def parallel_calc_all_models_to_scop(expNum, refPDBDir, coreNum=1 ,start =0):
    pdbFileList = glob.glob(f"../data/experiments/{expNum}/output/final_seq_*_model.pdb")
    argList = []
    for pdbModelID in range(start, len(pdbFileList)):
        queryFile = f"../data/experiments/{expNum}/output/final_seq_{pdbModelID}_model.pdb"
        argList.append([queryFile, refPDBDir])
    
    with Pool(coreNum) as pool:
        jsonList = pool.map(calc_one_model_to_scop, argList)
    
    return jsonList

def calc_one_model_to_scop(item):
    queryPDBFile, refPDBDir = item
    curJson = compare_one_against_all(queryPDBFile, refPDBDir)
    
    return curJson



if __name__ == "__main__":
    args = get_local_tmalign_args()
    main(args)