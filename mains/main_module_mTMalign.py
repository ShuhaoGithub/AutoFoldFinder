
import os, glob, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

from prokit.alignment.mtmalignWrapper import post_server, fetch_results_from_txt
from prokit.utils.args import get_server_args
from prokit.utils.logger import *
import traceback
import pandas as pd


def main(args):
    if args["mode"]=="post":
        post_mtmalign_server(args["exp_num"], start=args["exp_start"])
    if args["mode"] == "get":
        get_mtmalign_server(args["task_file"])
        
def post_mtmalign_server(expNum, start=0):
    pdbFileList = glob.glob(f"../data/experiments/{expNum}/output/final_seq_*_model.pdb")
    outTaskFile = f"../data/experiments/{expNum}/{expNum}-tasksubmit-{start}.txt"
    with open(outTaskFile, "w") as f:
        for pdbModelID in range(start, len(pdbFileList)):
            pdbFile = f"../data/experiments/{expNum}/output/final_seq_{pdbModelID}_model.pdb"
            taskName = expNum+"-"+os.path.basename(pdbFile).rsplit(".", 1)[0]
            taskName, taskID = post_server(pdbFile, taskName)
            f.write(f"{taskName} {taskID}\n")

def get_mtmalign_server(taskFile):
    'TMscore', 'seq_id',  'RMSD', 'aligned_len', 'pdb_link', 'Querry_seq', 'answer_seq'
    curList = fetch_results_from_txt(taskFile)
    # print(curList)
    csvFile = taskFile.rsplit(".", 1)[0]+"_summary.csv"
    colHeaders =['modelName', 'tmScore', 'seqIdentity', 'rmsd', 'aligned_len','pdbID', 'querySeq', 'answerSeq']
    dfData = pd.DataFrame(curList, columns=colHeaders)
    dfData.to_csv(csvFile, columns = colHeaders, index=None)

if __name__ == "__main__":
    server_args = get_server_args()
    main(server_args)