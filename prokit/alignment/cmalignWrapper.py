import os, glob, sys
from prokit.utils.logger import *
from prokit.utils.protein_utils import distance_bin_to_map
from prokit.configs.tool_path import CMALIGNTOOL
import numpy as np
from joblib import Parallel, delayed

def mock_dist_bin():
    
    npz = "../data/experiments/exp1/output/final_seq_1.npz"
    data=np.load(npz)
    pd = data['dist']
    myprint(pd.shape)
    return pd

def _get_map_len(map_file):
    with open(map_file, "r") as f:
        first_line = f.readline().split()
    return int(first_line[1])

def cmalign_wrapper_vs_one(query_map_file, ref_map_file):
    """ a wrapper for CMAlign to generate a aligned map and its score
        between query map and ref map
    """
    assert query_map_file.endswith(".map")
    query_map_basename = os.path.basename(query_map_file).rsplit(".", 1)[0]
    ref_map_basename = os.path.basename(ref_map_file).rsplit(".", 1)[0]
    result_file = query_map_file.rsplit("/", 1)[0]+\
                f"/{query_map_basename}.{ref_map_basename}.mapalign.out"
#     result_file = query_map_file.replace(".map", f"_{query_map_basename}_{ref_map_basename}_mapalign.out")
    len_mapA = _get_map_len(query_map_file)
    len_mapB = _get_map_len(ref_map_file)
    myprint(len_mapA, len_mapB)
    cmd_str=f"{CMALIGNTOOL} {query_map_file} {ref_map_file} {result_file} {len_mapA} {len_mapB}"
    myprint(cmd_str)
    cscore = os.popen(cmd_str).read().strip().split()[1]
    myprint(cscore)
    return float(cscore)

def batch_cmalign_wrapper(query_map_file, ref_map_dir):
    ref_map_file_list = glob.glob(f"{ref_map_dir}/*.map")  # for test
    cm_score_list= [] 
    for ref_map in ref_map_file_list:
        cm_score = cmalign_wrapper_vs_one(query_map_file, ref_map)
        cm_score_list.append(cm_score)
    return cm_score_list

def contact_mat_to_map_txt_line(contact_mat):
    contact_index_table = np.array(np.where(np.triu(contact_mat, 3) > 0)).reshape(2, -1).T
    myprint(contact_index_table)
    ones_matrix = contact_mat[contact_index_table[:,0], contact_index_table[:,1]].reshape(-1, 1)
    map_txt_line = np.concatenate([contact_index_table, ones_matrix], axis=1)
    myprint(map_txt_line)
    return map_txt_line, contact_mat.shape[0]

def map_txt_line_to_txtfile(nparray, len_index, outputfile=None):
    assert outputfile is not None
    with open(outputfile, "w") as f:
        f.write(f"LEN\t{len_index}\n")
        for row in nparray:
            f.write(f"CON\t{int(row[0])}\t{int(row[1])}\t{row[2]:.1f}\n")

    with open(outputfile+".txt", "w") as f:
        f.write(f"{len_index}\n")
        for row in nparray:
            f.write(f"{int(row[0])} {int(row[1])} {row[2]:.6f}\n")

def cmalign_score(query_dist_bin, ref_map_txt, temp_map_file):
    dist_mat, cont_mat = distance_bin_to_map(query_dist_bin)
    myprint(cont_mat[:5, :5])
    map_txt_line, num_residue= contact_mat_to_map_txt_line(cont_mat)
    map_txt_line_to_txtfile(map_txt_line, num_residue, outputfile=temp_map_file)
    cmalign_score = cmalign_wrapper_vs_one(temp_map_file, ref_map_txt)
    return cmalign_score
    

def batch_cmalign_score(query_dist_bin, ref_map_dir, temp_map_file):
    dist_mat, cont_mat = distance_bin_to_map(query_dist_bin)
    myprint(cont_mat[:5, :5])
    map_txt_line, num_residue= contact_mat_to_map_txt_line(cont_mat)
    map_txt_line_to_txtfile(map_txt_line, num_residue, outputfile=temp_map_file)
    cmalign_score_list = batch_cmalign_wrapper(temp_map_file, ref_map_dir)
    return cmalign_score_list

@timeit
def parallel_batch_cmalign_score(query_dist_bin, ref_map_dir, temp_map_file):
    dist_mat, cont_mat = distance_bin_to_map(query_dist_bin)
    myprint(cont_mat[:5, :5])
    map_txt_line, num_residue= contact_mat_to_map_txt_line(cont_mat)
    map_txt_line_to_txtfile(map_txt_line, num_residue, outputfile=temp_map_file)
    
    cmalign_score_list = Parallel(n_jobs= 10)(\
                                delayed(cmalign_wrapper_vs_one)\
                                (temp_map_file, os.path.join(ref_map_dir, str(i)+".map")) for i in range(1, 1233)\
                                )
    return cmalign_score_list

def parallel_batch_cmalign_score_top(query_dist_bin, top_indices, ref_map_dir, temp_map_file):
    dist_mat, cont_mat = distance_bin_to_map(query_dist_bin)
    myprint(cont_mat[:5, :5])
    map_txt_line, num_residue= contact_mat_to_map_txt_line(cont_mat)
    map_txt_line_to_txtfile(map_txt_line, num_residue, outputfile=temp_map_file)
    
    cmalign_score_list = Parallel(n_jobs= 10)(\
                                delayed(cmalign_wrapper_vs_one)\
                                (temp_map_file, os.path.join(ref_map_dir, str(i)+".map")) for i in list(top_indices)\
                                )
    return cmalign_score_list