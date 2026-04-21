import torch
import numpy as np

def sequence_to_gcn_tensor(sequence):
    """
    sequence shape
    (T , 105 , 2)
    output
    (1 , 2 , T , 105)
    """
    seq = np.array(sequence)
    seq = seq.transpose(2,0,1)
    tensor = torch.tensor(seq).unsqueeze(0).float()
    return tensor