from utils import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
from torch.autograd import Variable
import timeit


# This is for pytorch 0.4
# For version 0.3, change .item() => .data[0]

# In particular:
#  0.3  class_ = verify(model, fmatrix, cfg['in_frames']).data[0]
# >0.4  class_ = verify(model, fmatrix, cfg['in_frames']).item()




def verify(model, fmatrix, in_frames):
    # fmatrix of size [T, feat_dim]
    # (1) build frames with context first
    frames = build_frames(fmatrix, in_frames)
    # (2) build input tensor to network
    x = Variable(torch.FloatTensor(np.array(frames)))
    # (3) Infer the prediction through network
    y_ = model(x)
    # (4) Sum up the logprobs to obtain final decission    
    pred = y_.sum(dim=0)    
    class_ = pred.max(dim=0)[1]   
    
    return class_

def main(opts):
    with open(opts.train_cfg, 'r') as cfg_f:
        cfg = json.load(cfg_f)
    with open(cfg['spk2idx'], 'r') as spk2idx_f:
        spk2idx = json.load(spk2idx_f)
        
        idx2spk = dict((v, k) for k, v in spk2idx.items())
    # Feed Forward Neural Network
    model = nn.Sequential(nn.Linear(cfg['input_dim'] * cfg['in_frames'],
                                    cfg['hsize']),
                          nn.ReLU(),
                          nn.Linear(cfg['hsize'], cfg['hsize']),
                          nn.ReLU(),
                          nn.Linear(cfg['hsize'], cfg['hsize']),
                          nn.ReLU(),
                          nn.Linear(cfg['hsize'], cfg['num_spks']),
                          nn.LogSoftmax(dim=1))
    print('Created model:')
    print(model)
    print('-')
    # load weights
    model.load_state_dict(torch.load(opts.weights_ckpt))
    print('Loaded weights')
    out_log = open(opts.log_file, 'w')
    with open(opts.te_list_file, 'r') as test_f:        
        with open(opts.candidates_list, 'r') as cand_f:
            test_list = [l.rstrip() for l in test_f]                     
            timings = []
            beg_t = timeit.default_timer()
            for test_file, candidate in zip(test_list,cand_f):
                
                test_path = os.path.join(opts.db_path, test_file + '.' + opts.ext)            
                fmatrix = read_fmatrix(test_path)
                
                verif_ = verify(model, fmatrix, cfg['in_frames']).item()
                if idx2spk[verif_] == candidate.rstrip():
                    pass_ = "1"                        
                else:
                    pass_ = "0"
                if opts.blind:                    
                    out_log.write('{}\t{}\t{}\n'.format(test_file,candidate.rstrip(),pass_))
                else:
                    out_log.write('{}\t{}\t{}'.format(test_file,idx2spk[verif_],candidate))
                    print('{}\t{}\t{}   '.format(test_file, idx2spk[verif_],candidate))
                
    out_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply trained MLP to classify')

    
    parser.add_argument('--db_path', type=str, default='mcp',
                        help='path to feature files (default: ./mcp)')
    parser.add_argument('--te_list_file', type=str, default='spk_rec.test',
                        help='list with names of files to verify (default. spk_rec.test)')
    parser.add_argument('--candidates_list', type=str, default='spk_rec.test',
                        help='list with names of candidates to verify (default. spk_rec.test)')
    parser.add_argument('--weights_ckpt', type=str, default=None, help='model: ckpt file with weigths')
    parser.add_argument('--log_file', type=str, default='spk_classification.log',
                        help='result file (default: spk_classification.log')
    parser.add_argument('--train_cfg', type=str, default='ckpt/train.opts',
                        help="arguments used for training (default: ckpt/train.opts)")
    parser.add_argument('--ext', type=str, default='mcp',
                        help='Extension of feature files (default mcp)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Print information about required time, input shape, etc.')
    parser.add_argument('--blind', action='store_true', 
                        help='Performs a blind verification')


    opts = parser.parse_args()
    if opts.weights_ckpt is None:
        raise ValueError('Weights ckpt not specified!')
    main(opts)

