import argparse
import os
import tensorflow as tf
from network_EGAN import GAN
from utils import check_folder


def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow  bone suppression training')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--gpu_ids', type=str, default='4', help='gpu ids for GPU')
    parser.add_argument('--batch_sz', type=int, default=4, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=151, help='epoch number')
    parser.add_argument('--val_path', type=str, default='./data/val', help='path to val data')
    parser.add_argument('--train_path', type=str, default='./data/train', help='path to the training data')
    parser.add_argument('--test_path', type=str, default='./data/test', help='path to the test data')
    parser.add_argument('--checkpoints_dir', type=str, default='./ckpt', help='save the checkpoint files')
    parser.add_argument('--summary_dir', type=str, default='./summary', help='summary path')
    parser.add_argument('--test_result_dir', type=str, default='./testResult', help='save the test results')
    parser.add_argument('--crop_h', type=int, default=1024, help='summary path')
    parser.add_argument('--crop_w', type=int, default=1024, help='summary path')
    parser.add_argument('--edge_dir', type=str, default='./data/edgefile/', help='path to the edge data')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args is None:
        exit()    
    check_folder(args.checkpoints_dir)
    check_folder(args.summary_dir)
    model_params = vars(args)    
    for k, v in model_params.items():
        print("\t%s:%s"%(k,v))

    # open session
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    with tf.Session() as sess:
        gan = GAN(sess, args)
        # build graph
        gan.build_model()

        gan.train()


if __name__=='__main__':

    main()

