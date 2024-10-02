import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('--fusion_type', type=str, default='ecg', help='train or eval for [fusion,cxr,ecg]')
    parser.add_argument('--save_dir', type=str, help='Directory relative which all output files are stored',
                    default='/home/mimic/MIMIC_subset/MIMIC_subset/checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of chunks to train')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    # parser.add_argument('--model', type=str,default='wavevit_s', required=False, choices=['ResNet1d', 'spectrogram', 'ECGModel', 'wavevit_s', 'CXRModels'], help='Specify the model to train')
    parser.add_argument('--ecg_model', type=str, default='spectrogram', help='ECG model name')
    parser.add_argument('--cxr_model', type=str, default='wavevit_s', help='CXR model name')

    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--domain', type=str, default='frequency', choices=['frequency' , 'S_T'])



    
    return parser
