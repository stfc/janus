from cc_hdnnp.structure import AllStructures, Species, Structure
from cc_hdnnp.dataset import Dataset
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate distances between structures.')
    parser.add_argument(
        'str_id',
         metavar='N',
         type=int,
         nargs=1,
         default=0,
         help='Index of structure being compared in first file',
    )
    parser.add_argument(
        'files_in',
         type=str,
         nargs=2,
         help='Files containing structures being compared',
    )
    parser.add_argument(
        'file_out',
         type=str,
         nargs='?',
         help='Filepath to write distances to',
    )
    parser.add_argument(
        '--permute',
        '-p',
         type=bool,
         nargs='?',
         help='minimise the distance by permuting same elements',
    )
    parser.add_argument(
        '--verbose',
        '-v',
         type=bool,
         nargs='?',
         help='minimise the distance by permuting same elements',
    )
    args = parser.parse_args()
    i = args.str_id[0]
    file_1 = args.files_in[0]
    file_2 = args.files_in[1]
    
    if args.file_out:
        file_out = args.file_out
    else:
        file_out = None
    
    if args.permute:
        permute = args.permute
    else:
        permute = None
    
    if args.verbose:
        verbose = args.verbose
    else:
        verbose = None

    dataset_1 = Dataset(data_file=file_1)
    dataset_2 = Dataset(data_file=file_2)
    dataset_1.compare_structure(
        dataset_2[i],
        file_out=file_out,
        permute=permute,
        verbose=verbose
    )
