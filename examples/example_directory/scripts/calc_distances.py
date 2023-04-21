from cc_hdnnp.dataset import Dataset
from mpi4py import MPI
import sys
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate distances between structures.')
    parser.add_argument(
        'idx_2',
         metavar='N',
         type=int,
         nargs=1,
         default=0,
         help='Index of structure being compared in second dataset',
    )
    parser.add_argument(
        'files_in',
         type=str,
         nargs=2,
         help='Files containing structures being compared',
    )
    parser.add_argument(
        'formats',
         type=str,
         nargs=2,
         help='Formats of files containing structures being compared',
    )
    parser.add_argument(
        'idx_1',
         type=int,
         nargs=2,
         help='Indicies of structures being compared in first dataset',
    )
    parser.add_argument(
        'file_out',
         type=str,
         nargs='?',
         help='Filepath to write distances to',
    )
    parser.add_argument(
        'delimiter',
         type=str,
         nargs='?',
         help='Delimiter between values saved',
    )
    parser.add_argument(
        '--permute',
        '-p',
         type=int,
         nargs='?',
         help='Minimise the distance by permuting same elements',
    )
    args = parser.parse_args()
    idx_2 = args.idx_2[0]
    file_1 = args.files_in[0]
    file_2 = args.files_in[1]
    idx_1 = [args.idx_1[0], args.idx_1[1]]

    if args.formats:
        format_1 = args.formats[0]
        format_2 = args.formats[1]
    else:
        format_1 = "n2p2"
        format_2 = "n2p2"

    if args.file_out:
        file_out = args.file_out
    else:
        file_out = '../distances.csv'
    file_out += f".{idx_2}"

    if args.delimiter:
        delimiter = args.delimiter
    else:
        delimiter = ','

    if args.permute is not None:
        permute = bool(args.permute)
    else:
        permute = None

    params = {
        "file_out": file_out,
        "permute": permute,
        "delimiter": delimiter,
    }

    params = {key:value for key, value in params.items() if value is not None}

    dataset_1 = Dataset(
        data_file=file_1,
        format=format_1,
    )
    dataset_2 = Dataset(
        data_file=file_2,
        format=format_2,
    )

    try:
        dataset_1.compare_structure(
            frame=dataset_2[idx_2],
            idx_range=idx_1,
            append=False,
            use_mpi=True,
            **params,
        )
    except Exception as e:
        print(f"\n{type(e).__name__}: {e}\n")
        sys.stdout.flush()
        MPI.COMM_WORLD.Abort(1)

