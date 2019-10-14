#!/usr/bin/python
#

# python imports
from __future__ import print_function
import os
import glob
import sys
from scipy.misc import imread, imsave
import numpy as np
from numpngw import write_png

from json2labelImg import json2labelImg
from json2instanceImg import json2instanceImg


from tqdm import tqdm
from cityscape_panoptic_gt import panoptic_converter
from argparse import ArgumentParser
import os

args = None


def process_folder(fn):
    global args

    dst = fn.replace("_polygons.json", "_label{}s.png".format(args.id_type))

    # do the conversion
    try:
        json2labelImg(fn, dst, args.id_type)
    except:
        tqdm.write("Failed to convert: {}".format(fn))
        raise

    if args.instance:
        dst = fn.replace("_polygons.json",
                         "_instance{}s.png".format(args.id_type))

        # do the conversion
        # try:
        json2instanceImg(fn, dst, args.id_type)
        # except:
        #     tqdm.write("Failed to convert: {}".format(f))
        #     raise

    if args.color:
        # create the output filename
        dst = fn.replace("_polygons.json", "_labelColors.png")

        # do the conversion
        try:
            json2labelImg(fn, dst, 'color')
        except:
            tqdm.write("Failed to convert: {}".format(f))
            raise

    # if args.panoptic and args.instance:
        # panoptic_converter(f, out_folder, out_file)


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--datadir', default="")
    parser.add_argument('--id-type', default='level3Id')
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--instance', type=bool, default=False)
    parser.add_argument('--panoptic', type=bool, default=False)
    parser.add_argument('--num-workers', type=int, default=10)

    args = parser.parse_args()

    return args

# The main method


def main(args):
    if args.panoptic:
        args.instance = True
        for split in ['train', 'val']:
            folder_name = os.path.join(args.datadir, 'gtFine')
            output_folder = os.path.join(folder_name, split + "_panoptic")
            os.makedirs(output_folder, exist_ok=True)
            out_file = os.path.join(folder_name, split + "_panoptic.json")
            panoptic_converter(args.num_workers, os.path.join(
                folder_name, split), output_folder, out_file)
    else:
        sys.path.append(os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', 'helpers')))
        # how to search for all ground truth
        searchFine = os.path.join(args.datadir, "gtFine",
                                  "*", "*", "*_gt*_polygons.json")

        # search files
        filesFine = glob.glob(searchFine)
        filesFine.sort()

        files = filesFine

        if not files:
            tqdm.writeError(
                "Did not find any files. Please consult the README.")

        # a bit verbose
        tqdm.write("Processing {} annotation files".format(len(files)))

        # iterate through files
        progress = 0
        tqdm.write("Progress: {:>3} %".format(
            progress * 100 / len(files)), end=' ')

        from multiprocessing import Pool

        pool = Pool(args.num_workers)
        # results = pool.map(process_pred_gt_pair, pairs)
        results = list(tqdm(pool.imap(process_folder, files), total=len(files)))
        pool.close()
        pool.join()




if __name__ == "__main__":
    args = get_args()
    main(args)
