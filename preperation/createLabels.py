#!/usr/bin/python
#

# python imports
from __future__ import print_function
import os, glob, sys


from json2labelImg import json2labelImg
from json2instanceImg import json2instanceImg

from tqdm import tqdm
from argparse import ArgumentParser
import os

args = None

def process_folder(f):
    global args

    dst = f.replace( "_polygons.json" , f"_label{args.id_type}s.png" )

    # do the conversion
    try:
        json2labelImg( f , dst , args.id_type )
    except:
        tqdm.write("Failed to convert: {}".format(f))
        raise

    if args.instance:
        dst = f.replace( "_polygons.json" , f"_instance{args.id_type}s.png" )

        # do the conversion
        # try:
        json2instanceImg( f , dst , args.id_type )
        # except:
        #     tqdm.write("Failed to convert: {}".format(f))
        #     raise

    if args.color:
        # create the output filename
        dst = f.replace( "_polygons.json" , f"_labelColors.png" )

        # do the conversion
        try:
            json2labelImg( f , dst , 'color' )
        except:
            tqdm.write("Failed to convert: {}".format(f))
            raise


def get_args():
    parser = ArgumentParser()

    # parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default="")
    parser.add_argument('--id-type', default='id')
    parser.add_argument('--color', type=bool, default=False) #train, instance, color, all
    parser.add_argument('--instance', type=bool, default=False) #train, instance, color, all
    parser.add_argument('--num-workers', type=int, default=10) #train, instance, color, all

    
    args = parser.parse_args()

    return args

# The main method
def main(args):
    
    # how to search for all ground truth
    searchFine   = os.path.join( args.datadir , "gtFine"   , "*" , "*" , "*_gt*_polygons.json" )

    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()

    
    files = filesFine 
    
    if not files:
        tqdm.writeError( "Did not find any files. Please consult the README." )

    # a bit verbose
    tqdm.write("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    tqdm.write("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')





    from multiprocessing import Pool
    import time


    pool = Pool(args.num_workers)
    # results = pool.map(process_pred_gt_pair, pairs)
    results = list(tqdm(pool.imap(process_folder, files), total=len(files)))
    pool.close()
    pool.join()



if __name__ == "__main__":
    args = get_args()
    main(args)
