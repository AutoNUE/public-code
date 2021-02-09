#! /bin/bash

bdd_dd='bdd100k/seg/'
mapillary_dd='mapillary-vistas-dataset_public_v1.1/'
gta_dd='gta/'

sd='source_datasets_dir/'

python core/mapillary.py ${mapillary_dd} ${sd}
python core/gta.py ${gta_dd} ${sd}
python core/bdds.py ${bdd_dd} ${sd}
