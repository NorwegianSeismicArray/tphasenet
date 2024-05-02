#DEFINE THESE
USER=$(whoami)
PROJECTNAME=phasedetection
# the script that starts the training of (different) models
SCRIPT=train.py
HELPER_SCRIPT=train_utils.py
MODELS=models.py
REQUIREMENTS=requirements.txt
# directory with input (tf/data/)  and output data
#LOCATION=./tf #/waveform_data
# Hard-code temporary
# output for models
OUTPUT=tf
# working directory
WORK_DIR=`pwd`
# another working directory for faster training in case you are on a network disk (local disk, gpu, ...)
# Modify script if WORK_DIR and BASE_DIR are the same
BASE_DIR=/nobackup/$USER/$PROJECTNAME
CONFIG=configs/transphasenet_large.ini

test -d $OUTPUT || mkdir $OUTPUT
test -d $OUTPUT/output || mkdir $OUTPUT/output
test -d $OUTPUT/output/models || mkdir $OUTPUT/output/models
test -d $BASE_DIR || mkdir $BASE_DIR
test -d $BASE_DIR/output || mkdir $BASE_DIR/output
test -d $BASE_DIR/data || mkdir $BASE_DIR/data
test -d $BASE_DIR/logs || mkdir $BASE_DIR/logs
test -d $BASE_DIR/output/models || mkdir $BASE_DIR/output/models

# Fix where data is located
#rsync -ahr --progress $LOCATION/* $BASE_DIR/

cp $SCRIPT $HELPER_SCRIPT $MODELS $BASE_DIR/
cp $REQUIREMENTS $BASE_DIR/req.txt
cp configs/global.ini $BASE_DIR/tf/global_config.ini
cp $CONFIG $BASE_DIR/tf/model_config.ini
cd $BASE_DIR
bash -c "pip install -r req.txt
         python train.py"
cd $WORK_DIR

#CLEAN UP
rsync -ahr --progress $BASE_DIR/outputs/*.png $OUTPUT/plots
rsync -ahr --progress $BASE_DIR/outputs/*.npz $OUTPUT/predictions
rsync -ahr --progress $BASE_DIR/outputs/*.tf $OUTPUT/models
