# Copyright 2024, Erik Myklebust, Andreas Koehler, MIT license

"""
Code for generating script for training a phase picking models

"""

from omegaconf import OmegaConf
from setup_config import add_root_paths,get_config_dir,dict_to_namespace


print('Reading config ...')
config_dir = get_config_dir()
args = OmegaConf.load(f'{config_dir}/config.yaml')
args_dict = OmegaConf.to_container(args, resolve=True)
args = OmegaConf.create(args_dict)
OmegaConf.set_struct(args, False)
cfg = dict_to_namespace(args)
print('Config read.')

print('Writing script ...')
scriptfile = open('train_on_gpu.sh','w')
line='USER=$(whoami)\n'
scriptfile.write(line)
line=f'PROJECTNAME={cfg.run.project}\n'
scriptfile.write(line)
line=f'DATASET={cfg.data.input_dataset_name}\n'
scriptfile.write(line)
line=f'SCRIPT_LOCATION={cfg.run.code_location}\n'
scriptfile.write(line)
line=f'REQUIREMENTS={cfg.run.code_location}/requirements.txt\n'
scriptfile.write(line)
line=f'CONFIG={cfg.run.code_location}/config.yaml\n'
scriptfile.write(line)
line=f'LOCATION={cfg.data.inputdir}	#Will be mapped to /tf/data on gpu\n'
scriptfile.write(line)
line=f'OUTPUT={cfg.run.outputdir}\n'
scriptfile.write(line)

line='\n'
scriptfile.write(line)
line=f'BASE_DIR=/nobackup/$USER/$PROJECTNAME\n'
scriptfile.write(line)
line=f'mkdir $BASE_DIR\n'
scriptfile.write(line)
line=f'mkdir $BASE_DIR/data\n'
scriptfile.write(line)
line=f'mkdir $BASE_DIR/outputs\n'
scriptfile.write(line)
#line=f'mkdir $BASE_DIR/logs\n'
#line=f'mkdir $BASE_DIR/plots\n'
 
line='\n'
scriptfile.write(line)
line=f'rsync -ahr --progress $LOCATION/$DATASET* $BASE_DIR/data\n'
scriptfile.write(line)
line=f'cp $SCRIPT_LOCATION/train*.py $SCRIPT_LOCATION/models.py $BASE_DIR/ $SCRIPT_LOCATION/setup_config.py $BASE_DIR/\n'
scriptfile.write(line)
line=f'cp $REQUIREMENTS $BASE_DIR/req.txt\n'
scriptfile.write(line)
line=f'cp $CONFIG $BASE_DIR/\n'
scriptfile.write(line)

line=f'#echo "Starting docker"\n'
scriptfile.write(line)

## Do I need tensorflow here or is it suifficient to have it in requirements?
line=f'docker run -it --rm --gpus '"device=0"' -v $BASE_DIR:/tf tensorflow/tensorflow:latest-gpu bash -c " pip install -r tf/req.txt && python tf/train.py"\n'
scriptfile.write(line)
##docker run -it --rm --gpus '"device=1"' -v $BASE_DIR:/tf bash -c " pip install -r tf/req.txt &&
##                                                                                                        python tf/train.py"

#rsync -ahr --progress $BASE_DIR/outputs/*.png $OUTPUT/plots
line=f'rsync -ahr --progress $BASE_DIR/outputs/*.npz $OUTPUT/predictions\n'
scriptfile.write(line)
line=f'rsync -ahr --progress $BASE_DIR/outputs/*.tf $OUTPUT/models\n'
scriptfile.write(line)
