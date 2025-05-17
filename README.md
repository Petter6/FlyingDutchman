# FlyingDutchman

Install dependencies: 
```
python3.11 -m venv venv 
source venv/bin/activate 
pip install -r requirements.txt 
```

Building the container:
```
apptainer build blender_env.sif blender_env.def
```

## How to use DAIC
Make a config file in /home/.ssh/config: 
```
Host daic 
  User preijalt 
  HostName login.daic.tudelft.nl 
```

Copy like this: 
```
scp [file] daic:/[directory]
```

Login to daic: 
```
ssh daic
```

Home-folder: 
```
/home/nfs/preijalt
```

Running the .sif container (nv flag needed for gpu):
```
apptainer run --nv blender_env.sif --mode create --config config/create/base_config.json
```

Getting an interactive sessions (with a GPU):
```
sinteractive --gres=gpu:1 --mem=2G --time=00:10:00
```

