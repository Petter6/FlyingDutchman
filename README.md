# FlyingDutchman

Install dependencies: \
python3.11 -m venv venv \
source venv/bin/activate \
pip install -r requirements.txt \

## How to use DAIC
Make a config file in /home/.ssh/config: \
Host daic \
  User preijalt \
  HostName login.daic.tudelft.nl \
  ProxyJump preijalt@student-linux.tudelft.nl \

Copy like this: \
scp [file] daic:/[directory]\

Login to daic: \
ssh <YouNetID>@login.daic.tudelft.nl \

Home-folder: \
/home/nfs/preijalt \

