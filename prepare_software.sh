set -e
# deactivate virtualenv if already active
if command -v deactivate > /dev/null; then deactivate; fi
# python 3.8 or higher
if test -f synth-cell-env/bin/activate; then
echo "virtualenv already exists, skipping"
else
virtualenv -p python3 synth-cell-env
fi
# activate virtual environment
source synth-cell-env/bin/activate
# install python dependencies or use requirements.txt
pip3 install -r requirements.txt
set +e