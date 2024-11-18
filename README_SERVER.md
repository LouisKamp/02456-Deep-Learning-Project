# Run jupyter on HPC

0. `voltash`
1. `sh envDML.sh`
2. `source ./envDML/bin/activate`
3. `pip3 install -r server_requrements.txt`
4. `pip3 install jupyter`
5. `jupyter notebook --no-browser --port=40000 --ip=$HOSTNAME`