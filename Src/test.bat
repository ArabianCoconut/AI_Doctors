cd Downloads
curl -O https://www.python.org/ftp/python/3.10.7/python-3.10.7-amd64.exe

python-3.10.7-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
pip install --upgrade pip
pip install flask
pip install tensorflow
pip install keras
pip install numpy
pip install matplotlib
pip install glob
pip install os
pip install re
pip install opencv-python
pip install pillow

flask --app ai.py run
start localhost:5000
Echo "Setup Complete" >> setup_Result.txt
