**Credit**
The original authors are
Kang Wei
Jun Li
Chuan Ma
Ming Ding
Sha Wei
Fan Wu
Guihai Chen
Thilina Ranbaduge
Corresponding paper can be found here: https://arxiv.org/abs/2202.04309
Original github repository can be found here: https://github.com/AdamWei-boop/Vertical_FL


**Changes from me**
We intend to use this code for our course project that investigates drop out clients. Hence, I have removed DP and contribution calculation parts. I have also added logic for active clients that are still participating using a variable that can be toggled called “active_clients.”

**To get started:**
pipenv install -r requirements.txt
Python torch_vertical_FL_train_dropout_clients.py

You dont have to run the code using the args listed in the code. You can run the code directly using the above command. 
