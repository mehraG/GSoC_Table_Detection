Prerequisite:
- PyTorch
- OpenCV
- Flask
See https://detectron2.readthedocs.io/tutorials/install.html for detailed installation.

Steps to download Table detection model:
- Go to https://buaaeducn-my.sharepoint.com/:f:/g/personal/liminghao1630_buaa_edu_cn/Esy5fgoxZTVImem0R0DFyNMB134ZalOOhBiYgMj6CbjfnQ?e=JnnjnD
Source: https://github.com/doc-analysis/TableBank/
- Go to All_X101
- Download "model_final.pth"
- Paste this model in All_X101 folder inside your project.

File description:
"app.py":
Contains code of Flask to run a basic website.
To run the website type: "python3 app.py" it the terminal.

"ObjectDetector.py":
Contains the code for the Table detection model. Framework used is detectron2.

"Detectron2_Tutorial.ipynb":
This is a Jupyter Notebook which also do table detection. 
Go through it to get familiar with detectron2 code.
