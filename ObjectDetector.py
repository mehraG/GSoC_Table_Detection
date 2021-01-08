import cv2 as cv
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
import torch
import numpy as np
from PIL import Image

class Detector:

	def __init__(self):
		# obtain detectron2's default config
		self.cfg = get_cfg() 

		# load values from a file
		self.cfg.merge_from_file("./All_X101/All_X101.yaml") 

		# set device to cpu
		self.cfg.MODEL.DEVICE = "cpu"

		# get weights 
		self.cfg.merge_from_list(["MODEL.WEIGHTS", "./All_X101/model_final.pth"])
    
		# set the testing threshold for this model
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  


	# build model and convert for inference
	def convert_model_for_inference(self):
		# build model
		model = build_model(self.cfg)

		# save as checkpoint
		torch.save(model.state_dict(), 'checkpoint.pth')

		# return path to inference model
		return 'checkpoint.pth'


	# detectron model
	# adapted from detectron2 colab notebook: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5	
	def inference(self, file):
		predictor = DefaultPredictor(self.cfg)
		im = cv.imread(file)
		outputs = predictor(im)

		# get metadata
		metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

		# visualise
		v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

		# get image 
		img = Image.fromarray(np.uint8(v.get_image()))

		return img
