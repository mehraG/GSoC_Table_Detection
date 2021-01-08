from ObjectDetector import Detector
import io
from flask import Flask, render_template, request, send_file
from PIL import Image
import os

app = Flask(__name__)
detector = Detector()


def run_inference(img_path = 'file.jpg'):
	# run inference using detectron2
	result_img = detector.inference(img_path)
	# clean up
	try:
		os.remove(img_path)
	except:
		pass

	return result_img


@app.route("/")
def index():
	return render_template('index.html')


@app.route("/detect", methods=['POST'])
def upload():
	if request.method == 'POST':

		try:
			# open image
			file = Image.open(request.files['file'].stream)
			# remove alpha channel
			rgb_im = file.convert('RGB')
			rgb_im.save('file.jpg')
		
		# failure
		except:
			return render_template("failure.html")

	# run inference
	result_img = run_inference('file.jpg')

	# create file-object in memory
	file_object = io.BytesIO()

	# write PNG in file-object
	result_img.save(file_object, 'PNG')

	# move to beginning of file so `send_file()` it will read from start    
	file_object.seek(0)

	return send_file(file_object, mimetype='image/jpeg')

if __name__ == "__main__":
	# get port. Default to 8080
	port = int(os.environ.get('PORT', 8080))

	# run app
	app.run(host='0.0.0.0', port=port, debug=True)
