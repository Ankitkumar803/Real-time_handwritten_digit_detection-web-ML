from flask import Flask,render_template,request
from flask.json import jsonify
import digit_detect
from PIL import Image
import numpy as np
import base64
import io

app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return render_template('index.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        img_data = request.form['file']
        img_data = img_data.split(",",1)[1]
        # converting data68 sting to jpeg
        base64_decoded = base64.b64decode(img_data)
        img_data = Image.open(io.BytesIO(base64_decoded))


        imgGray = img_data.convert('L')
        #imgGray = imgGray.resize((28, 28))
        imgGray_np = np.array(imgGray)

        
        #converting gray scale to pixel of black-white
        imgGray_np[imgGray_np >= 120] = 255
        imgGray_np[imgGray_np < 120] = 0

        img = Image.fromarray(imgGray_np)
        img.save('my.png')

        imgGray_np = img.resize((28, 28))

        imgGray_np = np.ravel(imgGray_np)
        np.savetxt('imgGray_np.csv', imgGray_np, delimiter=',')

        all_theta = np.loadtxt('2darray.csv', delimiter=',')
        pred = digit_detect.predictOneVsAll(all_theta, imgGray_np)
        print("prediction",pred)
        return jsonify(request.form['userID'], request.form['file'])
        
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)