from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from predict import predict_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Optional: 50MB upload limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No video uploaded", 400

        video = request.files['video']
        if video.filename == '':
            return "Empty filename", 400

        filename = secure_filename(video.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(path)

        prediction, avg_conf, frames = predict_video(path)

        return render_template('result.html',
                               filename=filename,
                               prediction=prediction,
                               confidence=round(avg_conf, 4),
                               frames=frames)
    else:
        return redirect(url_for('index'))