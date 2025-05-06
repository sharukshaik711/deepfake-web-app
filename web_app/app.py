from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from predict import predict_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return "No video uploaded", 400

    video = request.files['video']
    filename = secure_filename(video.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(path)

    # Optional: Compress or resize video here with ffmpeg for performance (can be added later)

    return redirect(url_for('process_video', filename=filename))

@app.route('/process/<filename>')
def process_video(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        prediction, avg_conf, frames = predict_video(path)
    except Exception as e:
        return f"Error processing video: {str(e)}", 500

    return render_template('result.html',
                           filename=filename,
                           prediction=prediction,
                           confidence=round(avg_conf, 4),
                           frames=frames)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)