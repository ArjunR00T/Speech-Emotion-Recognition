from flask import Flask, render_template, request, redirect, url_for
import librosa
import soundfile as sf
import os, glob, pickle
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_feature(file_name, mfcc, chroma, mel):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print('ulpoading')
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = "audio.wav"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        with open('ser_model78.pkl', 'rb') as file:  # Open in binary reading mode
            newmodel = pickle.load(file)

        file = 'uploads/audio.wav'
        y, sr = librosa.load(file)

        num_samples = int(3 * sr)

        # Trim audio using slicing
        trimmed_audio = y[:num_samples]
        sf.write('uploads/trimeedd.wav', trimmed_audio, sr)

        file = 'uploads/trimeedd.wav'

        feature=extract_feature(file, mfcc=True, chroma=True, mel=False)
        feature = feature.reshape(1, -1)

        prediction = newmodel.predict(feature)
        print(prediction)

        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
