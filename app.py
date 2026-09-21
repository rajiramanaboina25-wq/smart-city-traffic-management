#import require python classes and packages

from flask import *
from auth_utils import *
from werkzeug.utils import secure_filename
import os,random
import matplotlib
matplotlib.use('Agg')
from functools import wraps
#loading python classes and packages
import os
import numpy as np
import librosa
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, RepeatVector, Bidirectional, LSTM, GRU, AveragePooling2D
from keras.layers import Convolution2D
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt   
from sklearn.neural_network import MLPClassifier
import IPython
import warnings
warnings.filterwarnings("ignore")


random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
MAX_UPLOAD_SIZE_MB = 512  # Maximum upload size in megabytes

app = Flask(__name__)
app.secret_key = "sdkfksdjfklsjdfkljsdfklj"  # Replace with a secure secret key


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # 

#define global variables to save accuracy and other metrics
accuracy = []
precision = []
recall = []
fscore = []

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/login')
def login():
    return render_template('login.html')


@app.route("/signup")
def signup_route():
    return signup()


@app.route("/signin")
def signin_route():
    return signin()

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

#function to extract acoustic features from audio files and this features include, RMSE, MFC, MEL spec and many more
def acoustic_features(X, sample_rate):
    stft = np.abs(librosa.stft(X))
    pitches, magnitudes = librosa.piptrack(X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])
    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))
    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)
    rmse = librosa.feature.rmse(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)
    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])
    ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))
    return ext_features

def extract_features(file, pad = False):
    X, sample_rate = librosa.load(file, sr = None)
    max_ = X.shape[0] / sample_rate
    if pad:
        length = (max_ * sample_rate) - X.shape[0]
        X = np.pad(X, (0, int(length)), 'constant')
    return acoustic_features(X, sample_rate)

#define function to get class label of given speaker audio file
def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

@app.route('/load_dataset')
@login_required
def load_dataset():
    global dataset,labels,path
    try:
        #defining global variables to save training data
        path = "Dataset"
        labels = []
        X = []
        Y = []
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if name not in labels:
                    labels.append(name.strip())
        print("Different Class Labels Found in Dataset : "+str(labels))

        message = "Dataset loaded successfully."
    except FileNotFoundError:
        message = "Error: Could not load dataset."
    except Exception as e:
        message = "An error occurred while loading the dataset."

    # Render the template and display the dataset and message
    return render_template("dataset.html", message=message)

@app.route("/preprocess")
@login_required
def preprocess_and_split_data():
    global X_train, X_test, y_train, y_test
    #function to load audio files and then extract acoustic features
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X.clear()
        Y.clear()
        for root, dirs, directory in os.walk(path):#loop all audio files
            for j in range(len(directory)):
                features = extract_features(root+"/"+directory[j])#call function to extract acoustic features from loaded audio file
                name = os.path.basename(root)
                label = getLabel(name)
                X.append(features)#add acoustic features and labels to X and Y training array
                Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    print("Dataset Audio Files Loaded")
    print("Total audio files found in Dataset = "+str(X.shape[0]))  
    names, count = np.unique(Y, return_counts = True)
    #shuffling and processing audio data files
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    Y = to_categorical(Y)
    print("Shuffling Dataset audio files completed")
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    data = np.load("model/data.npy", allow_pickle=True)
    X_train, X_test, y_train, y_test = data
    print("Dataset Train & Test Split Details")
    print("70% audio files used to train algorithms : "+str(X_train.shape[0]))
    print("30% audio files used to test algorithms : "+str(X_test.shape[0]))
    total_size = X.shape[0]
    training_size = str(X_train.shape[0])
    testing_size = str(X_test.shape[0])

    return render_template("preprocess.html", 
                           total_size=total_size,
                           training_size=training_size, 
                           testing_size=testing_size,
                           train_percentage=80, 
                           test_percentage=20)

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    p = round(precision_score(testY, predict,average='macro') * 100, 2)
    r = round(recall_score(testY, predict,average='macro') * 100, 2)
    f = round(f1_score(testY, predict,average='macro') * 100, 2)
    a = round(accuracy_score(testY,predict)*100, 2)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+" Accuracy  : "+str(a))
    print(algorithm+" Precision : "+str(p))
    print(algorithm+" Recall    : "+str(r))
    print(algorithm+" FSCORE    : "+str(f))   
    return a,p,r,f



@app.route('/existing_alg')
@login_required

def existing_algorithm():
   #training alone MLP classifier
    mlp = MLPClassifier(solver="sgd", activation='tanh', max_iter=20)
    #training MLP on training features
    mlp.fit(X_train, y_train)
    #performing prediction on test data
    predict = mlp.predict(X_test)
    #call function to calculate accuracy and other metrics
    a,p,r,f = calculateMetrics("MLP", y_test, predict)
    # Pass metrics to the template
    return render_template("existing_alg.html", 
                           accuracy=a, 
                           precision=p,
                           recall=r,
                           fscore=f 
                           )


@app.route('/proposed_alg')
@login_required

def proposed_algorithm():
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    #training propose stacked LSTM algorithm by combining MLP, DNN and LSTM as stacked ensemble algorithm 
    stacked_lstm_model = Sequential()
    #defining fully connected MLP layer
    stacked_lstm_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
    stacked_lstm_model.add(MaxPooling2D(pool_size = (1, 1)))
    stacked_lstm_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    stacked_lstm_model.add(MaxPooling2D(pool_size = (1, 1)))
    stacked_lstm_model.add(Flatten())
    stacked_lstm_model.add(RepeatVector(3))
    #adding LSTM layer as stacked to MLP
    stacked_lstm_model.add(Bidirectional(LSTM(32, activation = 'relu')))#==================adding LSTM
    #adding dnn dense layer  
    stacked_lstm_model.add(Dense(units = 64, activation = 'relu'))
    stacked_lstm_model.add(Dense(units = y_train1.shape[1]))
    #compiling, training and loading model
    stacked_lstm_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/stacked_lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/stacked_lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = stacked_lstm_model.fit(X_train1, y_train1, batch_size = 32, epochs = 30, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/stacked_lstm_hist.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        stacked_lstm_model.load_weights("model/stacked_lstm_weights.hdf5")
    #perform prediction on test data
    predict = stacked_lstm_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test2 = np.argmax(y_test1, axis=1)
    #call function to calculate accuracy and other metrics
    a,p,r,f = calculateMetrics("Propose Stacked LSTM", y_test, predict)
    return render_template("proposed_alg.html", 
                        accuracy=a, 
                           precision=p,
                           recall=r,
                           fscore=f
                     
                           )

@app.route('/extension_alg')
@login_required

def extension_algorithm():
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    #defining extension model by stacking MLP + DNN + LSTM and GRU
    extension_gru_model = Sequential()
    #defining fully connected MLP layer
    extension_gru_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
    extension_gru_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_gru_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    extension_gru_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_gru_model.add(Flatten())
    extension_gru_model.add(RepeatVector(3))
    #adding LSTM layer as stacked to MLP
    extension_gru_model.add(Bidirectional(LSTM(32, activation = 'relu')))#==================adding LSTM
    extension_gru_model.add(RepeatVector(3))
    #adding GRU layer as stacked to MLP
    extension_gru_model.add(Bidirectional(GRU(32, activation = 'relu')))#==================adding GRU
    #adding dnn dense layer 
    extension_gru_model.add(Dense(units = 32, activation = 'relu'))
    extension_gru_model.add(Dropout(0.3))
    extension_gru_model.add(Dense(units = y_train1.shape[1]))
    #compiling, training and loading model
    extension_gru_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/extension_gru_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_gru_weights.hdf5', verbose = 1, save_best_only = True)
        hist = extension_gru_model.fit(X_train1, y_train1, batch_size = 32, epochs = 30, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/extension_gru_hist.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        extension_gru_model.load_weights("model/extension_gru_weights.hdf5")

    
    predict = extension_gru_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test2 = np.argmax(y_test1, axis=1)
    #call function to calculate accuracy and other metrics
    a,p,r,f = calculateMetrics("Extension Stacked LSTM + GRU", y_test2, predict)
    return render_template("extension_alg.html", 
                           accuracy=a, 
                           precision=p,
                           recall=r,
                           fscore=f
                           )


@app.route('/display_graph')
@login_required

def display_graph():
    df = pd.DataFrame([
        ['MLP','Accuracy',accuracy[0]],
        ['MLP','Precision',precision[0]],
        ['MLP','Recall',recall[0]],
        ['MLP','FSCORE',fscore[0]],
        ['Propose Stacked LSTM','Accuracy',accuracy[1]],
        ['Propose Stacked LSTM','Precision',precision[1]],
        ['Propose Stacked LSTM','Recall',recall[1]],
        ['Propose Stacked LSTM','FSCORE',fscore[1]],
        ['Extension Stacked LSTM + GRU','Accuracy',accuracy[2]],['Extension Stacked LSTM + GRU','Precision',precision[2]],['Extension Stacked LSTM + GRU','Recall',recall[2]],
        ['Extension Stacked LSTM + GRU','FSCORE',fscore[2]],
                  ],columns=['Parameters','Algorithms','Value'])

    # Create the bar graph
    fig, ax = plt.subplots(figsize=(5, 3))  # Increase the figure size (10x6 inches)
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', ax=ax)

    # Set the title and labels
    ax.set_title("Algorithms Performance Comparison", fontsize=6)
    ax.set_xlabel("Metrics", fontsize=4)
    ax.set_ylabel("Values", fontsize=4)
    
    # Adjust tick labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(fontsize=4)

    # Move the legend outside the plot
    plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=4)

    # Save the graph as an image with high DPI for clarity
    graph_path = os.path.join(app.static_folder, 'graph.png')
    plt.tight_layout()  # Ensure everything fits well
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')  # Save the plot with 300 DPI for higher resolution
    plt.close()  # Close the plot to free memory

    # Render the HTML template and pass the image path
    return render_template("graph.html", graph_url='/static/graph.png')


def getModel():
    extension_gru_model = Sequential()
    extension_gru_model.add(Convolution2D(32, (1 , 1), input_shape = (312, 1, 1), activation = 'relu'))
    extension_gru_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_gru_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    extension_gru_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_gru_model.add(Flatten())
    extension_gru_model.add(RepeatVector(3))
    extension_gru_model.add(Bidirectional(LSTM(32, activation = 'relu')))#==================adding BILSTM
    extension_gru_model.add(RepeatVector(3))
    #adding GRU layer
    extension_gru_model.add(Bidirectional(GRU(32, activation = 'relu')))#==================adding BILSTM
    #defining dense layer with 256 neurons 
    extension_gru_model.add(Dense(units = 32, activation = 'relu'))
    extension_gru_model.add(Dropout(0.3))
    extension_gru_model.add(Dense(units = 2))
    extension_gru_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    extension_gru_model.load_weights("model/extension_gru_weights.hdf5")
    return extension_gru_model

@app.route('/predict')
@login_required

def upload():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
@login_required

def upload_file():

    if 'testdata' not in request.files:
        return render_template('predict.html', message='No file selected')

    dataset = request.files['testdata']

    if dataset.filename == '':
        return render_template('predict.html', message='No selected file')

    if not allowed_file(dataset.filename):
        return render_template('predict.html', message='Allowed file types: .wav, .mp3')

    filename = secure_filename(dataset.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # 1️⃣ Save uploaded file
        dataset.save(filepath)

        # 2️⃣ Copy saved file to static folder
        test_wav = "static/test.wav"
        if os.path.exists(test_wav):
            os.remove(test_wav)

        with open(filepath, "rb") as f:
            audio_bytes = f.read()

        with open(test_wav, "wb") as f:
            f.write(audio_bytes)

        # 3️⃣ Load model
        extension_model = getModel()

        # 4️⃣ Extract features
        features = extract_features(test_wav)

        temp = np.array([features])
        temp = temp.reshape(temp.shape[0], temp.shape[1], 1, 1)

        # 5️⃣ Predict
        prediction = extension_model.predict(temp)
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]

        print("Predicted Class:", predicted_label)

        return render_template('predict.html', results=predicted_label)

    except Exception as e:
        return render_template('predict.html', message=str(e))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    app.run(debug=True)