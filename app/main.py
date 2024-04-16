
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import pyaudio
from preprocess_module import preprocess, vggish_loader, model_loader
import threading

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel,QHBoxLayout
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
#import librosa
#import librosa.display
import tensorflow as tf
import tensorflow_hub as hub
import keras
import time
import math
from matplotlib.animation import FuncAnimation
sys.stdout.reconfigure(encoding='utf-8')
class AudioInterface(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-time Audio Processing")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.hlayout = QHBoxLayout(self.central_widget)  

        self.start_button = QPushButton("Start Recording", self)
        self.start_button.clicked.connect(self.start_recording)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Recording", self)
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)

        self.prediction_label1 = QLabel(self)
        self.layout.addWidget(self.prediction_label1)

        self.prediction_label2 = QLabel(self)
        self.layout.addWidget(self.prediction_label2)
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, 22050*30)
        # self.ax.set_ylim(-0.5, 0.5)

        self.vggish = vggish_loader()
        self.model = model_loader()

        self.p = pyaudio.PyAudio()
        for i in range(self.p.get_device_count()):
        	dev=self.p.get_device_info_by_index(i)
        	name=dev['name'].encode('utf-8')
        	print(i,name,dev['maxInputChannels'],dev['maxOutputChannels'])
        print(self.p.get_default_input_device_info())
        print(self.p.get_default_output_device_info())
        self.stream = None
        self.CHUNK = 1200
        self.RATE = 48000
        self.recording = False
        self.recording_start_time = None

    def start_recording(self):
        print("start recording!!!!!!!!!!!!!!!!!!")
        self.recording = True
        self.recording_start_time = time.time()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,#單聲道
                                  rate=self.RATE,#取樣頻率
                                  input=True,#True=>錄音
                                  frames_per_buffer=self.CHUNK,
                                  input_device_index= 0 # added after importing to xavier
                                  )
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        #threading.Thread(target = self.plot_wave).start()
        self.plot_wave()

    def stop_recording(self):
        self.recording = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def plot_wave(self):
        frames = []
        self.ax.clear()
        
        print("start ploting!!!!!!!!!!!!!!!!!!")
        # while self.recording:
        #     p = pyaudio.PyAudio()
        #     stream = p.open(format=pyaudio.paFloat32,
        #             channels=1,
        #             rate=self.RATE,
        #             input=True,
        #             frames_per_buffer=1024)
                    
        start_index = 0
        for i in range(int(self.RATE / self.CHUNK * 30)):###原本1改成30
            
            print("running!!!!!!",i)
            #print(self.stream)
            
            data = self.stream.read(self.CHUNK,exception_on_overflow = False)
            
            # print("running!!!!!!")
            #print(data.shape)
            print(data)
            data=np.frombuffer(data, dtype=np.float32)
            print('data',data)
            frames.extend(data)
            
            # end_index = start_index + len(data)
            # self.ax.plot(range(start_index, end_index), data,color='blue')
            # print("running!!!!!!")              
            # Update the starting index for the next iteration
            # start_index = end_index
        frames=frames[::2]
        frames=frames[0:661500]
        self.ax.plot(frames,color='blue')
        self.canvas.draw()
        # print("canvas draw!!!!!!")
        print('frames:'+str(len(frames)))
        self.stop_recording()
        frames=np.array(frames)
        self.process_audio(frames)
       

        
# """
# for _ in range(int(self.RATE / 1024 * 30)):
#     print("running!!!!!!")
#     data = self.stream.read(1024)
#     frames.extend(data)
    
#     # Plot only the most recent 1024 samples
#     end_index = start_index + len(data)
#     self.ax.plot(range(start_index, end_index), data)
#     self.canvas.draw()
    
#     # Update the starting index for the next iteration
#     start_index = end_index
# """

    
    # self.p.terminate()
    
            # self.ax.clear()
            # self.ax.plot(samples)
            # self.canvas.draw()
            # if time.time() - self.recording_start_time >= 30:
            #     self.stop_recording()
            #     break
            # self.process_audio(samples)

    def process_audio(self, samples):
        print("Predict!!!!!")
        #Perform VGGish feature extraction
        vggish_features,vggish_features_shape = preprocess(samples, self.vggish)
        print(vggish_features_shape)
        # vggish_features = self.vggish(samples)
#     print(vggish_features.shape)
        v1 = vggish_features.reshape(1, vggish_features.shape[0], vggish_features.shape[1], 1)
        v2 = vggish_features.reshape(1, vggish_features.shape[0], vggish_features.shape[1],1)
#     print(type(v1))
        try:
            #model.summary()
            x = self.model.predict([v1, v2])
            print(x)
            print(type(x))
            class_labels=['disco','metal','reggae','blues','rock','classical', 'jazz', 'hiphop','country', 'pop']
            predicted_probabilities = {}
            for i, class_label in enumerate(class_labels):
                predicted_probabilities[class_label] = x[0][i] 
            # 打印每個類別與其對應的預測概率值
            print("各類別的預測概率值:")
            max1=0.0
            max2=0.0
            predictions=["",""]
            for class_label, prob in predicted_probabilities.items():
                if prob>max1 :
                    predictions[1]=predictions[0]
                    predictions[0]=class_label
                    max2=max1
                    max1=prob
                elif  prob>max2 :
                    predictions[1]=class_label
                    max2=prob
                print(f"{class_label}: {prob*100:.2f}%")
            print(predictions)
            self.prediction_label1.setText(f"Top 1 Prediction : "+predictions[0]+f" {max1*100:.2f}%")
            self.prediction_label2.setText(f"Top 2 Prediction : "+predictions[1]+f" {max2*100:.2f}%")
        except Exception as e:
            print("An error occurred:", e)
        # #Make prediction
        # prediction = self.model.predict(vggish_features)

        # #Update prediction label
        # #Here you need to adjust prediction label text according to your prediction
        # self.prediction_label.setText(f"Prediction: {prediction}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioInterface()
    window.show()
    sys.exit(app.exec_())


# import pyaudio
# import numpy as np
# import librosa
# import tensorflow_hub as hub
# import keras

# sr = 48000 
# duration = 30  

# def record_audio(sr, duration):
   
#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paFloat32,
#                     channels=1,
#                     rate=sr,
#                     input=True,
#                     frames_per_buffer=1440641)#1440641
  
#     frames = stream.read(1440641)
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
   

#     audio_data = np.frombuffer(frames, dtype=np.float32)

#     return audio_data

# def normalize_audio(audio_data, target_db):
#     # 计算音频数据的RMS值（Root Mean Square）
#     rms = np.sqrt(np.mean(np.square(audio_data)))

#     # 计算音频数据的分贝值
#     current_db = 20 * math.log10(rms / (2**16))

#     # 计算需要调整的增益值
#     gain = target_db - current_db

#     # 应用增益调整
#     normalized_audio_data = audio_data * (10**(gain / 20))

#     return normalized_audio_data
# def process_audio(audio_data, sr):

#     audio_data, _ = librosa.effects.trim(audio_data)
#     print(audio_data)
#     # normalized_audio = librosa.util.normalize(audio_data)
#     # # normalized = normalize_audio(audio_data,20)
#     # # print(normalized)
#     return audio_data

# if __name__ == "__main__":

#     audio_data = record_audio(sr, duration)
    
 
#     processed_audio = process_audio(audio_data, sr)
    

#     print("shape", processed_audio.shape)
#     vggish = hub.load('https://www.kaggle.com/models/google/vggish/frameworks/TensorFlow2/variations/vggish/versions/1')
#     model = keras.models.load_model("C:\\Users\\Lowen\\Downloads\\music_style\\music_style\\model5000.h5")  # corrected loading method
#     print(0)
#     vggish_features = vggish(audio_data).numpy()
#     print(1)
#     print(vggish_features.shape)
#     print("Model Input Shape:", model.input_shape)
#     print("Model Output Shape:", model.output_shape)
#     v1 = vggish_features.reshape(1, vggish_features.shape[0], vggish_features.shape[1], 1)
#     v2 = vggish_features.reshape(1, vggish_features.shape[0], vggish_features.shape[1],1)
#     print(type(v1))
#     try:
#         #model.summary()
#         x = model.predict([v1, v2])
#         print(x)
#         print(type(x))
#         # classes={0: 'blues',1: 'classical',2: 'country',3: 'folk',4: 'hiphop',5: 'jazz',6: 'metal',7: 'opera',8: 'pop',9: 'rock'}
#         class_labels=['blues','classical','country','folk','hiphop','jazz','metal','opera','pop','rock']
#         #class_labels=['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop','country', 'pop']
#         #class_labels=['folk','country','pop','rock','jazz','metal','classical','hiphop','opera','blues']
#         predicted_probabilities = {}
#         for i, class_label in enumerate(class_labels):
#             predicted_probabilities[class_label] = x[0][i]

#         # 打印每個類別與其對應的預測概率值
#         print("各類別的預測概率值:")
#         for class_label, probability in predicted_probabilities.items():
#             print(f"{class_label}: {probability*100:.2f}%")
#     except Exception as e:
#         print("An error occurred:", e)
