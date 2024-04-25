
import sys
import numpy as np
import time
import math
import keras
import pyaudio
import tensorflow as tf
from preprocess_module import preprocess, vggish_loader, model_loader
import threading

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel,QHBoxLayout,QStackedLayout
from PyQt5.QtCore import Qt
from PyQt5 import QtGui,QtWidgets
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from matplotlib.animation import FuncAnimation
#import librosa
#import librosa.display


sys.stdout.reconfigure(encoding='utf-8')
class GenreWindow(QWidget):
    def __init__(self):
        super().__init__()
        print("construct subwindow!!")
        layout = QHBoxLayout(self)

        #針對預測第一高的曲風建立layout顯示風格和機率
        container1 = QWidget()
        layout1=QVBoxLayout(container1)
        container1.setStyleSheet("background-color: #79FF79;")
        self.label1 = QLabel(self)
        self.label1p = QLabel(self)
        self.label1p.setAlignment(Qt.AlignCenter)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setStyleSheet("font-size: 40px; color: black;font-weight: bold;")
        self.label1p.setStyleSheet("font-size: 36px; color: black;font-weight: bold;")
        layout1.addWidget(self.label1)
        layout1.addWidget(self.label1p)

        #針對預測第二高的曲風建立layout顯示風格和機率
        container2 = QWidget()
        layout2=QVBoxLayout(container2)
        container2.setStyleSheet("background-color: #66B3FF;")
        self.label2 = QLabel(self)
        self.label2p = QLabel(self)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2p.setAlignment(Qt.AlignCenter)
        self.label2.setStyleSheet("font-size: 40px; color: black;font-weight: bold;")
        self.label2p.setStyleSheet("font-size: 36px; color: black;font-weight: bold;")
        layout2.addWidget(self.label2)
        layout2.addWidget(self.label2p)

        #將兩個重直layout整合進一個水平layout進行水平擺放
        layout.addWidget(container1)
        layout.addWidget(container2)
        self.setLayout(layout)

    def setContent(self,label1,label2,p1,p2):
        self.label1.setText(label1)
        self.label2.setText(label2)
        self.label1p.setText(f" {p1*100:.2f}%")
        self.label2p.setText(f" {p2*100:.2f}%")
    


class AudioInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QtGui.QIcon(r'logo1.png'))
        self.setWindowTitle("Real-time Audio Processing")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.hlayout = QHBoxLayout(self.central_widget)  
        
        #進度文字:READY、RECORDING、STOP、PREDICTING、FINISH
        self.progress_text = QLabel("READY",self)
        self.progress_text.setAlignment(Qt.AlignCenter)
        self.progress_text.setStyleSheet("font-size: 30px; color: black;font-weight: bold;")
        self.hlayout.addWidget(self.progress_text)

        #開始錄音按鈕
        self.start_button = QPushButton("Start Recording", self)
        self.start_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.start_button.clicked.connect(self.start_recording)
        self.hlayout.addWidget(self.start_button)
        
        #停止錄音按鈕
        self.stop_button = QPushButton("Stop Recording", self)
        self.stop_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        self.hlayout.addWidget(self.stop_button)
        self.layout.addLayout(self.hlayout)

        
        #stackedlayout把錄音波型圖和預測結果視窗上下堆疊擺放，一次只能顯示一個
        #setCurrentIndex(0)=>波形圖,setCurrentIndex(1)=>結果視窗
        self.stackedlayout=QStackedLayout(self)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.stackedlayout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, 22050*30)
        self.ax.set_axis_off()

        self.subwindows=GenreWindow()
        self.stackedlayout.addWidget(self.subwindows)
        self.stackedlayout.setCurrentIndex(0)
        self.layout.addLayout(self.stackedlayout)

        self.vggish = vggish_loader()
        self.model = model_loader()

        self.p = pyaudio.PyAudio()

        self.data=[]
        self.frames=[]
        # for i in range(self.p.get_device_count()):
        # 	dev=self.p.get_device_info_by_index(i)
        #     name=dev['name'].encode('utf-8')
        #     print(i,name,dev['maxInputChannels'],dev['maxOutputChannels'])
        print(self.p.get_default_input_device_info())
        print(self.p.get_default_output_device_info())
        self.stream = None
        self.CHUNK = 1200
        self.RATE = 48000
        self.recording = False
        self.recording_start_time = None

    def start_recording(self):
        self.stackedlayout.setCurrentIndex(0)
        self.subwindows.hide()
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
        threading.Thread(target = self.get_data,daemon=True).start()
        threading.Thread(target = self.plot_wave,daemon=True).start()

    # def kill_threads():
    #     threads = threading.enumerate()
    #     print(len(thread))
    #     for thread in threads:
    #         if thread.name != "MainThread":  
    #             print(f"Terminating thread: {thread.name}")
    #             thread.join(timeout=0)  

    def stop_recording(self):
        self.progress_text.setText("STOP")
        self.recording = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        # self.kill_threads()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    #負責收錄音資料
    def get_data(self):
        print("recording!!!!!!")
        self.frames=[]
        Record=["RECORDING","RECORDING.","RECORDING..","RECORDING..."]
        prev_index=0
        for i in range(int(self.RATE / self.CHUNK * 30)):###原本1改成30
            if i % 20== 0:  #progress text文字動態變動
                self.progress_text.setText(Record[prev_index])
                prev_index = (prev_index + 1) % 4 
            # print("running!!!!!!",i)
            #print(self.stream)
            if(self.recording==False):
                return
            data = self.stream.read(self.CHUNK,exception_on_overflow = False)
            
            # print("running!!!!!!")
            #print(data.shape)
            # print(data)
            data=np.frombuffer(data, dtype=np.float32)
            # print('data',data)
            self.frames.extend(data)
        self.stop_recording()
        self.frames=self.frames[::2]
        print("downsampling!")
        print(len(self.frames))
        audio_data=self.frames[0:661500]
        # print("canvas draw!!!!!!")
        print('Record finish ,frames:'+str(len(audio_data)))
        
        self.frames=np.array(audio_data)
        self.process_audio(self.frames)




    def plot_wave(self):
        self.ax.clear()
        
        print("start ploting!!!!!!!!!!!!!!!!!!")
        while self.recording: 
            self.ax.clear() 
            self.ax.set_xlim(0, 22050*30) 
            self.ax.set_ylim(-0.5,0.5) 
            self.ax.axis('off')     
            data=self.frames.copy()
            data=data[::2]
            if(len(data)>661500):
                data=data[:661500]
            self.ax.plot(data,color='blue')
            self.canvas.draw()
            
            # end_index = start_index + len(data)
            # self.ax.plot(range(start_index, end_index), data,color='blue')
            # print("running!!!!!!")              
            # Update the starting index for the next iteration
            # start_index = end_index        
        # print("canvas draw!!!!!!")
        print("Drawing finished!")
       

        
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
        self.progress_text.setText("PREDICTING♪")
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
            #設定介面progress text以及結果視窗的文字，並切換視窗
            self.progress_text.setText("FINISH!")
            self.subwindows.setContent(predictions[0].upper(),predictions[1].upper(),max1,max2)
            self.stackedlayout.setCurrentIndex(1)
        except Exception as e:
            print("An error occurred:", e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioInterface()
    window.show()
    sys.exit(app.exec_())



