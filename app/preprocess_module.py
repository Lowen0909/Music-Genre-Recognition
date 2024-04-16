# preprocess_module.py
import tensorflow as tf
import tensorflow_hub as hub
import keras
#import librosa

def vggish_loader():
    try:          
        vggish = hub.load('https://www.kaggle.com/models/google/vggish/frameworks/TensorFlow2/variations/vggish/versions/1')
        print("load vggish  OK")
        return vggish
    except:
        print("Vggish load Error")
        return None

def model_loader():
    try:
        model = keras.models.load_model(r"/home/ubuntu/content/LSTMPCNN/model0411.h5")
        print("load model OK")
        return model
    except Exception as e:
        print("Model load Error:", e)
        return None


def preprocess(audio_data, vggish):
    sr = 22050
    try:
        vggish_features = vggish(audio_data).numpy()
        
        if len(vggish_features.shape) == 2:
            return vggish_features, vggish_features.shape
        else:
            print('提取的特征不是二维的。')
            return None, None
    except Exception as e:
        print(f"提取特徵時發生錯誤: {e}")
        return None, None
