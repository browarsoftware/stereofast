from StereoGeneratorEngine import StereoGeneratorEngine as sge
from keras.models import load_model
from StereoGeneratorEngine.layers import BilinearUpSampling2D

custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
model_file = "modelOK_T_NYU.h5"
#model_file = "nyu.h5"#resolution=(1280, 720)
model = load_model(model_file, custom_objects=custom_objects, compile=False)
model.summary()

sge.generate_stereo_from_VideoCapture(
    model,
    show_depth=True,
    show_frames=True,
    show_fps = True#,
    #resolution=(1280, 720)
)