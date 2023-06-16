from midas.StereoGeneratorEngine import StereoGeneratorEngine as sge

live_camera = True

if not live_camera:
    #files names, can be dowloaded from https://drive.google.com/drive/folders/1xt5gVhP2kyXIWQe8xpC3nwCDejKFD0Zw?usp=sharing
    # subfolder video
    my_file = ['banka.mp4', 'biblioteka.mp4', 'dolina.mp4', 'ekspres.mp4', 'kolo.mp4', 'kwiatki.mp4', 'lotnisko.mp4',
               'lwice.mp4',
               'mycie_rak.mp4', 'owce.mp4', 'papuga.mp4', 'porsche.mp4', 'port.mp4', 'ptaki.mp4', 'rowerzysta.mp4',
               'seoul.mp4', 'szlak.mp4',
               'telefon.mp4', 'wiatrak.mp4', 'wodospad.mp4', 'wulkan.mp4', 'zaba.mp4', 'zegarek.mp4']
    # Use one of:
    # "MiDaS_small"
    # "DPT_Hybrid"
    # "DPT_Large"
    [midas, device, transform] = sge.prepare_dnn("DPT_Large")
    for dd in [25,50,75]:
        maxdeviation = dd
        for ff in my_file:
            #path to video files for conversion
            in_path = 'd:/data/video/' + ff
            out_path = 'help.mp4'
            #ouput patj
            out_path_sound = 'd:/data/video/results/' + ff
            sge.generate_stereo_from_VideoCapture(
                    device, midas, transform,
                    #horizontal = False,
                    #include_sound = True,
                    show_depth = False,
                    show_frames = False,
                    show_fps = False,
                    resolution = (1280, 720),#(640, 360)
                    theta = 0.75,
                    trim_left=0,
                    maxdeviation = maxdeviation,
                    in_path = in_path, out_path = out_path, out_path_sound = out_path_sound)
else:
    # Use one of:
    # "MiDaS_small"
    # "DPT_Hybrid"
    # "DPT_Large"
    [midas, device, transform] = sge.prepare_dnn("MiDaS_small")
    sge.generate_stereo_from_VideoCapture(
                    device, midas, transform,
                    show_depth=True,
                    show_frames=True,
                    show_fps = True
                    )

