import moviepy.editor as mpe
from moviepy.editor import *

def add_sound(in_path, out_path, out_path_sound):
    my_clip = mpe.VideoFileClip(out_path)

    video = VideoFileClip(in_path)  # 2.
    audio_background = video.audio  # 3.

    # audio_background = mpe.AudioFileClip('some_background.mp3')
    # final_audio = mpe.CompositeAudioClip([my_clip.audio, audio_background])
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(out_path_sound)
