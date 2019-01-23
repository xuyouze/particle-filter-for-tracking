# Created by if-pc at 2018/12/11

from tracker import Tracker

video_name = "RedTeam"
file_folder = "videos/" + video_name + "/img"
ground_truth_rec = "videos/" + video_name + "/groundtruth_rect.txt"
track_window = "tracker"

# is_color must be true if the video is colored, otherwise False
# The delimiter is the separator of the groundtruth_rect.txt some video is "." or " " or ","
tracker = Tracker(file_folder, ground_truth_rec, delimiter=",", is_color=True, video_name=video_name,
                  track_window=track_window)
tracker.run()
