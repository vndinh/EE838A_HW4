clear; clc;

for v = 1:5
	video = VideoReader(sprintf('../data/60fps/video%d.mp4',v));
	num_of_frames = video.NumberofFrames;
	video_dir = sprintf('../data/60fps/video%d', v);
	if (~exist(video_dir, 'dir'))
      mkdir(video_dir)
    end
	for i = 1:num_of_frames
  	frameRGB = read(video, i);
  	imwrite(frameRGB, sprintf('../data/60fps/video%d/%06d.png', v, i));
	end
end
