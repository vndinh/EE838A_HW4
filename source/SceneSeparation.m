clear; clc;

all_scene_separation = csvread('../data/scene_separation_hw4.csv');
all_scene_separation = int16(all_scene_separation);
num_of_video = 5;
for video = 1:num_of_video
  column1 = 2*(video-1) + 1;
  column2 = column1 + 1;
  video_scene_separation = all_scene_separation(:,column1:column2);
  video_scene_separation(~any(video_scene_separation,2),:) = [];
  num_of_scenes = length(video_scene_separation);
  for scene = 1:num_of_scenes
    scene_path = sprintf('../data/train/video%d/scene_%03d', video, scene);
    if (~exist(scene_path, 'dir'))
      mkdir(scene_path)
    end
    for frame = video_scene_separation(scene,1):video_scene_separation(scene,2)
      frame_path = sprintf('../data/60fps/video%d/%06d.png', video, frame);
      copyfile(frame_path, scene_path);
    end
  end
end
