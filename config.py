num_scenes = 20

# number of objects per scene
num_obj = 1

y_resolution = 540
x_resolution = 960

textures_enabled = True
bg_path = None

camera_pos = (0, 0, 0)

#depth of the objects
z_min, z_max = 1.5, 5

light_intensity = 90
light_orientation = ''
light_type = 'POINT'

gaussian_trans = False
# pixel translation / euler radian rot. 
sigma_trans, sigma_rot = 0.5, 0.0

uniform_trans = True
x_trans, y_trans, z_trans = 0.5, 0.0, 0.0

# dataset folder path (deletes folder if existent)
folder_path = '/Users/Petter/Documents/uni/thesis/RAFT/datasets/Sintel-0.5x_trans'
tmp_exr_path = './tmp/img_frame1.exr'