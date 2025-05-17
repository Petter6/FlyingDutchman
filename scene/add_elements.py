import os 
import bpy
import numpy as np 

def add_fog(config):
    # Create a new cube at (0, 0, 0) with size 10
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), scale=(5, 5, 5))  # Scale 5 = size 10
    cube = bpy.context.active_object
    cube.name = "VolumetricCube"

    # Create a new material
    mat = bpy.data.materials.new(name="VolumetricFog")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # === Create Node Network ===

    # Texture Coordinate
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-1200, 0)

    # Mapping
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = (-1000, 0)
    mapping.inputs['Scale'].default_value = (1.0, 1.0, 1.0)
    mapping.vector_type = 'POINT'

    # Noise Texture
    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.location = (-800, 0)
    noise.noise_dimensions = '2D'
    noise.inputs['Scale'].default_value = 745.0
    noise.inputs['Distortion'].default_value = 12.4

    # Color Ramp
    ramp = nodes.new(type='ShaderNodeValToRGB')
    ramp.location = (-600, 0)
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[1].position = 0.15

    # Volume Scatter
    vol_scatter = nodes.new(type='ShaderNodeVolumeScatter')
    vol_scatter.location = (-200, -200)
    vol_scatter.inputs['Anisotropy'].default_value = -0.108

    # Principled Volume
    principled_vol = nodes.new(type='ShaderNodeVolumePrincipled')
    principled_vol.location = (-200, 100)
    principled_vol.inputs['Density'].default_value = 0.04
    principled_vol.inputs['Anisotropy'].default_value = 0.655
    principled_vol.inputs['Temperature'].default_value = 1000

    # Mix Shader
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (100, 0)
    mix_shader.inputs['Fac'].default_value = config['effects']['fog_percentage']

    # Material Output
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)

    # === Connect Nodes ===
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], vol_scatter.inputs['Density'])

    links.new(vol_scatter.outputs['Volume'], mix_shader.inputs[2])
    links.new(principled_vol.outputs['Volume'], mix_shader.inputs[1])

    links.new(mix_shader.outputs['Shader'], output.inputs['Volume'])

    # Assign the material to the cube
    cube.data.materials.append(mat)

def add_background(config):
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    nodes.clear()

    bg = nodes.new(type="ShaderNodeBackground")
    env_texture = nodes.new(type="ShaderNodeTexEnvironment")
    output = nodes.new(type="ShaderNodeOutputWorld")

    hdri_path = os.path.join(config['background']['2d_path'], np.random.choice(os.listdir(config['background']['2d_path']))) 
   
    env_texture.image = bpy.data.images.load(hdri_path)

    links.new(env_texture.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    return bg  # So we can animate it in `create_scene`
