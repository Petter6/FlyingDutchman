import bpy

def render_two_frames(frame1, frame2, scene, base_filepath):
    try:
        # Set the render engine to Cycles (required for GPU rendering)
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.feature_set = 'SUPPORTED'
        bpy.context.scene.cycles.sample_clamp_indirect = 10.0

        bpy.context.scene.cycles.samples = 128  # Adjust this value as needed
        bpy.context.scene.cycles.use_adaptive_sampling = True

        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # or 'OPTIX' if available

        bpy.context.scene.cycles.max_bounces = 8  # Adjust as needed
        bpy.context.scene.cycles.diffuse_bounces = 4
        bpy.context.scene.cycles.glossy_bounces = 4
        bpy.context.scene.cycles.transmission_bounces = 4

        bpy.context.scene.cycles.use_persistent_data = True

        # Enable GPU rendering
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()

        # Ensure the GPU device is selected for rendering
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use = True  # Enable all available devices (GPU)

        #Set the resolution
        bpy.context.scene.render.resolution_x = 960
        bpy.context.scene.render.resolution_y = 540
        bpy.context.scene.render.resolution_percentage = 100  # Ensure it's not scaled down

        # Ensure the active view layer exists
        view_layer = bpy.context.view_layer

        if view_layer is None:
            print("No active view layer found. Please ensure a valid view layer is selected.")
        else:
            # Access render pass properties safely
            try:
                view_layer.use_pass_combined = True
                print("Successfully set use_pass_combined.")
            except AttributeError:
                print("Failed to set use_pass_combined. Check render settings and engine.")

        # Enable necessary passes
        # view_layer.use_pass_combined = True
        view_layer.use_pass_vector = True
        # view_layer.use_pass_z = True
    
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'

         # Render the second frame
        bpy.context.scene.frame_set(frame2)  # Set to the second frame
        bpy.context.scene.render.filepath = f"./tmp/img_frame1.exr"
        bpy.ops.render.render(write_still=True)

        print("hallo")

        bpy.context.scene.render.image_settings.file_format = 'PNG'

        # Render the first frame
        bpy.context.scene.frame_set(frame1)  # Set to the first frame
        bpy.context.scene.render.filepath = f"{base_filepath}/frame{frame2}.png"
        bpy.ops.render.render(write_still=True)

        # Render the second frame
        bpy.context.scene.frame_set(frame2)  # Set to the second frame
        bpy.context.scene.render.filepath = f"{base_filepath}/frame{frame1}.png"
        bpy.ops.render.render(write_still=True)

        print("Rendering complete.")

    except Exception as e:
        print(f"Error during rendering: {e}")