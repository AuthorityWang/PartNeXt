import trimesh
import numpy as np
import colorsys

def norm_mesh_or_scene(mesh):
    rescale = max(mesh.extents)/2.
    tform = [
        -(mesh.bounds[1][i] + mesh.bounds[0][i])/2.
        for i in range(3)
    ]
    matrix = np.eye(4)
    matrix[:3, 3] = tform
    mesh.apply_transform(matrix)
    matrix = np.eye(4)
    matrix[:3, :3] /= rescale
    mesh.apply_transform(matrix)
    return mesh

def scene2meshes(scene):
    if not isinstance(scene, trimesh.Scene):
        raise ValueError("Input must be trimesh.Scene")
    meshes = []
    geometry_nodes = scene.graph.geometry_nodes
    for name, geometry in scene.geometry.items():
        if isinstance(geometry, trimesh.Trimesh):
            if name not in geometry_nodes:
                raise ValueError(f"Transform of geometry node {name} not found")
            object_node_name = geometry_nodes[name]
            # should only have exactly one object node
            if (len(object_node_name) != 1):
                raise ValueError(f"Expected one object node for geometry node {name}")
            object_node_name = object_node_name[0]
            if object_node_name not in scene.graph:
                raise ValueError(f"Object node {object_node_name} not found")
            
            # in gltf, geometry name can be different from object node name
            # so find the object node of the geometry
            # and apply the transform of the object node to the geometry
            transform, _ = scene.graph[object_node_name]
            geometry.apply_transform(transform)
            meshes.append(geometry)
        else:
            raise ValueError("Scene must contain only trimesh.Trimesh")
    return meshes

def generate_mask_color(num_masks):
    hues = np.linspace(0, 1, num_masks, endpoint=False)
    colors = []
    for hue in hues:
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append([r, g, b])
    colors = np.array(colors)
    return colors

def generate_pcd_mask_color(masks):
    n_masks = len(masks)
    num_points = masks[0].shape[0]
    colors = generate_mask_color(n_masks)
    part_vis_colors = np.zeros((num_points, 3))
    for i in range(n_masks):
        mask = masks[i]
        part_vis_colors[mask] = colors[i]
    return part_vis_colors