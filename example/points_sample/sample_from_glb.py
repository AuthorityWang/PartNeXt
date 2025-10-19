import numpy as np
import json
import trimesh
import gltf
from utils import scene2meshes
from scipy.spatial import cKDTree
from trimesh.path.path import Path3D
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial

from sampling import sample_surface

def load_mesh(obj_path):
    mesh = trimesh.load(obj_path)
    return mesh

def compute_inverse_mapping(meshes):
    face_inverse_mapping = []
    face_offsets = [0]
    # vertex_offsets = [0]
    for mesh in meshes:
        face_inverse_mapping.append(
            np.full([len(mesh.faces)], len(face_inverse_mapping))
        )
        face_offsets.append(face_offsets[-1] + len(mesh.faces))
        # vertex_offsets.append(vertex_offsets[-1] + len(mesh.vertices))
    face_inverse_mapping = np.concatenate(face_inverse_mapping)
    return face_inverse_mapping, face_offsets

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])  # Path compression
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)

        if rootP != rootQ:
            # Union by rank
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1


def find_duplicate_faces(mesh: trimesh.Trimesh, eps=1e-6):
    # Find unique vertices
    kd_tree = cKDTree(mesh.vertices)
    dd, ii = kd_tree.query(mesh.vertices, k=2)
    # Union-find to group vertices
    uf = UnionFind(len(mesh.vertices))
    for i, (d, j) in enumerate(zip(dd[:, 1], ii[:, 1])):
        if d < eps:
            uf.union(i, j)
    # Extract groups
    groups = {}
    mapping = []
    for i in range(len(mesh.vertices)):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
        mapping.append(root)

    # Remap faces
    new_faces = []
    for face in mesh.faces:
        new_face = [mapping[i] for i in face]
        new_faces.append(new_face)
    new_faces = np.sort(new_faces, axis=1)
    unique_faces, unique_indices, unique_inverse, unique_counts = np.unique(
        new_faces, axis=0, return_index=True, return_inverse=True, return_counts=True
    )

    is_duplicate_face = unique_counts[unique_inverse] > 1
    face_unique_ids = unique_indices[unique_inverse]
    return is_duplicate_face, face_unique_ids

def sample_from_glb(raw_mesh_path, num_samples, seed):
    glb_scene = load_mesh(raw_mesh_path)

    if not isinstance(glb_scene, trimesh.Scene):
        glb_scene = trimesh.Scene(geometry=[glb_scene])

    mesh_list = scene2meshes(glb_scene)

    merged_mesh = trimesh.util.concatenate(mesh_list)
    face_weight = np.array(merged_mesh.area_faces)

    if len(mesh_list) > 1:
        is_duplicate_face, _ = find_duplicate_faces(merged_mesh)
        # print(f"Found {is_duplicate_face.sum()} duplicate faces")

        is_simple_material = []
        for mesh in mesh_list:
            # check mesh.visual exists
            if isinstance(mesh, Path3D):
                return {}
            if (
                isinstance(mesh.visual, TextureVisuals)
                and mesh.visual.uv is None
                and isinstance(mesh.visual.material, SimpleMaterial)
                and np.all(mesh.visual.material.diffuse == (204, 204, 204, 255))
            ):
                is_simple_material.append(np.full([len(mesh.faces)], True))
            else:
                is_simple_material.append(np.full([len(mesh.faces)], False))
        is_simple_material = np.concatenate(is_simple_material)
        face_weight[is_simple_material & is_duplicate_face] = 0

    face_inverse_mapping, face_offsets = compute_inverse_mapping(mesh_list)
    assert len(face_inverse_mapping) == len(merged_mesh.faces)

    obj_points, obj_face_index, tri_uv = sample_surface(
        merged_mesh, num_samples, seed=seed, face_weight=face_weight
    )

    # [N, 2, 1]
    u, v = tri_uv[:, 0].squeeze(-1), tri_uv[:, 1].squeeze(-1)
    objaverse_point_materials = []
    for idx, submesh in enumerate(mesh_list):
        mask = face_inverse_mapping[obj_face_index] == idx
        sub_i = obj_face_index[mask] - face_offsets[idx]
        sub_u = u[mask]
        sub_v = v[mask]
        albedo, normal, metal, roughness = gltf.sample_features(
            submesh, sub_i, sub_u, sub_v
        )
        objaverse_point_materials.append((albedo, normal, metal, roughness))
    obj_colors = np.zeros([num_samples, 4])
    for i, (albedo, _, _, _) in enumerate(objaverse_point_materials):
        obj_colors[face_inverse_mapping[obj_face_index] == i] = albedo
    # clamp to 01
    obj_colors = np.clip(obj_colors, 0, 1)

    # get rgb, get rid of alpha
    obj_colors = obj_colors[:, :3]

    return obj_points, obj_colors, obj_face_index, mesh_list

def get_max_depth(node, current_depth):
    if "children" not in node:
        return current_depth
    return max([get_max_depth(child, current_depth + 1) for child in node["children"]])

def get_hierarchy_depth(hierarchy):
    return get_max_depth(hierarchy, 0)
    
def get_mask_id_list(hierarchy, depth):
    def collect_node_mask_id(node, current_depth, depth, result):
        if 'maskId' in node:
            if current_depth == depth:
                result.append([node['maskId']])
            elif current_depth < depth:
                result.append([node['maskId']])
            else:
                result.append(node['maskId'])

        if 'children' in node:
            if current_depth == depth:
                result.append([])
                for child in node['children']:
                    collect_node_mask_id(child, current_depth + 1, depth, result[-1])
            else:
                for child in node['children']:
                    collect_node_mask_id(child, current_depth + 1, depth, result)

    result = []
    collect_node_mask_id(hierarchy, 0, depth, result)
    return result

# collect the mask + all_submask for any node in the hierarchy
def get_mask_id_list(node):
    def merge_mask(mask_list, node):
        if 'maskId' in node:
            mask_list.append(node['maskId'])
        if 'children' in node:
            for child in node['children']:
                merge_mask(mask_list, child)
    
    result = []
    merge_mask(result, node)
    return result

# collect each node's mask id list
def get_all_node_mask_id_list(hierarchy):
    all_node_mask_id_list = {}
    def collect_node_mask_id(node):
        if node["refNodeId"] == 0 or node["refNodeId"] == -1:
            # root node
            pass
        else:
            # all_node_mask_id_list.append(node["maskId"])
            all_node_mask_id_list[node["nodeId"]] = get_mask_id_list(node)
        if "children" in node:
            for child in node["children"]:
                collect_node_mask_id(child)
    
    collect_node_mask_id(hierarchy)
    return all_node_mask_id_list

# regard everyone node (except root and other under root) as a mask
def get_mask_for_pcd_idx_full(obj_points, obj_colors, obj_face_index, mesh_list, annotation_json_path, num_samples):

    # load face2label
    with open(annotation_json_path, 'r') as f:
        mask_annotation = json.load(f)

    pcsam_masks = []
    pcsam_nodeids = []

    # collect all nodes's mask
    all_node_mask_id_list = get_all_node_mask_id_list(mask_annotation["hierarchyList"][0])

    mesh_face_list = mask_annotation["mesh_face_num"]
    for nodeId in all_node_mask_id_list:
        mask_list_idx = all_node_mask_id_list[nodeId]
        mask_faceid_list = []
        for mask_idx in mask_list_idx:
            for mesh_idx in list(mask_annotation["masks"][str(mask_idx)].keys()):
                mask_faceid_list.extend(
                    np.array(mask_annotation["masks"][str(mask_idx)][mesh_idx]) + 
                    np.sum([mesh_face_list[str(i)] for i in range(int(mesh_idx))])
                )
        mask_faceid_list = np.array(mask_faceid_list)
        pcsam_mask = np.isin(obj_face_index, mask_faceid_list)
        nonzero_count = np.count_nonzero(pcsam_mask)
        # all background
        if nonzero_count == 0:
            continue
        # all foreground
        if nonzero_count == num_samples:
            continue
        pcsam_masks.append(pcsam_mask)
        pcsam_nodeids.append(nodeId)

    result = dict(
        obj_points=obj_points,
        obj_colors=obj_colors,
        masks=np.array(pcsam_masks), 
        nodeids = np.array(pcsam_nodeids)
    )
    return result

# for any node, construct the full mask (self mask + all sub mask)
def glb2pcsam_idx_full(raw_mesh_path, annotation_json_path, num_samples, seed):
    obj_points, obj_colors, obj_face_index, mesh_list = sample_from_glb(raw_mesh_path, num_samples, seed)
    return get_mask_for_pcd_idx_full(obj_points, obj_colors, obj_face_index, mesh_list, annotation_json_path, num_samples)