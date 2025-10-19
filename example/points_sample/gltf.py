import numpy
import logging
from trimesh import Trimesh, Scene
from trimesh.visual import ColorVisuals, TextureVisuals
from trimesh.visual.material import SimpleMaterial, MultiMaterial, PBRMaterial
from calibur import unbind, sample2d


def vertex_weighted(vertex_attrs, i0, i1, i2, w, u, v):
    w, u, v = w[..., None], u[..., None], v[..., None]
    return vertex_attrs[i0] * w + vertex_attrs[i1] * u + vertex_attrs[i2] * v


def none_to_one(x):
    logging.info("Defaulting MUL factor to 1")
    return 1 if x is None else x

def none_to_255(x):
    logging.info("Defaulting MUL factor to 255")
    return 255 if x is None else x


def handle_single_material(mat, uv):
    if isinstance(mat, SimpleMaterial):
        logging.info("GLTF/S")
        mat = mat.to_pbr()
    assert isinstance(mat, PBRMaterial), type(mat)
    # mat.
    if mat.baseColorTexture is None:
        logging.info("GLTF/PBR/Albedo/RGBA")
        albedo = numpy.broadcast_to(none_to_255(mat.baseColorFactor), [len(uv), 4]) / 255.0
    else:
        logging.info("GLTF/PBR/Albedo/Texture")
        if mat.baseColorFactor is None:
            logging.info("GLTF/PBR/Albedo/Texture/FNone")
            albedo = sample2d(mat.baseColorTexture.convert("RGBA"), uv) / 255.0
        else:
            albedo = mat.baseColorFactor / 255.0 * sample2d(mat.baseColorTexture.convert("RGBA"), uv) / 255.0
    if mat.normalTexture is None:
        logging.info("GLTF/PBR/Normal/Default")
        normal_tangent = numpy.zeros([len(uv), 3])
        normal_tangent[..., 2] = 1
    else:
        logging.info("GLTF/PBR/Normal/Texture")
        normal_tangent = sample2d(mat.normalTexture.convert("RGBA"), uv)[..., :3] / 127 - 128 / 127
    if mat.metallicRoughnessTexture is None:
        logging.info("GLTF/PBR/R/Values")
        roughness = numpy.broadcast_to(none_to_one(mat.roughnessFactor), [len(uv)])
        logging.info("GLTF/PBR/M/Values")
        metal = numpy.broadcast_to(none_to_one(mat.metallicFactor), [len(uv)])
    else:
        logging.info("GLTF/PBR/R/Texture")
        roughness = none_to_one(mat.roughnessFactor) * sample2d(mat.metallicRoughnessTexture.convert("RGBA"), uv)[..., 1] / 255.0
        logging.info("GLTF/PBR/M/Texture")
        metal = none_to_one(mat.metallicFactor) * sample2d(mat.metallicRoughnessTexture.convert("RGBA"), uv)[..., 2] / 255.0
    return albedo, normal_tangent, metal, roughness


def sample_features(mesh: Trimesh, i, u, v):
    # i, u, v: [M]
    i0, i1, i2 = unbind(mesh.faces[i], axis=-1)  # [M, 3]
    w = 1 - u - v
    albedo = numpy.zeros([len(i), 4])
    normal_tangent = numpy.zeros([len(i), 3])
    normal_tangent[..., 2] = 1
    metal = numpy.zeros([len(i)])
    roughness = numpy.ones([len(i)]) * 0.5
    if isinstance(mesh.visual, ColorVisuals):
        if mesh.visual.kind is None:
            logging.info("GLTF/C/N")
            albedo = numpy.broadcast_to(mesh.visual.main_color, [len(i), 4]) / 255.0
        elif mesh.visual.kind == 'vertex':
            logging.info("GLTF/C/V")
            albedo = vertex_weighted(mesh.visual.vertex_colors, i0, i1, i2, w, u, v) / 255.0
        elif mesh.visual.kind == 'face':
            logging.info("GLTF/C/F")
            albedo = mesh.visual.face_colors[i] / 255.0
        else:
            assert False, mesh.visual.kind
    elif isinstance(mesh.visual, TextureVisuals):
        # Handle empty uv
        if mesh.visual.uv is None:
            logging.info("uv is missing")
            mat = mesh.visual.material
            if isinstance(mat, SimpleMaterial):
                mat = mat.to_pbr()
            if not isinstance(mat, PBRMaterial):
                raise ValueError("Do not know how to deal with material type {} when uv is missing".format(type(mat)))
            assert mat.baseColorTexture is None and mat.normalTexture is None and mat.metallicRoughnessTexture is None
            mesh.visual.uv = numpy.zeros([len(mesh.vertices), 2])
        
        uvsample = vertex_weighted(mesh.visual.uv, i0, i1, i2, w, u, v)
        if isinstance(mesh.visual.material, MultiMaterial):
            logging.info("GLTF/T/M")
            for m in range(len(mesh.visual.material.materials)):
                mask = mesh.visual.face_materials[i] == m
                a, n, m, r = handle_single_material(mesh.visual.material.materials[m], uvsample[mask])
                albedo[mask] = a
                normal_tangent[mask] = n
                metal[mask] = m
                roughness[mask] = r
        else:
            return handle_single_material(mesh.visual.material, uvsample)
    else:
        raise ValueError("Unknown visual type {}".format(type(mesh.visual)))
    return albedo, normal_tangent, metal, roughness