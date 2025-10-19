from partnext import PartNeXtDataset, PartNeXtObject

# replace this with your own data path
glb_dir = "/data/sea_disk0/wangph1/workspace/partnext/data/data"
annotation_dir = "/data/sea_disk0/wangph1/workspace/partnext/data/PartNeXt"

pn_dataset = PartNeXtDataset(glb_dir, annotation_dir)

# get the number of objects in the dataset
num_objects = pn_dataset.get_num_objects()
# get the list of object ids in the dataset
object_id_list = pn_dataset.get_object_ids()

# load an object, replace the object id
pn_object = pn_dataset.load_object("b3e33144d8224385a2036b431e1b1451")
# visualize the object (trimesh viewer)
pn_object.visualize()
# our hierarchy, which is similar to PartNet, only contains geometry in leaf nodes
pn_object_hierarchy = pn_dataset.get_hierarchy()
# object geometry, as a list of trimesh object
pn_object_geometry_list = pn_object.geometry_list
# object mesh, as a trimesh Scene
pn_object_mesh = pn_object.mesh
# object masks, dict format
pn_object_masks = pn_object.masks
# get object parts geometry (leaf node level), as a dict
pn_object_all_parts = pn_object.get_all_parts()

# visualize a part geometry (trimesh viewer)
pn_object_all_parts["5"].show()