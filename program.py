from spsim.gemmi import rotate_structure, structure_to_cif
from scipy.spatial.transform import Rotation as R
import gemmi

structure = gemmi.read_structure('test_data/trajectory/6vxx.pdb')
rotation = R.from_euler(seq='ZYZ', angles=(0, 180, 0), degrees=True)
rotated_structure = rotate_structure(structure, rotation, center=None)
structure_to_cif(rotated_structure, 'output.cif')

