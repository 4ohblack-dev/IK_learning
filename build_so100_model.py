"""Build the SO100 URDF in the joystick base frame and locate the fixed jaw tip.

Only coordinate conventions are changed: native SO100 joint geometry is kept.
World +X is forward, +Y is left. Native SO100 joint signs are preserved;
the base rotation maps its forward -Y direction into world +X.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from mathmetics import as_xyz, rpy_matrix

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'SO-ARM100/Simulation/SO100/so100.urdf'
DEST = ROOT / 'models/so100_image_zero.urdf'
MODEL_ID = 'so100_image_zero_v1'


def fixed_jaw_tip(root, urdf_path):
    """Return the distal edge midpoint in gripper coordinates, in metres."""
    gripper = root.find("link[@name='gripper']")
    for visual in gripper.findall("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is not None and Path(mesh.attrib['filename']).name == 'Fixed_Jaw.stl':
            break
    else:
        raise ValueError("gripper has no Fixed_Jaw.stl visual")

    stl = (urdf_path.parent / mesh.attrib['filename']).read_bytes()
    count = int.from_bytes(stl[80:84], 'little')
    if count == 0 or len(stl) != 84 + 50 * count:
        raise ValueError("expected the official binary Fixed_Jaw.stl")
    triangle = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    vertices = np.frombuffer(stl, dtype=triangle, offset=84)['vertices'].reshape(-1, 3)
    vertices = vertices.astype(float) * as_xyz(mesh.get('scale', '1 1 1'))
    # The fixed jaw extends along its mesh's local -Y and ends in a straight edge.
    edge = vertices[np.isclose(vertices[:, 1], vertices[:, 1].min(), atol=1e-7, rtol=0)]
    tip = (edge.min(axis=0) + edge.max(axis=0)) / 2
    origin = visual.find('origin')
    if origin is not None:
        tip = as_xyz(origin.get('xyz', '0 0 0')) + rpy_matrix(as_xyz(origin.get('rpy', '0 0 0'))) @ tip
    return np.round(tip, 7), hashlib.sha256(stl).hexdigest()


def build():
    tree = ET.parse(SOURCE, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    root = tree.getroot()
    tcp, tcp_mesh_sha256 = fixed_jaw_tip(root, SOURCE)
    root.set('name', MODEL_ID)
    joints = {j.get('name'): j for j in root.findall('joint')}
    # Preserve native joint angles. The upright calibration posture uses nonzero angles.
    pan = joints['shoulder_pan']
    pan.find('parent').set('link', 'base_link')
    pan.find('origin').set('xyz', '0.0452 0 0.0165')
    pan.find('origin').set('rpy', f'1.57079 0 {np.pi/2:.15g}')
    pan.find('axis').set('xyz', '0 1 0')
    base = root.find("link[@name='base']")
    base.set('name', 'base_link')
    # Base meshes/inertia must rotate with the coordinate change too.
    for element in list(base):
        origin = element.find('origin')
        if origin is None:
            origin = ET.SubElement(element, 'origin')
        origin.set('xyz', '0 0 0')
        origin.set('rpy', f'0 0 {np.pi/2:.15g}')
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', '../SO-ARM100/Simulation/SO100/' + mesh.attrib['filename'])
    ET.SubElement(root, 'link', {'name': 'tool_tip'})
    tip = ET.SubElement(root, 'joint', {'name': 'tool_tip_fixed', 'type': 'fixed'})
    ET.SubElement(tip, 'parent', {'link': 'gripper'})
    ET.SubElement(tip, 'child', {'link': 'tool_tip'})
    tcp_xyz = ' '.join(np.format_float_positional(value + 0.0, trim='-') for value in tcp)
    ET.SubElement(tip, 'origin', {'xyz': tcp_xyz, 'rpy': '0 0 0'})
    DEST.parent.mkdir(exist_ok=True)
    ET.indent(tree, space='  ')
    tree.write(DEST, encoding='utf-8', xml_declaration=True)
    (DEST.parent / 'so100_model_source.json').write_text(json.dumps({
        'model_id': MODEL_ID,
        'source_url': 'https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO100/so100.urdf',
        'source_sha256': hashlib.sha256(SOURCE.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'native_pan_sign': 1,
        'servo_position_directions': {'1': -1, '2': -1, '3': -1, '4': 1, '5': 1},
        'world_rotation_z_rad': np.pi / 2,
        'tcp_gripper_xyz_m': (tcp + 0.0).tolist(),
        'tcp_mesh_sha256': tcp_mesh_sha256,
        'reference': 'Official SO100 joint angles with the base rotated to +X forward and +Y left',
        'physical_calibration_verified': False,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    build()
