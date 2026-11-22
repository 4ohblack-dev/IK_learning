"""Build a SO100 URDF whose zero matches image.png (upright upper arm).

Only coordinate conventions are changed: native SO100 joint geometry is kept.
World +X is forward, +Y is left. Native SO100 joint signs are preserved;
the base rotation maps its forward -Y direction into world +X.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'SO-ARM100/Simulation/SO100/so100.urdf'
DEST = ROOT / 'models/so100_image_zero.urdf'
MODEL_ID = 'so100_image_zero_v1'
TCP = (0.0098493, -0.106375, 0.0)  # midpoint of Fixed_Jaw.stl distal edge, metres


def build():
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set('name', MODEL_ID)
    joints = {j.get('name'): j for j in root.findall('joint')}
    # SO100's native zero already matches image.png: upper arm vertical,
    # lower arm and wrist horizontal. Preserve its joint origins and limits.
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
    ET.SubElement(tip, 'origin', {'xyz': ' '.join(map(str, TCP)), 'rpy': '0 0 0'})
    DEST.parent.mkdir(exist_ok=True)
    ET.indent(tree, space='  ')
    tree.write(DEST, encoding='utf-8', xml_declaration=True)
    (DEST.parent / 'so100_model_source.json').write_text(json.dumps({
        'model_id': MODEL_ID,
        'model_id': MODEL_ID,
        'source_url': 'https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO100/so100.urdf',
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'native_q_at_image_zero': {name: 0.0 for name in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')},
        'native_pan_sign': 1,
        'servo_position_directions': {'1': -1, '2': -1, '3': -1, '4': 1, '5': 1},
        'world_rotation_z_rad': np.pi / 2,
        'tcp_gripper_xyz_m': TCP,
        'reference': 'image.png; ideal upright upper arm, horizontal forearm and wrist',
        'physical_calibration_verified': False,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    build()
