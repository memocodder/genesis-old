# %%

import torch
import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


# %%

def xyz_to_quat(euler_xyz, rpy=True, degrees=True):
    if isinstance(euler_xyz, torch.Tensor):
        if degrees:
            euler_xyz *= torch.pi / 180.0
        roll, pitch, yaw = euler_xyz.unbind(-1)
        cosr = (roll * 0.5).cos()
        sinr = (roll * 0.5).sin()
        cosp = (pitch * 0.5).cos()
        sinp = (pitch * 0.5).sin()
        cosy = (yaw * 0.5).cos()
        siny = (yaw * 0.5).sin()
        sign = 1.0 if rpy else -1.0
        qw = cosr * cosp * cosy + sign * sinr * sinp * siny
        qx = sinr * cosp * cosy - sign * cosr * sinp * siny
        qy = cosr * sinp * cosy + sign * sinr * cosp * siny
        qz = cosr * cosp * siny - sign * sinr * sinp * cosy
        return torch.stack([qw, qx, qy, qz], dim=-1)
    elif isinstance(euler_xyz, np.ndarray):
        if rpy:
            rot = Rotation.from_euler("xyz", euler_xyz, degrees=degrees)
        else:
            rot = Rotation.from_euler("zyx", euler_xyz[::-1], degrees=degrees)
        return rot.as_quat(scalar_first=True)
    else:
        # gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(euler_xyz)=}")
        raise f'the input must be either torch.Tensor or np.ndarray. got: {type(euler_xyz)=}'



gs.init(theme="light", backend=gs.cpu)


# %%


num_env = 2
sep = True

scene = gs.Scene(
    show_viewer=False
)


# %%


scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
# %%


agent = scene.add_entity(gs.morphs.Sphere(pos=(-1., 0.0, 1.0)))
goal1 = scene.add_entity(gs.morphs.Sphere(pos=(1, 1.0, 1.0)))
goal2 = scene.add_entity(gs.morphs.Sphere(pos=(1, -2.0, 1.0)))



# %%


cam = scene.add_camera(
    res=(128, 128), pos=(0.0, 0.0, 2.5), lookat=(0, 0, 0.5), fov=50, GUI=False
)
# %%
device = torch.device("cpu")
# %%

scene.build(n_envs=num_env) # , compile_kernels=False
# scene.build()

# %%

# new_agent_pos = torch.rand((num_env, 3), dtype=torch.float32)*-3
new_agent_pos = torch.zeros((num_env, 3), dtype=torch.float32)
new_agent_pos[:] = torch.tensor([1, 1, 1])
agent.set_pos(new_agent_pos)

# %%

# new_goal1_pos = torch.zeros((num_env, 3), dtype=torch.float32)
# new_goal1_pos[:] = torch.tensor([1, 1, 1])
new_goal1_pos = torch.tensor([[46, 1, 1], [1, 1, 0.7]])
goal1.set_pos(new_goal1_pos)

new_goal2_pos = torch.tensor([[1, 1, 6], [1, 1, 4]])
goal2.set_pos(new_goal2_pos)
goal1pos = goal1.get_pos()
print(goal1pos)
print(agent.idx)
print(goal1.idx)
print(goal2.idx)
# %%


for i in range(1):
    scene.step()
print('---------------')

pos = agent.get_pos()
print(f'agent {agent.idx} pos', pos)
goal1pos = goal1.get_pos()
print(f'goal {goal1.idx} pos', goal1pos)
goal2pos = goal2.get_pos()
print(f'goal {goal2.idx} pos', goal2pos)
conts = agent.get_contacts()

print('---------------')
print('geom_a', conts['geom_a'])
print('geom_b', conts['geom_b'])
print('valid_mask', conts['valid_mask'])
print('---------------')
# print(conts)


# %%

new_goal1_pos = torch.rand((num_env, 3), dtype=torch.float32)*torch.tensor([2, 2, 0])
new_goal2_pos = torch.rand((num_env, 3), dtype=torch.float32)*torch.tensor([-2, -2, 0])
print(new_agent_pos)
goal1.set_pos(new_goal1_pos)
goal2.set_pos(new_goal2_pos)

# %%

scene._sim.rigid_solver._entities[0]
# %%

new_agent_quat = torch.zeros((num_env, 4), dtype=torch.float32)
new_agent_quat[:] = torch.tensor(xyz_to_quat(np.array([0,0,10])))

# %%
agent.set_quat(new_agent_quat)

grid_values = scene.step()
print(grid_values.shape)
for i in range(num_env):
    img = grid_values[i, :, :, 0].T
    img = img.cpu().numpy()
    # print(f'img.shape: {img.shape}')

    plt.imshow(img)
    plt.show()


# entities = scene._sim._entities
# print(entities[1].get_pos())
# print(type(entities[1].get_pos()))
# print(entities[1].get_quat())

# print(len(entities))
# links = entities[1]._links
# # print(len(links))
# vgeoms = links[0]._vgeoms
# print(len(vgeoms))
# print(vgeoms[0])
# vmesh = vgeoms[0]._vmesh
# print(vmesh)
# print('---')
# print('trimesh:\n', vmesh.trimesh)
# faces = vmesh.trimesh.faces
# faces_exemple = faces[:10]
# print(f'faces_exemple: {faces_exemple}')
# vertices = vmesh.trimesh.vertices
# vertices_exemple = vertices[faces_exemple]
# print(f'vertices_exemple: {vertices_exemple}')

# print('faces.shape', faces.shape)
# print('vertices.shape', vertices.shape)
# print(len(faces))
# print(type(faces)) 
# # print('faces:\n', faces)
# geoms_state = scene.rigid_solver.geoms_state
# print(type(geoms_state.aabb_min))
# print(geoms_state.aabb_min)
# print(geoms_state.aabb_max)

# self.geoms_state[i_g, i_b].aabb_min = lower
# self.geoms_state[i_g, i_b].aabb_max = upper

# %%


for i in range(30):
    rgb, d, _, _ = cam.render(depth=True, rgb=False)
    print(d.shape)

# %%


render_scene = scene.visualizer._context._scene
scene_meshes = render_scene.meshes
for mesh in scene_meshes:
    for p in mesh.primitives:
        print(p)
# print(meshes)
# render_scene.get_pose(render_scene.main_camera_node)[:3, 3]
# render_scene.main_camera_node._visualizer._scene
# print(goal.get_pos().shape)

# scene.viewer.follow_entity(entity)

# follower_camera = scene.add_camera(res=(640,480),
#                                         pos=(0.0, 2.0, 0.5),
#                                         lookat=(0.0, 0.0, 0.5),
#                                         fov=40,
#                                         GUI=True)
 
# follower_camera.follow_entity(entity, fixed_axis=(None, None, 0.5), smoothing=0.5, fix_orientation=True)


# %%

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower
def add_z_dim(tensor):
    rows = tensor.shape[0]
    zeros = torch.ones((rows, 1), dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, zeros), dim=1)

goal_xy = gs_rand_float(-0.2, 0.2, (num_envs, 2), device)
goal_xyz = add_z_dim(goal_xy)
goal.set_pos(goal_xyz, zero_velocity=True, envs_idx=[0, 1, 2])
# %%

for i in range(10):
    scene.step()

# %%

def check_collision(agent, threshold_idx) -> torch.Tensor:

    agent_geom_idx = agent.idx
    contact_info = agent.get_contacts()

    valid_mask = contact_info['valid_mask']
    geom_a = contact_info['geom_a']
    geom_b = contact_info['geom_b']

    # 1. Находим индекс партнера по контакту
    partner_geom_idx = torch.where(geom_a == agent_geom_idx, geom_b, geom_a)

    # 2. Формируем маску целевых столкновений: контакт валиден И индекс партнера > порога
    is_target_collision = valid_mask & (partner_geom_idx > threshold_idx)

    # 3. Проверяем, было ли хотя бы одно такое столкновение для каждого env
    result_per_env = torch.any(is_target_collision, dim=1)

    return result_per_env

# --- Пример использования (Тест 4) ---
print("\n--- Краткий Тест (без squeeze) ---")
conts4 = {
 'geom_a': torch.tensor([[1, 1], [1, 5], [1, 1]], dtype=torch.int32),
 'geom_b': torch.tensor([[3, 4], [1, 1], [6, 2]], dtype=torch.int32),
 'valid_mask': torch.tensor([[True, True], [True, False], [True, True]])
}
agent_idx4 = 1
threshold4 = 3
result4_no_squeeze = check_collision(conts4, agent_idx4, threshold4)
print(f"Входные контакты:\n{conts4}")
print(f"Агент ID: {agent_idx4}, Порог ID: {threshold4}")
print(f"Результат (без squeeze): {result4_no_squeeze}") # Ожидаем: tensor([ True, False, True])

# --- Пример использования (Тест 1, где C=1) ---
print("\n--- Краткий Тест (без squeeze, C=1) ---")
conts1 = {
 'geom_a': torch.tensor([[1], [1]], dtype=torch.int32), # Shape [2, 1]
 'geom_b': torch.tensor([[2], [2]], dtype=torch.int32), # Shape [2, 1]
 'valid_mask': torch.tensor([[ True], [False]])          # Shape [2, 1]
}
agent_idx1 = 1
threshold1 = 1
result1_no_squeeze = check_collision(conts1, agent_idx1, threshold1)
print(f"Входные контакты:\n{conts1}")
print(f"Агент ID: {agent_idx1}, Порог ID: {threshold1}")
print(f"Результат (без squeeze): {result1_no_squeeze}") # Ожидаем: tensor([ True, False])

