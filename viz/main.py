import pyvista as pv
import numpy as np

mesh = pv.read("mesh.vtk")
print("Number of points:", mesh.n_points)

mesh = pv.UnstructuredGrid("mesh.vtk")  # or pv.read("mesh.vtk")

names = ["solution", "error", "residual", "interp_x_vals", "p_pt_residual"]

for i in range(10):
    names.append( f"near_null-{i}")

for i in range(50):
    names.append(f"basis-{i}")

for func in names:
    func_data = np.load(f"../{func}.npz")
    mesh.point_data[func] = func_data

current_index = 0

plotter = pv.Plotter()

warped_mesh = mesh.warp_by_scalar(
    scalars=names[current_index],
    factor=1.0 / np.max(np.abs(mesh.point_data[names[current_index]])), 
    normal=(0, 0, 1),
    inplace=False   
)

plotter.add_mesh(
    warped_mesh, 
    scalars=names[current_index], 
    show_edges=False,
    cmap="inferno"
)

plotter.add_scalar_bar(names[current_index])

def toggle_scalar():
    global current_index
    plotter.clear()

    current_index = (current_index + 1) % len(names)
    name = names[current_index]
    
    warped_mesh = mesh.warp_by_scalar(
        scalars=name,
        factor=1.0 / np.max(np.abs(mesh.point_data[name])),       
        normal=(0, 0, 1),
        inplace=False   
    )

    plotter.add_mesh(
        warped_mesh,
        scalars=name,
        show_edges=False,
        cmap="inferno"
    )

    plotter.add_scalar_bar(title=name)

# Register the callback: press 't' to toggle
plotter.add_key_event("t", toggle_scalar)

plotter.show()


