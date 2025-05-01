import pyvista as pv
import numpy as np
import glob

i = 2
mesh = pv.read(f"{i}.vtk")
print("Number of points:", mesh.n_points)

mesh = pv.UnstructuredGrid(f"{i}.vtk")  # or pv.read("mesh.vtk")

names = ["smooth_near_null.npz", "c_points.npz"]
basis_fine = glob.glob("basis_fine_*.npz")
basis_coarse = glob.glob("basis_coarse_*.npz")
names = names + basis_fine + basis_coarse

for func in names:
    func_data = np.load(func)
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
    show_edges=True,
    scalars=names[0]
)

plotter.add_scalar_bar(names[current_index])

def toggle_scalar(fwd):
    global current_index
    plotter.clear()

    if fwd:
        current_index = (current_index + 1) % len(names)
    else:
        current_index = (current_index - 1)
        if current_index < 0:
            current_index = len(names) - 1

    name = names[current_index]

    if current_index == 0:
        warped_mesh = mesh.warp_by_scalar(
            scalars=name,
            factor=1.0 / np.max(np.abs(mesh.point_data[name])),       
            normal=(0, 0, 1),
            inplace=False   
        )

        plotter.add_mesh(
            warped_mesh,
            show_edges=True,
            scalars=name,
        )
    else:
        plotter.add_mesh(
            mesh,
            scalars=name,
            style="points",
            render_points_as_spheres=True,
            point_size=20,
            opacity="linear",
            #cmap="inferno"
        )
        plotter.add_mesh(
            mesh,
            style="wireframe",
        )

    plotter.add_scalar_bar(title=name)

# Register the callback: press 't' to toggle
plotter.add_key_event("t", lambda: toggle_scalar(True))
plotter.add_key_event("T", lambda: toggle_scalar(False))

plotter.show()
