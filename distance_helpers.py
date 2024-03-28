import numpy as np
# import timeit
from scipy.spatial.distance import pdist, squareform  # squareform only needed to get max diameter points
import vtk
# from vtk.util.vtkImageImportFromArray import *
import math

colors = vtk.vtkNamedColors()

def rendering3d(file_name, normal, center, point1, point2):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(file_name)
    reader.Update()

    ## decision boundary
    dec = vtk.vtkPlaneSource()
    # dec.SetNormal(normal)
    dec.SetOrigin(center)
    dec.SetPoint1(point1)
    dec.SetPoint2(point2)
    dec.Update()

    # sphere.Update()
    dec_mapper = vtk.vtkPolyDataMapper()
    dec_mapper.SetInputConnection(dec.GetOutputPort())

    dec_actor = vtk.vtkActor()
    dec_actor.SetMapper(dec_mapper)
    dec_actor.GetProperty().SetOpacity(0.3)
    dec_actor.GetProperty().SetColor(0, 0.5, 0.5)
    ## extrameatal tumor
    extra_sur = vtk.vtkDiscreteMarchingCubes()
    extra_sur.SetInputConnection(reader.GetOutputPort())
    extra_sur.SetValue(0, 1)
    extra_sur.Update()

    extra_mapper = vtk.vtkPolyDataMapper()
    extra_mapper.SetInputConnection(extra_sur.GetOutputPort())
    extra_mapper.ScalarVisibilityOff()

    extra_actor = vtk.vtkActor()
    extra_actor.SetMapper(extra_mapper)
    extra_actor.GetProperty().SetOpacity(0.3)
    extra_actor.GetProperty().SetColor(1, 0, 0)

    ## intrameatal tumor
    intra_sur = vtk.vtkDiscreteMarchingCubes()
    intra_sur.SetInputConnection(reader.GetOutputPort())
    intra_sur.SetValue(0, 2)

    intra_mapper = vtk.vtkPolyDataMapper()
    intra_mapper.SetInputConnection(intra_sur.GetOutputPort())
    intra_mapper.ScalarVisibilityOff()

    intra_actor = vtk.vtkActor()
    intra_actor.SetMapper(intra_mapper)
    intra_actor.GetProperty().SetOpacity(0.3)
    intra_actor.GetProperty().SetColor(0, 1, 0)

    a_camera = vtk.vtkCamera()
    a_camera.SetViewUp(0, 0, -1)
    a_camera.SetPosition(0, -1, 0)
    a_camera.SetFocalPoint(0, 0, 0)
    a_camera.ComputeViewPlaneNormal()
    a_camera.Azimuth(30.0)
    a_camera.Elevation(30.0)

    ren = vtk.vtkRenderer()
    ren.AddActor(extra_actor)
    ren.AddActor(intra_actor)
    ren.AddActor(dec_actor)
    # p1 = dec_actor.GetBounds()
    # p2 = intra_actor.GetBounds()
    # p3 = extra_actor.GetBounds()
    ren.SetActiveCamera(a_camera)
    ren.ResetCamera()
    ren.SetBackground(colors.GetColor3d('White'))
    # a_camera.Dolly(1.5)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(640, 480)
    renWin.SetWindowName('Mask')
    ren.ResetCameraClippingRange()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()


def angle(vector1, vector2):
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return math.degrees(angle)

# # dummy data for testing
# x_extra = np.random.rand(2000, 3)

# ## diameters any direction (exhaustive search). not recommended, very slow. just included for comparison.
# def exhaustive_search(x_extra):
#     max_distance = 0
#     for i in range(len(x_extra)):
#         for j in range(i + 1, len(x_extra)):
#             # Calculate the Euclidean distance between points[i] and points[j]
#             p1 = x_extra[i]
#             p2 = x_extra[j]
#             distance = np.linalg.norm(x_extra[i] - x_extra[j])
#             # If this distance is greater than the current max_distance, update max_distance
#             if distance > max_distance:
#                 max_distance = distance
#                 max_points = (x_extra[i], x_extra[j])
#     print(max_distance, max_points)
#     return max_distance, max_points


## just calculating the max distance, not returning points
def max_pdist(x_extra):
    distances = pdist(x_extra)
    max_distance = np.max(distances)
    return max_distance


# ## returning max distance and points, slightly slower but still way faster than exhaustive search
# def pdist_withpoints(x_extra):
#     distances = pdist(x_extra)
#     square_distances = squareform(distances)
#     max_distance = np.max(distances)
#     i, j = np.unravel_index(square_distances.argmax(), square_distances.shape)
#     max_points = (x_extra[i], x_extra[j])
#     print(max_distance, max_points)
#     return max_distance, max_points


# time the functions
# time_exhaustive = timeit.timeit('exhaustive_search(x_extra)', globals=globals(), number=10)

# time_pdist = timeit.timeit('pdist_func(x_extra)', globals=globals(), number=10)  # roughly 700x faster than exhaustive search

# time_pdist_indeces = timeit.timeit('pdist_withpoints(x_extra)', globals=globals(), number=10)  # roughly 200x faster than exhaustive search
