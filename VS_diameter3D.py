## draw diameter from segment
import SimpleITK as sitk
import numpy as np
from numpy.linalg import norm
import math
from skimage import data, util
import matplotlib.pyplot as plt
import cv2
from sklearn import svm
from sklearn.datasets import make_blobs
import time
import vtk
from vtk.util.vtkImageImportFromArray import *
from vtk.util import numpy_support
from skimage import transform
from PIL import Image
from scipy import ndimage
import os, sys
import pandas as pd
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import pdist
from distance_helpers import pdist_func

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


if __name__ == '__main__':
    ROOT_PATH = '//vf-DataSafe/DataSafe$/div2/radi/Brughoek_predict_1255/01_followup_cleanedup/followup_part3'
    # SEG_PATH = 'F:\\07_impactstudy\\seg_mask\\01_t2'
    # mask_path = 'F:\\01_data\\07_impactdata\\seg_data\\01_t2\\20040064_0.nii.gz'

    # img_path = 'F:\\01_data\\07_impactdata\\nifti_data\\t2\\20040064_0_t2.nii.gz'
    SAVE_PATH = 'C:/Users/mjgoldkuhle/ownCloud/LUMC/data/selected_features'

    patients = os.listdir(ROOT_PATH)
    data = []
    for patient in patients:
        print(patient)
        
        tmp = patient.split('_')
        patient_id = tmp[1]
        DATE_PATH = os.path.join(ROOT_PATH, patient)
        dates = os.listdir(DATE_PATH)
        for date in dates:
            print(date)
            patient_data = [patient]
            patient_data.append(date)
            try:
                img_path = os.path.join(ROOT_PATH, patient, date, patient_id + '_' + date + '_t1ce.nii.gz')
                mask_path = os.path.join(ROOT_PATH, patient, date, patient_id + '_' + date + '_t1ce_seg.nii.gz')
                mask = sitk.ReadImage(mask_path)
                mask_np = sitk.GetArrayFromImage(mask)
                img = sitk.ReadImage(img_path)
                spacing = img.GetSpacing()
                img_np = sitk.GetArrayFromImage(img)
                img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np)) * 255
            except:
                print('no mask or image')
                patient_data.append(0)
                patient_data.append(0)
                patient_data.append(0)
                data.append(patient_data)
                continue

            target_size = np.round(
                 np.array(img_np.shape) * np.array((spacing[2], spacing[1], spacing[0])) / np.array((1.0, 1.0, 1.0))).astype(
                 int)

            mask_np_resized = transform.resize(mask_np, target_size, order=0, preserve_range=True, anti_aliasing=False)
            img_np_resized = transform.resize(img_np, target_size, order=0, preserve_range=True, anti_aliasing=False)

            indexes_intra = np.where(mask_np_resized == 1)
            indexes_extra = np.where(mask_np_resized == 2)
            num_intra = len(indexes_intra[0])
            num_extra = len(indexes_extra[0])
            print(num_intra, num_extra)
            if num_intra * num_extra == 0 or num_extra < 2:
                print('both intra- and extra- tumor are required and at least 2 extra pixels')
                patient_data.append(0)
                patient_data.append(0)
                patient_data.append(0)
                data.append(patient_data)
                continue
            #x_ori = np.zeros((num_extra + num_intra, 3))
            x_intra = np.zeros((num_intra, 3))
            x_extra = np.zeros((num_extra,3))
            y = np.zeros((num_extra + num_intra,))
            for j in range(num_intra):
                x_intra[j][0] = indexes_intra[2][j]
                x_intra[j][1] = indexes_intra[1][j]
                x_intra[j][2] = indexes_intra[0][j]
                y[j] = 0
            for j in range(num_extra):
                x_extra[j][0] = indexes_extra[2][j]
                x_extra[j][1] = indexes_extra[1][j]
                x_extra[j][2] = indexes_extra[0][j]
                y[num_intra + j] = 1
            x = np.concatenate((x_intra,x_extra),axis=0)
            boundingrect = [np.min(x[:, 0]), np.min(x[:, 1]), np.min(x[:, 2]), np.max(x[:, 0]) - np.min(x[:, 0]),
                            np.max(x[:, 1]) - np.min(x[:, 1]), np.max(x[:, 2]) - np.min(x[:, 2])]
            clf = svm.SVC(kernel="linear")
            clf.fit(x, y)
            ## W1 * x1 + W2 * x2 + W3 * x3 +I = 0
            W1 = clf.coef_[0][0]
            W2 = clf.coef_[0][1]
            W3 = clf.coef_[0][2]
            I = clf.intercept_[0]
            z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]
            normal = (W1, W2, W3)
            normal_vector = normal / np.linalg.norm(normal)
            #projected_points = []
            distances = []
            for point in x_extra:                              ## we only care about intrameatal tumor
                distance = np.dot(point, normal_vector)
                distances.append(distance)
            #projected_points = np.array(projected_points)
            distances = np.array(distances)

            perpendicular_diameter_3d = max(distances) - min(distances)
            print('perpendicular diameter', perpendicular_diameter_3d)

            ##get the parallel plane
            tmp_vector = np.random.rand(3)
            while True:
                parallel_vector_1st = np.cross(normal_vector, tmp_vector)
                if np.linalg.norm(parallel_vector_1st) > 1e-10:  # Change the threshold as needed
                    break
            ## project points to the parallel plane
            parallel_vector_2st = np.cross(normal_vector, parallel_vector_1st)
            parallel_vector_1st = parallel_vector_1st / np.linalg.norm(parallel_vector_1st)
            parallel_vector_2st = parallel_vector_2st / np.linalg.norm(parallel_vector_2st)
            x_extra_2d = np.zeros((x_extra.shape[0], 2))
            for i,point_3d in enumerate(x_extra):
                x_extra_2d[i, 0] = np.dot(point_3d, parallel_vector_1st)
                x_extra_2d[i, 1] = np.dot(point_3d, parallel_vector_2st)

            if num_extra == 2:
                dist_matrix = distance_matrix(x_extra_2d, x_extra_2d)
            else:
                # Calculate the Convex Hull of the points
                hull = ConvexHull(x_extra_2d)

                # Get the points that make up the Convex Hull
                hull_points = x_extra_2d[hull.vertices, :]
                # Calculate the distance between all pairs of points in the Convex Hull
                dist_matrix = distance_matrix(hull_points, hull_points)
            # Find the maximum distance
            parallel_diameter_3d = np.max(dist_matrix)
            print('parallel diameter', parallel_diameter_3d)

            # ## diameters any direction(exhaustive search). not recommended, very slow.
            # max_distance = 0
            # for i in range(len(x_extra)):
            #     for j in range(i + 1, len(x_extra)):
            #         # Calculate the Euclidean distance between points[i] and points[j]
            #         p1 = x_extra[i]
            #         p2 = x_extra[j]
            #         distance = np.linalg.norm(x_extra[i] - x_extra[j])
            #         # If this distance is greater than the current max_distance, update max_distance
            #         if distance > max_distance:
            #             max_distance = distance
            #             max_points = (x_extra[i], x_extra[j])

            # refactor for higher performance and cleaner code. has yet to be tested.
            # Find the maximum distance
            max_distance = pdist_func(x_extra)
            print('max distance', max_distance)
            print('x_extra shape', x_extra.shape)

            patient_data.append(parallel_diameter_3d)
            patient_data.append(perpendicular_diameter_3d)
            patient_data.append(max_distance)
            data.append(patient_data)
    df = pd.DataFrame(data, columns=['patient', 'date', 'parallel_diameter', 'perpendiculardiameter','max_diameter'])
    df.to_csv(os.path.join(SAVE_PATH, 't1ce_3d_part3_test.csv'), index=False)
    df.to_excel(os.path.join(SAVE_PATH, 't1ce_3d_part3_test.xlsx'), index=False)
