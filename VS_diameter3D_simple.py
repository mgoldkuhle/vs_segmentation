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
import os,sys
import pandas as pd
colors = vtk.vtkNamedColors()

def rendering3d(file_name,normal,center,point1,point2):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(file_name)
    reader.Update()

    ## decision boundary
    dec = vtk.vtkPlaneSource()
    #dec.SetNormal(normal)
    dec.SetOrigin(center)
    dec.SetPoint1(point1)
    dec.SetPoint2(point2)
    dec.Update()

    #sphere.Update()
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
    #p1 = dec_actor.GetBounds()
    #p2 = intra_actor.GetBounds()
    #p3 = extra_actor.GetBounds()
    ren.SetActiveCamera(a_camera)
    ren.ResetCamera()
    ren.SetBackground(colors.GetColor3d('White'))
    #a_camera.Dolly(1.5)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(640, 480)
    renWin.SetWindowName('Mask')
    ren.ResetCameraClippingRange()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()

def angle(vector1,vector2):
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return math.degrees(angle)

if __name__=='__main__':
    ROOT_PATH = 'F:\\07_impactstudy\\nifti_data_totest\\t2'
    SEG_PATH = 'F:\\07_impactstudy\\seg_mask\\01_t2'
    #mask_path = 'F:\\01_data\\07_impactdata\\seg_data\\01_t2\\20040064_0.nii.gz'

    #img_path = 'F:\\01_data\\07_impactdata\\nifti_data\\t2\\20040064_0_t2.nii.gz'
    SAVE_PATH = 'F:\\07_impactstudy\\results\\t2_new'

    patients = os.listdir(ROOT_PATH)
    data = []
    #patients = ['20040041_0_0000.nii.gz']
    for patient in patients:
        print(patient)
        patient_data = [patient]
        tmp = patient.split('_')
        patient_id = tmp[0] + '_' + tmp[1]
        img_path = os.path.join(ROOT_PATH,patient)
        mask_path = os.path.join(SEG_PATH,patient_id+'.nii.gz')
        mask = sitk.ReadImage(mask_path)
        mask_np = sitk.GetArrayFromImage(mask)
        img = sitk.ReadImage(img_path)
        spacing = img.GetSpacing()
        img_np = sitk.GetArrayFromImage(img)
        img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np)) * 255
        '''
        target_size = np.round(
            np.array(img_np.shape) * np.array((spacing[2], spacing[1], spacing[0])) / np.array((1.0, 1.0, 1.0))).astype(
            np.int)
    
        resized_mask = transform.resize(mask_np, target_size, order=0, preserve_range=True, anti_aliasing=False)
        resized_img = transform.resize(img_np, target_size, order=0, preserve_range=True, anti_aliasing=False)
        '''
        #time_start = time.time()
        #img_s = cv2.cvtColor((img_np[i]).astype('uint8'), cv2.COLOR_GRAY2RGB)                    ## for visualization
        indexes_intra = np.where(mask_np==1)
        indexes_extra = np.where(mask_np==2)
        num_intra = len(indexes_intra[0])
        num_extra = len(indexes_extra[0])
        if num_intra * num_extra == 0:
            print('both intra- and extra- tumor are required')
            patient_data.append(0)
            patient_data.append(0)
            patient_data.append(0)
            data.append(patient_data)
            continue
        x_ori = np.zeros((num_extra + num_intra,3))
        y_ori = np.zeros((num_extra + num_intra,))
        for j in range(num_intra):
            x_ori[j][0] = indexes_intra[2][j]
            x_ori[j][1] = indexes_intra[1][j]
            x_ori[j][2] = indexes_intra[0][j]
            y_ori[j] = 0
        for j in range(num_extra):
            x_ori[num_intra + j][0] = indexes_extra[2][j]
            x_ori[num_intra + j][1] = indexes_extra[1][j]
            x_ori[num_intra + j][2] = indexes_extra[0][j]
            y_ori[num_intra+j] = 1
        x = x_ori
        y = y_ori
        boundingrect = [np.min(x[:,0]),np.min(x[:,1]),np.min(x[:,2]),np.max(x[:,0])-np.min(x[:,0]), np.max(x[:,1])-np.min(x[:,1]), np.max(x[:,2]) - np.min(x[:,2])]
        clf = svm.SVC(kernel="linear")
        clf.fit(x, y)
        ## W1 * x1 + W2 * x2 + W3 * x3 +I = 0
        W1 = clf.coef_[0][0]
        W2 = clf.coef_[0][1]
        W3 = clf.coef_[0][2]
        I = clf.intercept_[0]

        z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]

        #tmp = np.linspace(250, 350, 30)
        #i, j = np.meshgrid(tmp, tmp)

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot3D(x[y == 0, 0], x[y == 0, 1], x[y == 0, 2], 'ob')
        #ax.plot3D(x[y == 1, 0], x[y == 1, 1], x[y == 1, 2], 'sr')
        #ax.plot_surface(i, j, z(i, j))
        #ax.view_init(30, 60)
        #plt.show()
        normal = (W1,W2,W3)
        #center = (boundingrect[0],boundingrect[1],(-W1 * boundingrect[0]-W2 * boundingrect[1]-I)/W3)

        center = (boundingrect[0],boundingrect[1],z(boundingrect[0],boundingrect[1]),0)
        point1 = (boundingrect[0] + boundingrect[3],boundingrect[1],(-W1 * (boundingrect[0] + boundingrect[3])-W2 * boundingrect[1]-I)/W3,0)
        #point1 = (x[0][0],x[0][1],x[0][2])
        #point2 = (boundingrect[0],boundingrect[1] + boundingrect[4],(-W1 * boundingrect[0]-W2 * (boundingrect[1] + boundingrect[4])-I)/W3)
        point2 = (boundingrect[0], boundingrect[1] + boundingrect[4],z (boundingrect[0], boundingrect[1] + boundingrect[4]),0)
        '''
        size = mask.GetSize()
        origin = mask.GetOrigin()
        spacing = mask.GetSpacing()
        direction = mask.GetDirection()
        img_size = np.array(size)
        ## affine matrix
    
        A = np.array((1,0,0,0,1,0,0,0,1)).reshape(3,3)
        A = A * np.array(spacing)
        S = np.array((0,0,0)).reshape((3, 1))
        A = np.concatenate((A, S), axis=1)
        center = np.matmul(A,center).squeeze()
        point1 = np.matmul(A,point1).squeeze()
        point2 = np.matmul(A, point2).squeeze()
    
        rendering3d(mask_path,normal,center.tolist(),point1.tolist(),point2.tolist())
        '''
        max_diameter = 0
        parallel_diameters = []
        perpendicular_diameters = []
        for i in range(mask_np.shape[0]):
            mask_s = mask_np[i]
            img_s = cv2.cvtColor((img_np[i]).astype('uint8'), cv2.COLOR_GRAY2RGB)
            a = -W1/W2
            b = (-I-i*W3)/W2
            point_1 = (50, round(50 * a + b))
            point_2 = (200, round(200 * a + b))
            #img_s = cv2.arrowedLine(img_s, point_1, point_2, (0, 0, 255), 1)

            ## find the bounding box of intrameatal tumor
            grayimage = ((mask_s == 2) * 255).astype('uint8')  ## we only care about intrameatal tumor
            edged = cv2.Canny(grayimage, 30, 200)
            contours, hierarchy = cv2.findContours(edged,
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                continue
            point_num = 0
            index_c = 0
            for j in range(len(contours)):
                if contours[j].shape[0] > point_num:                ##use the largest contour
                    point_num = contours[j].shape[0]
                    index_c = j

            boundRect = cv2.boundingRect(contours[index_c])
            x0 = int(boundRect[0])
            x1 = int(boundRect[0] + boundRect[2])
            y0 = int(boundRect[1])
            y1 = int(boundRect[1] + boundRect[3])
            #img_s = cv2.rectangle(img_s, (x0, y0), (x1, y1), (0, 255, 0), 1)
            img_s = cv2.drawContours(img_s, contours[index_c], -1, (0, 255, 0), 1)

            ##parallel direction
            x_points = np.array((x0, x0, x1, x1))
            y_points = np.array((y0, y1, y0, y1))
            b_range = y_points - a * x_points
            b_min = np.min(b_range)
            b_max = np.max(b_range)
            b_list = np.arange(b_min, b_max, 1).tolist()
            diameters = []
            bs = []
            pointa_list = []
            pointb_list = []
            for nb in b_list:
                points = []
                # for nx in xs:
                if y0 <= a * x0 + nb <= y1:
                    xmin = round(x0) - 1
                elif a * x0 + nb < y0:
                    xmin = round((y0 - nb) / a) - 1
                elif a * x0 + nb > y1:
                    xmin = round((y1 - nb) /a ) - 1

                if y0 <= a * x1 + nb <= y1:
                    xmax = round(x1) + 1
                elif a * x1 + nb < y0:
                    xmax = round((y0 - nb)/a) + 1
                elif a * x1 + nb > y1:
                    xmax = round((y1 - nb)/a) + 1
                #img_s = cv2.circle(img_s,(int(xmin),int(a * xmin + nb)),1, (255, 255, 0), 1)
                #img_s = cv2.circle(img_s, (int(xmax), int(a * xmax + nb)), 1, (255, 255, 0), 1)

                for nx in range(xmin, xmax):
                    curpoint = (round(nx), round(a * nx + nb))
                    dis = cv2.pointPolygonTest(contours[index_c], curpoint, True)
                    if abs(dis) < 1.0:
                        points.append(curpoint)
                if (len(points) > 1):
                    diameter = math.hypot((points[0][0] - points[-1][0]) * spacing[0],
                                          (points[0][1] - points[-1][1]) * spacing[1])
                    diameters.append(diameter)
                    bs.append(nb)
                    pointa_list.append(points[0])
                    pointb_list.append(points[-1])

            if len(diameters) > 0:
                diameters = np.array(diameters)
                max_indexes = np.argwhere(diameters == np.max(diameters))
                b_index = np.array(b_list)[max_indexes]
                b_index = np.abs(b_index - (b_max + b_min) / 2)
                nearest_diameter = np.argmin(b_index)
                max_parallel_diameter = diameters[max_indexes[nearest_diameter][0]]
                point_a = pointa_list[max_indexes[nearest_diameter][0]]
                point_b = pointb_list[max_indexes[nearest_diameter][0]]
                img_s = cv2.line(img_s, point_a, point_b, (255, 255, 0), 1)
            else:
                max_parallel_diameter = 0
                print("no diameter")
    ## perpendicular direction
            a_hat = -1/a
            b_range = y_points - a_hat * x_points
            b_min = np.min(b_range)
            b_max = np.max(b_range)
            b_list = np.arange(b_min, b_max, 1).tolist()
            diameters = []
            bs = []
            pointa_list = []
            pointb_list = []
            for nb in b_list:
                points = []
                if y0 <= a_hat * x0 + nb <= y1:
                    xmin = round(x0) - 1
                elif a_hat * x0 + nb < y0:
                    xmin = round((y0 - nb)/a_hat) - 1
                elif a_hat * x0 + nb > y1:
                    xmin = round((y1 - nb)/a_hat) - 1

                if y0 <= a_hat * x1 + nb <= y1:
                    xmax = round(x1) + 1
                elif a_hat * x1 + nb < y0:
                    xmax = round((y0 - nb) / a_hat) + 1
                elif a_hat * x1 + nb > y1:
                    xmax = round((y1 - nb) / a_hat) + 1
                points = []
                # for nx in xs:

                for nx in range(xmin, xmax):
                    curpoint = (round(nx), round(a_hat * nx + nb))
                    dis = cv2.pointPolygonTest(contours[index_c], curpoint, True)
                    if abs(dis) < 1.0:
                        points.append(curpoint)
                # print(points)
                if (len(points) > 1):
                    diameter = math.hypot((points[0][0] - points[-1][0]) * spacing[0],
                                          (points[0][1] - points[-1][1]) * spacing[1])
                    diameters.append(diameter)
                    bs.append(nb)
                    pointa_list.append(points[0])
                    pointb_list.append(points[-1])

            if len(diameters) > 0:
                diameters = np.array(diameters)
                max_indexes = np.argwhere(diameters == np.max(diameters))
                b_index = np.array(b_list)[max_indexes]
                b_index = np.abs(b_index - (b_max + b_min) / 2)
                nearest_diameter = np.argmin(b_index)
                max_perpendicular_diameter = diameters[max_indexes[nearest_diameter][0]]
                point_a = pointa_list[max_indexes[nearest_diameter][0]]
                point_b = pointb_list[max_indexes[nearest_diameter][0]]
                img_s = cv2.line(img_s, point_a, point_b, (0, 255, 255), 1)
            else:
                max_perpendicular_diameter = 0
                print("no diameter")
            #time_end = time.time()
            #print("parallel diameter", max_parallel_diameter)
            #print("perpendicular diameter", max_perpendicular_diameter)
            #cv2.imshow('Contours', img_s)
            #cv2.waitKey(0)
            save_dir = os.path.join(SAVE_PATH,patient_id)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            result_dir = os.path.join(save_dir,patient_id+str(i)+'_'+str(format(max_parallel_diameter,'.4f'))+'_'+str(format(max_perpendicular_diameter,'.4f'))+'.jpg')
            cv2.imwrite(result_dir,img_s)
            parallel_diameters.append(max_parallel_diameter)
            perpendicular_diameters.append(max_perpendicular_diameter)
            #if max_parallel_diameter >max_diameter:
            #    max_diameter = max_parallel_diameter
            #if max_perpendicular_diameter > max_diameter:
            #    max_diameter = max_perpendicular_diameter
        parallel_diameters = np.array(parallel_diameters)
        perpendicular_diameters = np.array(perpendicular_diameters)
        max_parallel_diameter = np.max(parallel_diameters)
        max_perpendicular_diameter_indepent = np.max(perpendicular_diameters)
        max_perpendicular_diameter = perpendicular_diameters[np.argmax(parallel_diameters)]
        patient_data.append(max_parallel_diameter)
        patient_data.append(max_perpendicular_diameter)
        patient_data.append(max_perpendicular_diameter_indepent)
        data.append(patient_data)
        print('max diameter for this patient', max_diameter)
    df = pd.DataFrame(data, columns=['patient', 'parallel_diameter','perpendiculardiameter','perpendicular_diameter_independent'])
    df.to_excel(os.path.join(SAVE_PATH,'results.xlsx'), index=False)
