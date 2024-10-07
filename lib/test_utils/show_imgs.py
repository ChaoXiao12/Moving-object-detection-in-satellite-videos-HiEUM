import cv2
import matplotlib.pyplot as plt
import numpy as np

CONFIDENCE_thres = 0.3
COLORS = [(255, 0, 0)]

FONT = cv2.FONT_HERSHEY_SIMPLEX

def cv2_demo(frame, detections):
    det = []
    for i in range(detections.shape[0]):
        if detections[i, 4] >= CONFIDENCE_thres:
            pt = detections[i, :]
            cv2.rectangle(frame,(int(pt[0])-4, int(pt[1])-4),(int(pt[2])+4, int(pt[3])+4),COLORS[0], 2)
            cv2.putText(frame, str(pt[4]), (int(pt[0]), int(pt[1])), FONT, 1, (0, 255, 0), 1)
            det.append([int(pt[0]), int(pt[1]),int(pt[2]), int(pt[3]),detections[i, 4]])
    return frame, det

def view_cloud(coord, save_flag=0, fig_save_name = '',resolution=[1024, 1024]):
    points_temp = coord.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_temp[:, 2], points_temp[:, 3], points_temp[:, 1], s=0.50)
    ax.set_xlabel('x', fontproperties="Times New Roman", fontsize=14)
    ax.set_ylabel('y', fontproperties="Times New Roman", fontsize=14)
    ax.set_zlabel('T', fontproperties="Times New Roman", fontsize=14)
    ax.set_xlim(0, resolution[0])
    ax.set_ylim(0, resolution[1])
    if save_flag:
        plt.savefig(fig_save_name, dpi=500)
    else:
        plt.show()

def view_dets(dets, conf_th=0.3, save_flag=0, fig_save_name = '',resolution=[1024, 1024]):
    dets_all = []
    for i in range(len(dets)):
        det = dets[i].detach().cpu().numpy()
        det_select = det[det[:,4]>conf_th, :]
        det_p = np.zeros([det_select.shape[0], 3])
        ct = (det_select[:, 0:2]+det_select[:, 2:4])/2
        det_p[:,0] = ct[:, 1]
        det_p[:, 1] = ct[:, 0]
        det_p[:, 2] = i
        dets_all.append(det_p)
    points_temp = np.concatenate(dets_all, 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_temp[:, 0], points_temp[:, 1], points_temp[:, 2], s=0.50)
    ax.set_xlabel('x', fontproperties="Times New Roman", fontsize=14)
    ax.set_ylabel('y', fontproperties="Times New Roman", fontsize=14)
    ax.set_zlabel('T', fontproperties="Times New Roman", fontsize=14)
    ax.set_xlim(0, resolution[0])
    ax.set_ylim(0, resolution[1])
    if save_flag:
        plt.savefig(fig_save_name, dpi=500)
    else:
        plt.show()