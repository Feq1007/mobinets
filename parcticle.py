import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats

np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)
import cv2

def drawLines(img, points, r, g, b):
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))

# ��centerΪ���Ļ�ʮ��
def drawCross(img, center, r, g, b):
    d = 5
    t = 2
    LINE_AA = cv2.LINE_AA if cv2.__version__[0] == '3' else cv2.LINE_AA
    color = (r, g, b)
    ctrx = center[0,0]
    ctry = center[0,1]
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)
    
#����¼��ص�������event��ʾ����¼�����x,y)��ʾ������꣬flags��CV_EVENT_FLAG����ϣ�null˵��û���������Ĳ���
def mouseCallback(event, x, y, flags,null):
    global center
    global trajectory#���
    global previous_x
    global previous_y
    global zs
    global robot_pos
    
    center=np.array([[x,y]])
    trajectory=np.vstack((trajectory,np.array([x,y])))#�߶ε����Ϊ��ǰ����
    #noise=sensorSigma * np.random.randn(1,2) + sensorMu
    
    if previous_x >0:#���֮ǰ��x����
        #���ƶ����򣬻����ƣ�y����Ϊ��������Ϊ����x����Ϊ��ǣ�����Ϊ�۽ǣ�
        heading=np.arctan2(np.array([y-previous_y]), np.array([previous_x-x ]))
        if heading>0:
            heading=-(heading-np.pi)
        else:
            heading=-(np.pi+heading)
            
        #����룺axis=1��ʾ�����������������������ķ���(ƽ���Ϳ�����ŷʽ����)
        distance=np.linalg.norm(np.array([[previous_x,previous_y]])-np.array([[x,y]]) ,axis=1)
        
        std=np.array([2,4])#����������������������
        u=np.array([heading,distance])#����;���
        
        #ͨ����ǰ���Ӽ��˶�����;���Ԥ������������
        predict(particles, u, std, dt=1.)
        
        #�ر���������ĵľ��룬�����
        zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))
        
        update(particles, weights, z=zs, R=50, landmarks=landmarks)#ͨ��Ȩ���޸����Ӹ���
        indexes = systematic_resample(weights)#�ز�������
        resample_from_index(particles, weights, indexes)#���ݲ��������ز���
        
        #ͨ����ȫƽ��ֵ���������λ��
        robot_pos = np.average(particles[:], weights=weights, axis=0)
        print(robot_pos)
        
        
    #��������
    previous_x=x
    previous_y=y

#sensorMu=0
#sensorSigma=3

#��������׼��
sensor_std_err=5


#����һ�����������ʼ�����ӣ�N��ʾ����
def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))# ��״Ϊ Nx2 �ľ���
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)#�����ʼ������x[0]~x[1]֮�䣬������
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)#�����ʼ������y[0]~y[1]֮�䣬������
    return particles

#�������Ԥ��
def predict(particles, u, std, dt=1.):
    """
    *** ���ܣ����ı���������
    *** particles�����Ӿ���    
    *** u[0]=heading,u[1]=distance
    *** std:[2,4]
    *** dt:������
    """
    N = len(particles)
    # np.random.randn(N)
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    particles[:, 0] += np.cos(u[0]) * dist
    particles[:, 1] += np.sin(u[0]) * dist
   
#������������
def update(particles, weights, z, R, landmarks):
    """
    *** particles:���Ӿ���
    *** weigths:����Ȩ��
    *** z:�ر굽������ĵľ��룬�����
    *** R:50
    *** landmarks:�ر�
    """
    b, loc, scale = 1.5, 0, 1
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        distance=np.power((particles[:,0] - landmark[0])**2 +(particles[:,1] - landmark[1])**2,0.5)#ÿ�����ӵ��ر�ľ��� : N*1
        weights *= scipy.stats.norm(distance, R).pdf(z[i])#��ֵ��distance��������R=50����̬�ֲ�,.pdf(z[i])��ʾ�÷ֲ���ȡz[i]�ĸ���
    weights += 1.e-300 # avoid round-off to zero
    weights /= sum(weights)#��һ��

#Ȩ��ƽ���ĺͣ����ڹ�һ��
def neff(weights):
    return 1. / np.sum(np.square(weights))

#�ز���
def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N #�ز���λ��:������

    indexes = np.zeros(N, 'i')#int������
    cumulative_sum = np.cumsum(weights)#��n����ǰn���
    i, j = 0, 0
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:#���λ��iС��Ȩ��ǰj���
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes
    
#����
def estimate(particles, weights):
    pos = particles[:, 0:1]
    mean = np.average(pos, weights=weights, axis=0)#��ֵ
    var = np.average((pos - mean)**2, weights=weights, axis=0)#����
    return mean, var

#ͨ���������ز�����Ȩֵ��һ��
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)

    
x_range=np.array([0,800])
y_range=np.array([0,600])

#Number of partciles
N=400

#�ر������
landmarks=np.array([ [144,73], [410,13], [336,175], [718,159], [178,484], [665,464]  ])
#�ر�ĸ���
NL = len(landmarks)
#�����ʼ��N�������������xΪ��������꣬yΪ���������
particles=create_uniform_particles(x_range, y_range, N)

#����Ȩֵ��ʼ��Ϊ1
weights = np.array([1.0]*N)

#��ʾ���ڴ�С
WIDTH=800
HEIGHT=600
WINDOW_NAME="Particle Filter"

# Create a black image, a window and bind the function to window
img = np.zeros((HEIGHT,WIDTH,3), np.uint8)#��������㼰������ֵ��RGB��
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME,mouseCallback)#��������¼���ֵ

# �������
center=np.array([[-10,-10]])

trajectory=np.zeros(shape=(0,2))#���
robot_pos=np.zeros(shape=(1,2))#������λ��
previous_x=-1#��һ������x
previous_y=-1#��һ������y
DELAY_MSEC=50#�ӳٺ�����

while(1):
    cv2.imshow(WINDOW_NAME,img)
    img = np.zeros((HEIGHT,WIDTH,3), np.uint8)
    drawLines(img, trajectory,   0,   255, 0)
    drawCross(img, center, r=255, g=0, b=0)
    
    #landmarks
    for landmark in landmarks:
        cv2.circle(img,tuple(landmark),10,(255,0,0),-1)
    
    #draw_particles:������
    for particle in particles:
        cv2.circle(img,tuple((int(particle[0]),int(particle[1]))),1,(255,255,255),-1)
        
    #ˢ��ͼ�񣬰���esc�˳�
    if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
        break
    
    #��ʾ��ʾ��Ϣ
    cv2.circle(img,(10,10),10,(255,0,0),-1)
    cv2.circle(img,(10,30),3,(255,255,255),-1)
    drawLines(img, np.array([[10,55],[25,55]]), 0, 255, 0)
    cv2.putText(img,"Landmarks",(30,20),1,1.0,(255,0,0))
    cv2.putText(img,"Particles",(30,40),1,1.0,(255,255,255))
    cv2.putText(img,"Robot Trajectory(Ground truth)",(30,60),1,1.0,(0,255,0))     
    cv2.putText(img,"robot position : ",(30,80),1,1.0,(0,255,0))
#    cv2.putText(img,str(robot_pos[0],(30,100),1,1.0,(255,255,0))
    
cv2.destroyAllWindows()