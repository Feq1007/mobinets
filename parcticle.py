import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats

np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)
import cv2

def drawLines(img, points, r, g, b):
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))

# 以center为中心画十字
def drawCross(img, center, r, g, b):
    d = 5
    t = 2
    LINE_AA = cv2.LINE_AA if cv2.__version__[0] == '3' else cv2.LINE_AA
    color = (r, g, b)
    ctrx = center[0,0]
    ctry = center[0,1]
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)
    
#鼠标事件回调函数，event表示鼠标事件，（x,y)表示鼠标坐标，flags是CV_EVENT_FLAG的组合，null说明没有来自外界的参数
def mouseCallback(event, x, y, flags,null):
    global center
    global trajectory#轨道
    global previous_x
    global previous_y
    global zs
    global robot_pos
    
    center=np.array([[x,y]])
    trajectory=np.vstack((trajectory,np.array([x,y])))#线段的起点为当前坐标
    #noise=sensorSigma * np.random.randn(1,2) + sensorMu
    
    if previous_x >0:#鼠标之前的x坐标
        #求移动方向，弧度制（y向上为正，向下为负，x向上为锐角，向下为钝角）
        heading=np.arctan2(np.array([y-previous_y]), np.array([previous_x-x ]))
        if heading>0:
            heading=-(heading-np.pi)
        else:
            heading=-(np.pi+heading)
            
        #求距离：axis=1表示按行向量处理，求多个行向量的范数(平方和开根，欧式距离)
        distance=np.linalg.norm(np.array([[previous_x,previous_y]])-np.array([[x,y]]) ,axis=1)
        
        std=np.array([2,4])#？？？？？？？？？？？
        u=np.array([heading,distance])#方向和距离
        
        #通过当前粒子及运动方向和距离预测粒子新坐标
        predict(particles, u, std, dt=1.)
        
        #地标与鼠标中心的距离，带误差
        zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))
        
        update(particles, weights, z=zs, R=50, landmarks=landmarks)#通过权重修改粒子概率
        indexes = systematic_resample(weights)#重采样索引
        resample_from_index(particles, weights, indexes)#根据采样索引重采样
        
        #通过带全平均值计算机器人位置
        robot_pos = np.average(particles[:], weights=weights, axis=0)
        print(robot_pos)
        
        
    #更新坐标
    previous_x=x
    previous_y=y

#sensorMu=0
#sensorSigma=3

#传感器标准差
sensor_std_err=5


#步骤一：创建随机初始化粒子，N表示个数
def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))# 形状为 Nx2 的矩阵
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)#随机初始化，在x[0]~x[1]之间，横坐标
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)#随机初始化，在y[0]~y[1]之间，纵坐标
    return particles

#步骤二：预测
def predict(particles, u, std, dt=1.):
    """
    *** 功能：仅改变粒子坐标
    *** particles：粒子矩阵    
    *** u[0]=heading,u[1]=distance
    *** std:[2,4]
    *** dt:距离误差？
    """
    N = len(particles)
    # np.random.randn(N)
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    particles[:, 0] += np.cos(u[0]) * dist
    particles[:, 1] += np.sin(u[0]) * dist
   
#步骤三：更新
def update(particles, weights, z, R, landmarks):
    """
    *** particles:粒子矩阵
    *** weigths:粒子权重
    *** z:地标到鼠标中心的距离，带误差
    *** R:50
    *** landmarks:地标
    """
    b, loc, scale = 1.5, 0, 1
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        distance=np.power((particles[:,0] - landmark[0])**2 +(particles[:,1] - landmark[1])**2,0.5)#每个粒子到地标的距离 : N*1
        weights *= scipy.stats.norm(distance, R).pdf(z[i])#均值是distance，方差是R=50的正态分布,.pdf(z[i])表示该分布下取z[i]的概率
    weights += 1.e-300 # avoid round-off to zero
    weights /= sum(weights)#归一化

#权重平方的和：用于归一化
def neff(weights):
    return 1. / np.sum(np.square(weights))

#重采样
def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N #重采样位置:逐渐增加

    indexes = np.zeros(N, 'i')#int型数组
    cumulative_sum = np.cumsum(weights)#第n项是前n项和
    i, j = 0, 0
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:#如果位置i小于权重前j项和
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes
    
#评估
def estimate(particles, weights):
    pos = particles[:, 0:1]
    mean = np.average(pos, weights=weights, axis=0)#均值
    var = np.average((pos - mean)**2, weights=weights, axis=0)#方差
    return mean, var

#通过索引来重采样，权值归一化
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)

    
x_range=np.array([0,800])
y_range=np.array([0,600])

#Number of partciles
N=400

#地标的坐标
landmarks=np.array([ [144,73], [410,13], [336,175], [718,159], [178,484], [665,464]  ])
#地标的个数
NL = len(landmarks)
#随机初始化N粒子坐标矩阵：其x为随机横坐标，y为随机纵坐标
particles=create_uniform_particles(x_range, y_range, N)

#粒子权值初始化为1
weights = np.array([1.0]*N)

#显示窗口大小
WIDTH=800
HEIGHT=600
WINDOW_NAME="Particle Filter"

# Create a black image, a window and bind the function to window
img = np.zeros((HEIGHT,WIDTH,3), np.uint8)#各个坐标点及其像素值（RGB）
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME,mouseCallback)#关联鼠标事件与值

# 鼠标中心
center=np.array([[-10,-10]])

trajectory=np.zeros(shape=(0,2))#轨道
robot_pos=np.zeros(shape=(1,2))#机器人位置
previous_x=-1#上一个坐标x
previous_y=-1#上一个坐标y
DELAY_MSEC=50#延迟毫秒数

while(1):
    cv2.imshow(WINDOW_NAME,img)
    img = np.zeros((HEIGHT,WIDTH,3), np.uint8)
    drawLines(img, trajectory,   0,   255, 0)
    drawCross(img, center, r=255, g=0, b=0)
    
    #landmarks
    for landmark in landmarks:
        cv2.circle(img,tuple(landmark),10,(255,0,0),-1)
    
    #draw_particles:画粒子
    for particle in particles:
        cv2.circle(img,tuple((int(particle[0]),int(particle[1]))),1,(255,255,255),-1)
        
    #刷新图像，按下esc退出
    if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
        break
    
    #显示提示信息
    cv2.circle(img,(10,10),10,(255,0,0),-1)
    cv2.circle(img,(10,30),3,(255,255,255),-1)
    drawLines(img, np.array([[10,55],[25,55]]), 0, 255, 0)
    cv2.putText(img,"Landmarks",(30,20),1,1.0,(255,0,0))
    cv2.putText(img,"Particles",(30,40),1,1.0,(255,255,255))
    cv2.putText(img,"Robot Trajectory(Ground truth)",(30,60),1,1.0,(0,255,0))     
    cv2.putText(img,"robot position : ",(30,80),1,1.0,(0,255,0))
#    cv2.putText(img,str(robot_pos[0],(30,100),1,1.0,(255,255,0))
    
cv2.destroyAllWindows()