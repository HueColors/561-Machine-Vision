import numpy as np

np.set_printoptions(suppress=True, precision=6)

#points from images
points_img1 = np.array([
    [336, 449],  #hand
    [436, 496],  #left eye
    [502, 490],  #right eye
    [380, 567],  #left blush
    [536, 566],  #right blush
    [458, 592],  #left mouth
    [494, 590],  #right mouth
    [618, 699]   #foot
], dtype=np.float32)

points_img2 = np.array([
    [325, 469],  #hand
    [414, 519],  #left eye
    [479, 514],  #right eye
    [361, 586],  #left blush
    [509, 587],  #right blush
    [431, 612],  #left mouth
    [468, 613],  #right mouth
    [602, 719]   #foot
], dtype=np.float32)

#normalize points
def norm_pts(points):
    #find mean for centriod
    mean = np.mean(points, axis=0)
    
    #avg dist of points to mean
    avg_dist = np.mean(np.sqrt(np.sum((points - mean) ** 2, axis=1)))
    
    #scale based on avg dist
    scale = np.sqrt(2) / avg_dist

    #normalization matrix
    norm_mat = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])

    #normalize points
    normalized_points = np.dot(norm_mat, np.vstack((points.T, np.ones((1, points.shape[0])))))

    #return points and norm matrix
    return normalized_points[:2].T, norm_mat

#calculate matrix A
def A_matrix(pts_img1, pts_img2):
    A = []
    for i in range(len(pts_img1)):
        #image 1 coordinates
        u1, v1 = pts_img1[i]
        #image 2 coordinates
        u2, v2 = pts_img2[i]
        
        #append new row for point pair
        A.append([u1 * u2, u1 * v2, u1, v1 * u2, v1 * v2, v1, u2, v2, 1])

    #convert list into array
    A = np.array(A)
    return A


#de-normalize E
def E_denorm(norm_E, T1, T2):
    E = np.dot(T2.T, np.dot(norm_E, T1))
    return E


#calculate essential E
def E_matrix(A):
    #SVD on A
    U, S, Vt = np.linalg.svd(A)
    
    #reshape Vt from A into E
    E = Vt[-1].reshape(3, 3)
    
    #SVD on E
    Ue, Se, Vte = np.linalg.svd(E)
    
    #Set smallest value to 0 for rank 2
    Se[2] = 0
    
    #recombine into new E with rank 2
    E = np.dot(Ue, np.dot(np.diag(Se), Vte))
    
    return E


#calculate rotation and translation matrices from E
def calc_R_t(E):
    #SVD on E
    U, _, Vt = np.linalg.svd(E)
    
    #W matrix for R=UWVt / R=UWtVt
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    #calculate R1 and R2
    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W.T, Vt))

    #check valid rotations
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    #get translation vector from U 
    t = U[:, 2]

    return R1, R2, t

#normalize translation vector
def t_norm(t):
    norm_t = t / np.linalg.norm(t)
    return norm_t


#-----------------------------------------------------------------------------

#normalize points
norm_pts_img1, norm_mat_img1 = norm_pts(points_img1)
norm_pts_img2, norm_mat_img2 = norm_pts(points_img2)

#make A matrix
A = A_matrix(norm_pts_img1, norm_pts_img2)

#calculate essential E matrix
norm_E = E_matrix(A)

#denormalize E
E = E_denorm(norm_E, norm_mat_img1, norm_mat_img2)
print("Essential Matrix E:\n", E, "\n")

#calculate rotation and translation matrices
R1, R2, t = calc_R_t(E)

#normalize translation vector
norm_t = t_norm(t)

#results
print("Rotation Matrix R1:\n", R1, "\n")
print("Rotation Matrix R2:\n", R2, "\n")
print("Translation Matrix t:\n", norm_t, "\n")