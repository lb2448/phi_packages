import numpy as np
import scipy.stats as st
import numba as nb
from scipy.spatial.transform import Rotation as R
import time


def rodrigues(to_vec, from_vec=None):
    if from_vec is None:
        from_vec = [0, 0, 1]
    n0 = np.array(from_vec)
    n1 = to_vec/np.linalg.norm(to_vec)
    theta = np.arccos(np.dot(n0, n1))
    rot_axis = np.cross(n0, n1)
    mrp = rot_axis/np.linalg.norm(rot_axis) * np.tan(theta / 4)
    rot = R.from_mrp(mrp)
    return rot


def plane_ransac(data, outlier_p=.2, ransac_success_p=.9, n=8, max_distance=.03):
    np.random.seed(0)
    data_n = data.shape[0]
    d = np.ceil(data_n*(1-outlier_p))
    w = st.hypergeom(data_n, d, n).pmf(n)
    k = np.ceil(np.log(1-ransac_success_p)/np.log(1-w)).astype(int)
    best_model = None
    inliers_mask = None
    best_mean_distance = np.inf
    for i in range(k):
            permutation = np.random.permutation(data.shape[0])
            rand_sample = data[permutation[:n]]
            maybe_model = fit_plane(rand_sample)
            inlier_candidates = data[permutation[n:]]
            candidate_dist = plane_dist(maybe_model, inlier_candidates)
            accepetd_candidate = candidate_dist <= max_distance
            if accepetd_candidate.sum() > d:
                inliers_mask = np.hstack((permutation[:n], permutation[n:][accepetd_candidate]))
                inliers = data[inliers_mask]
                good_model = fit_plane(inliers)
                distances = plane_dist(good_model, inliers)
                mean_distance = distances.mean()
                if best_mean_distance > mean_distance:
                    best_model = good_model
                    best_mean_distance = mean_distance
                    inliers_mask = inliers_mask
    return best_model, inliers_mask
    

def ransac_interations(data_n, model_candidates, outlier_p, ransac_success_p):
    d = np.ceil(data_n*(1-outlier_p))
    w = st.hypergeom(data_n, d, model_candidates).pmf(model_candidates)
    k = np.ceil(np.log(1-ransac_success_p)/np.log(1-w)).astype(int) 
    return k


#@nb.jit(nopython=True, nogil=True, cache=True)
def fit_plane(data):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    t = np.zeros((x.size, 3))
    t[:, 0] = x
    t[:, 1] = y
    t[:, 2] = np.ones((x.size,))
    solve = np.linalg.lstsq(t, -np.ones((x.size,)) * z)
    a, b, d = solve[0]
    c = 1
    residuals = solve[1]
    return (a, b, c, d), residuals


#@nb.jit(nopython=True, nogil=True, cache=True)
def plane_dist(model, data):
    a, b, c, d = model
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    d = np.abs(x*a + y*b + z*c + d)/np.sqrt(a**2 + b**2 + c**2)
    return d


#@nb.jit(nopython=True, nogil=True, cache=True)
def fit_sphere(data):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    x_mu, y_mu, z_mu = x.mean(), y.mean(), z.mean()
    std = data.std()
    std = std if std != 0 else 1
    x_norm, y_norm, z_norm = (x - x_mu) / std, (y - y_mu) / std, (z - z_mu) / std
    t = np.zeros((x.size, 4))
    t[:, 0] = x_norm
    t[:, 1] = y_norm 
    t[:, 2] = z_norm
    t[:, 3] = np.ones((x.size,))
    solve = np.linalg.lstsq(t, -x_norm**2 - y_norm**2 - z_norm**2)
    c1, c2, c3, c4 = solve[0]
    residuals = solve[1]
    x0_norm, y0_norm, z0_norm = -c1/2, -c2/2, -c3/2
    x0, y0, z0 = x0_norm * std + x_mu, y0_norm * std + y_mu, z0_norm * std + z_mu
    r = np.sqrt(x0_norm ** 2 + y0_norm ** 2 + z0_norm ** 2 - c4) * std
    center = np.array([x0, y0, z0])
    return (center, r), residuals


#@nb.jit(nopython=True, nogil=True, cache=True)
def fit_circle(data, normal):
    if not np.allclose(normal, [0, 0, 1], atol=1e-3):
        rot = rodrigues([0, 0, 1], normal[:3])
        data = rot.apply(data)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    x_mu, y_mu = x.mean(), y.mean()
    std = data.std()
    std = std if std != 0 else 1
    x_norm, y_norm = (x - x_mu) / std, (y - y_mu) / std
    t = np.zeros((x.size, 3))
    t[:, 0] = x_norm 
    t[:, 1] = y_norm
    t[:, 2] = np.ones((x.size,))
    solve = np.linalg.lstsq(t, -x_norm**2 - y_norm**2, rcond=None)
    c1, c2, c3, = solve[0]
    residuals = solve[1]
    x0_norm, y0_norm = -c1 / 2, -c2 / 2, 
    x0, y0 = x0_norm * std + x_mu, y0_norm * std + y_mu
    r = np.sqrt(x0_norm ** 2 + y0_norm ** 2 + - c3) * std
    center = np.array([x0, y0, z.mean()])
    center = rot.inv().apply(center)
    return center, r


def fit_annulus(data, normal):
    center, rad = fit_circle(data, normal)
    _, _, rads = cartesian2spherical((center, rad), data)
    '''r1 = np.percentile(rads, 1)
    r2 = np.percentile(rads, 99)'''
    r1 = rads.min()
    r2 = rads.max()
    return (center, rad), normal, (r1, r2), 


def annulus_dist(model, data):
    (center, rad), normal, (r1, r2) = model
    centered = data - center
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2) / rad - 1
    #print(rad, r1, r2, d.min(), d.max())
    d[np.logical_and((d > r1), (d < r2))] = 0
    d[d < r1] = r1 - d[d < r1]
    d[d > r2] = d[d > r2] - r2
    return d


#@nb.jit(nopython=True, nogil=True, cache=True)
def sphere_dist(model, data):
    center, r = model
    centered = data - center
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2) / r - 1
    return d


def cartesian2spherical(sphere, data):
    center, rad = sphere
    x0, y0, z0 = center
    x, y, z = data[:, 0] - x0, data[:, 1] - y0, data[:, 2] - z0
    theta = np.arctan(y / x) + np.pi / 2
    phi = np.arctan2(np.sqrt(x ** 2 + y ** 2), z) + np.pi
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return theta, phi, r


def cartesian2polar(circle, data):
    center, rad = circle
    x0, y0, z0 = center
    x, y = data[:, 0] - x0, data[:, 1] - y0
    theta = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    return theta, r


def make_plane_points(shape, normal, zero_point, size=0.005):
    normal = np.array(normal)
    x_lim = shape[0] / 2
    y_lim = shape[1] / 2
    size_x = int(shape[0] // size)
    size_y = int(shape[1] // size)
    space_x = np.linspace(-x_lim, x_lim, num=size_x) 
    space_x = np.tile(space_x[None].T, (1, size_y))
    space_y = np.linspace(-y_lim, y_lim, num=size_y) 
    space_y = np.tile(space_y, ( size_x, 1))
    space_z = np.zeros((size_x, size_y))
    plane = np.dstack((space_x, space_y, space_z,))
    if not np.allclose(normal, [0, 0, 1]):
        n0 = normal/np.linalg.norm(normal)
        n1 = np.array([0, 0, 1])
        theta = np.arccos(np.dot(n0,n1))
        rot_axis = np.cross(n0,n1)
        mrp = rot_axis/np.linalg.norm(rot_axis) * np.tan(theta / 4)
        r = R.from_mrp(mrp)
        plane = r.apply(plane.reshape(-1, 3))
    plane += zero_point
    return plane.reshape(-1, 3)


def make_sphere_points(rad, zero_point, size=0.005):
    circ = int(np.round(2*np.pi*rad/size))
    theta = np.linspace(0, np.pi, num=circ//2)
    phi = np.linspace(0, 2*np.pi, num=circ)
    sphere_x = np.sin(theta)[:, None] * np.cos(phi)[None] * rad
    sphere_y = np.sin(theta)[:, None] * np.sin(phi)[None] * rad
    sphere_z = np.tile((np.cos(theta) * rad)[None].T, (1, circ))
    sphere_points = np.dstack((sphere_x, sphere_y, sphere_z)).reshape(-1, 3)
    sphere_points += zero_point
    return sphere_points.reshape(-1, 3)


'''fit_sphere(np.ones((5, 3)))
fit_plane(np.ones((5, 3)))
plane_dist((0, 0, 1, 0), np.ones((5, 3)))
sphere_dist((np.array([0, 0, 0]), 1), np.ones((5, 3)))
t = time.time()
#fit_circle(np.ones((5000, 3)), [1,0,0])
fit_sphere(np.ones((5000, 3)))
print(time.time() - t)'''