import numpy as np


class Ray:
    """A ray"""
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.u = normalize(direction-origin)


class Star:
    """A star"""
    def __init__(self, center, radius,
                 axis=np.array([0., 1., 0.]),
                 res=100):
        self.center = center
        self.radius = radius
        self.axis = normalize(axis)
        self.inc = 90.
        self.meridian = 0.
        self.res = res
        self.shape = (res, res)
        self.u1 = 0.
        self.u2 = 0.
        self.x = np.linspace(-radius, radius, res)
        self.y = np.linspace(-radius, radius, res)
        self.P = np.zeros((res, res, 3))
        self.N = np.zeros((res, res, 3))
        self.mu = np.zeros((res, res))
        self.r1 = np.zeros((res, res))
        self.theta1 = np.zeros((res, res))
        self.phi1 = np.zeros((res, res))
        self.r = np.zeros((res, res))
        self.theta = np.zeros((res, res))
        self.phi = np.zeros((res, res))
        self.lat = np.zeros((res, res))
        self.lon = np.zeros((res, res))
        self.I = np.zeros((res, res))
        self.spots = np.array([])

    def add(self, spots, overwrite=False):
        """
        Add a feature
        """
        if overwrite:
            self.spots = spots
        else:
            self.spots = np.append(self.spots, spots)

    def calc_I(self):
        """
        Calculate the intensity map
        """
        self.I = np.ones((self.res, self.res))
        self.I = np.ma.masked_where(np.isnan(self.r), self.I)
        for spot in self.spots:
            dist = haversine(self.lat, self.lon, spot.lat, spot.lon)
            spotted = np.ma.masked_where(dist<=spot.radius, dist)
            self.I[spotted.mask] = spot.contrast

    def limb_darken(self):
        """
        Apply the quadratic limb darkening law
        """
        self.I = self.I - self.u1*(self.I - self.mu) - self.u2*(self.I - self.mu)**2

    def rotate(self, angle):
        """
        Rotate about axis by a given angle in degrees
        """
        self.meridian = (self.meridian + angle) % 360.
        if self.meridian > 180.:
            self.meridian -= 360.
        for j, i in np.ndindex(star.shape):
            self.P[j,i,:] = rotate_basis(self.P[j,i,:], gamma=np.radians(-angle))
        self.r = np.sqrt(np.sum(self.P**2, axis=2))
        self.theta = np.arccos(self.P[:,:,2]/self.r)
        self.phi = np.arctan2(self.P[:,:,0], self.P[:,:,1])
        self.lat = np.degrees(self.theta-np.pi/2.)
        self.lon = np.degrees(self.phi)
        self.calc_I()
        self.limb_darken()

    def set_meridian(self, new_meridian):
        """
        Set the meridian to a specified longitude in degrees
        """
        angle = new_meridian-self.meridian
        self.rotate(angle)

    def set_inclination(self, new_inclination):
        """
        Set the inclination to a specified degree value
        """
        angle = new_inclination - self.inc
        for j, i in np.ndindex(star.shape):
            self.P[j,i,:] = rotate_basis(self.P[j,i,:], alpha=np.radians(-angle))
        self.r = np.sqrt(np.sum(self.P**2, axis=2))
        self.theta = np.arccos(self.P[:,:,2]/self.r)
        self.phi = np.arctan2(self.P[:,:,0], self.P[:,:,1])
        self.lat = np.degrees(self.theta-np.pi/2.)
        self.lon = np.degrees(self.phi)
        self.calc_I()
        self.limb_darken()
        self.inc = new_inclination


class Spot:
    """A spot"""
    def __init__(self, lat, lon, radius, contrast):
        self.lat = np.float(lat)
        self.lon = np.float(lon)
        self.radius = np.float(radius)
        self.contrast = np.float(contrast)


class Scene:
    """A scene"""
    def __init__(self, objects, res=100):
        self.objects = objects
        self.res = res


def normalize(x):
    x /= np.linalg.norm(x)
    return x


def intersect(Ray, Star):
    """
    Intersection of a ray and a sphere

    See: https://en.wikipedia.org/wiki/Line-sphere_intersection
    """
    a = np.dot(Ray.u, Ray.u)
    origin_center = Ray.origin - Star.center
    b = 2*np.dot(Ray.u, origin_center)
    c = np.dot(origin_center, origin_center) - Star.radius**2
    discriminant = b**2 - 4*a*c
    if discriminant >= 0:
        t1 = (-b + np.sqrt(discriminant))/2.
        t2 = (-b - np.sqrt(discriminant))/2.
        t1, t2 = np.min([t1, t2]), np.max([t1, t2])
        if t1 >= 0:
            return t1
        else:
            return t2
    return np.inf


def angle_between(v0, v1):
    """
    Determine the angle between two vectors
    """
    v0 = normalize(v0)
    v1 = normalize(v1)
    theta = np.arccos(np.dot(v0, v1))
    return theta


def get_Euler_angles(u, theta):
    """
    Get the Euler angles for a specified rotation about an axis.
    Adapted from `starry` jupyter notebook.
    """
    ux, uy, uz = u[0], u[1], u[2]
    # Numerical tolerance
    tol = 1e-16
    if theta == 0:
        theta = tol
    if ux == 0 and uy == 0:
        ux = tol
        uy = tol

    # Elements of the transformation matrix
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    RA01 = ux * uy * (1 - costheta) - uz * sintheta
    RA02 = ux * uz * (1 - costheta) + uy * sintheta
    RA11 = costheta + uy * uy * (1 - costheta)
    RA12 = uy * uz * (1 - costheta) - ux * sintheta
    RA20 = uz * ux * (1 - costheta) - uy * sintheta
    RA21 = uz * uy * (1 - costheta) + ux * sintheta
    RA22 = costheta + uz * uz * (1 - costheta)

    # Determine the Euler angles
    if ((RA22 < -1 + tol) and (RA22 > -1 - tol)):
        cosbeta = -1
        sinbeta = 0
        cosgamma = RA11
        singamma = RA01
        cosalpha = 1
        sinalpha = 0
    elif ((RA22 < 1 + tol) and (RA22 > 1 - tol)):
        cosbeta = 1
        sinbeta = 0
        cosgamma = RA11
        singamma = -RA01
        cosalpha = 1
        sinalpha = 0
    else:
        cosbeta = RA22
        sinbeta = np.sqrt(1 - cosbeta ** 2)
        norm1 = np.sqrt(RA20 * RA20 + RA21 * RA21)
        norm2 = np.sqrt(RA02 * RA02 + RA12 * RA12)
        cosgamma = -RA20 / norm1
        singamma = RA21 / norm1
        cosalpha = RA02 / norm2
        sinalpha = RA12 / norm2
    alpha = np.arctan2(sinalpha, cosalpha)
    beta = np.arctan2(sinbeta, cosbeta)
    gamma = np.arctan2(singamma, cosgamma)

    return alpha, beta, gamma


def rotate_basis(P, alpha=0., beta=0., gamma=0.):
    """
    Rotates coordinate basis for point P by specified angles.
    """
    Rx = np.array([[1., 0., 0.],
                  [0., np.cos(alpha), -np.sin(alpha)],
                  [0., np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0., np.sin(beta)],
                  [0., 1., 0.],
                  [-np.sin(beta), 0., np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0.],
                  [np.sin(gamma), np.cos(gamma), 0.],
                  [0., 0., 1]])
    R = Rz @ Ry @ Rx
    return R @ P


def rotate_axis_angle(P, u, theta):
    """
    Rotates coordinate around axis u by angle theta
    """
    u = normalize(u)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    ux, uy, uz = u[0], u[1], u[2]
    R = np.array([[costheta + ux**2*(1.-costheta),
                   ux*uy*(1.-costheta) - uz*sintheta,
                   ux*uz*(1.-costheta) + uy*sintheta],
                  [uy*ux*(1.-costheta) + uz*sintheta,
                   costheta + uy**2*(1.-costheta),
                   uy*uz*(1.-costheta) - ux*sintheta],
                  [uz*ux*(1.-costheta) - uy*sintheta,
                   uz*uy*(1.-costheta) + ux*sintheta,
                   costheta + uz**2*(1.-costheta)]])
    return R @ P


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    (specified in decimal degrees)

    Returns distance in degress
    """
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    dist = 2*np.arcsin(np.sqrt(a))
    return np.degrees(dist)
