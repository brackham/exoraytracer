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
