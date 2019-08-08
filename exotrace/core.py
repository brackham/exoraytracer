"""`exotrace` core functionality."""
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates.matrix_utilities import rotation_matrix

# __all__ = ['Ray', 'Star', 'Spot', 'Scene', 'intersect']


class Ray:
    """A Ray."""

    def __init__(self, origin, direction):
        """Initialize a Ray."""
        self.origin = origin
        self.direction = direction
        self.u = normalize(direction-origin)


class Body:
    """Base class for bodies in the system."""

    def __init__(self, center, radius, axis=np.array([0., 1., 0.]),
                 inc=90., meridian=0.):
        """
        Initialize a Body.

        Parameters
        ----------
        center : array-like
            The body center position.
        radius : float
            The body radius.
        axis : array-like
            The axis of rotation.
        inc : float
            The inclination in degrees.
        meridian : float
            The meridian in degrees.

        Returns
        -------
        A Body.

        """
        self.center = center
        self.radius = radius
        self.axis = normalize(axis)
        self.inc = inc
        self.meridian = meridian
        self.u1 = 0.
        self.u2 = 0.

    def calc_flux(self):
        """Calculate the flux map."""
        self.flux = np.ones((self.res, self.res))
        self.flux = np.ma.masked_where(np.isnan(self.r), self.flux)
        for spot in self.spots:
            dist = haversine(self.lat, self.lon, spot.lat, spot.lon)
            spotted = np.ma.masked_where(dist <= spot.radius, dist)
            self.flux[spotted.mask] = spot.contrast

    def limb_darken(self):
        """Apply the quadratic limb darkening law."""
        self.flux = (self.flux -
                     self.u1*(self.flux - self.mu) -
                     self.u2*(self.flux - self.mu)**2)

    def rotate(self, angle):
        """Rotate about axis by a given angle in degrees."""
        self.meridian = (self.meridian + angle) % 360.
        if self.meridian > 180.:
            self.meridian -= 360.


class Star(Body):
    """A Star."""

    def __init__(self, *args, **kwargs):
        """Initialize a Star."""
        super().__init__(*args, **kwargs)
        self.spots = np.array([])

    def add(self, spots, overwrite=False):
        """Add a feature."""
        if overwrite:
            self.spots = spots
        else:
            self.spots = np.append(self.spots, spots)


class Spot:
    """A Spot."""

    def __init__(self, lat, lon, radius, contrast):
        """Initialize a Spot."""
        self.lat = np.float(lat)
        self.lon = np.float(lon)
        self.radius = np.float(radius)
        self.contrast = np.float(contrast)


class Scene:
    """A Scene."""

    def __init__(self, bodies=np.array([]), res=100):
        """Initialize a Scene."""
        self.bodies = bodies
        self.res = res
        self.shape = (res, res)
        self.get_extent()

        self.body = get_none_array(self.shape)
        self.t = np.ones(self.shape)*np.inf
        self.P = np.ones((self.res, self.res, 3))*np.nan
        self.N = np.ones((self.res, self.res, 3))*np.nan
        self.mu = np.ones(self.shape)*np.nan
        self.r = np.ones(self.shape)*np.nan
        self.theta = np.ones(self.shape)*np.nan
        self.phi = np.ones(self.shape)*np.nan
        self.lat = np.ones(self.shape)*np.nan
        self.lon = np.ones(self.shape)*np.nan
        self.flux = np.ones(self.shape)*np.nan

    def add(self, bodies):
        """Add bodies to Scene."""
        self.bodies = np.append(self.bodies, bodies)
        self.get_extent()

    def get_extent(self):
        """Get the extent of the Scene."""
        if len(self.bodies) > 0:
            xmin = np.min([body.center[0]-body.radius for body in self.bodies])
            xmax = np.max([body.center[0]+body.radius for body in self.bodies])
            ymin = np.min([body.center[1]-body.radius for body in self.bodies])
            ymax = np.max([body.center[1]+body.radius for body in self.bodies])
            zmax = np.max([body.center[2]+body.radius for body in self.bodies])
        else:
            xmin, xmax = -1, 1
            ymin, ymax = -1, 1
            zmax = np.inf
        self.extent = (np.min([xmin, ymin]), np.max([xmax, ymax]))
        self.x = np.linspace(*self.extent, self.res)
        self.y = np.linspace(*self.extent, self.res)
        self.zmax = zmax

    def trace(self):
        """Perform the ray trace."""
        for j, i in np.ndindex(self.shape):
            ray = Ray(origin=np.array([self.x[i], self.y[j], self.zmax]),
                      direction=np.array([self.x[i], self.y[j], 0.]))
            t_min = np.inf
            for body in self.bodies:
                t = intersect(ray, body)
                if t >= t_min:
                    continue
                t_min = t
                P = ray.origin + ray.u*t
                N = normalize(P-body.center)
                mu = np.abs(np.cos(angle_between(ray.u, N)))

                self.body[j, i] = body
                self.t[j, i] = t
                self.P[j, i] = P
                self.N[j, i] = N
                self.mu[j, i] = mu

        # Set the inclinations of the bodies.
        for body in self.bodies:
            rot_N = self.N @ rotation_matrix(body.inc, axis='x')
            mask2d, mask3d = self.get_masks(body)
            self.N[~mask3d] = rot_N[~mask3d]

        # Set the meridians of the bodies.
        for body in self.bodies:
            rot_N = self.N @ rotation_matrix(90.+body.meridian, axis='z')
            mask2d, mask3d = self.get_masks(body)
            self.N[~mask3d] = rot_N[~mask3d]

        # The standard transformation places the observer at x=+inf
        r, theta, phi = cart2sph(self.N[:, :, 0],
                                 self.N[:, :, 1],
                                 self.N[:, :, 2])
        # Instead, let's place the observer at z=+inf
#         r, theta, phi = cart2sph(self.N[:, :, 2],
#                                  self.N[:, :, 0],
#                                  self.N[:, :, 1])
        lat = np.degrees(theta)
        lon = np.degrees(phi)
        self.r = r
        self.theta = theta
        self.phi = phi
        self.lat = lat
        self.lon = lon
        self.flux = np.ones(self.shape)

    def get_masks(self, body):
        """Get 2D and 3D masks for Body."""
        mask2d = np.ma.masked_where(self.body != body, self.body).mask
        mask3d = np.broadcast_to(np.expand_dims(mask2d, axis=2), self.N.shape)
        return mask2d, mask3d

    def show(self, array='flux', body=None):
        """Show a property of the Scene."""
        arrays = {'flux': self.flux,
                  'mu': self.mu,
                  't': self.t,
                  'P[0]': self.P[:, :, 0],
                  'P[1]': self.P[:, :, 1],
                  'P[2]': self.P[:, :, 2],
                  'N[0]': self.N[:, :, 0],
                  'N[1]': self.N[:, :, 1],
                  'N[2]': self.N[:, :, 2],
                  'r': self.r,
                  'theta': self.theta,
                  'phi': self.phi,
                  'lat': self.lat,
                  'lon': self.lon}

        cmaps = {'flux': 'viridis',
                 'mu': 'viridis',
                 't': 'viridis',
                 'N[0]': 'RdBu',
                 'N[1]': 'RdBu',
                 'N[2]': 'RdBu',
                 'theta': 'RdBu',
                 'phi': 'RdBu',
                 'lat': 'RdBu',
                 'lon': 'RdBu'}

        vmins = {'N[0]': -1,
                 'N[1]': -1,
                 'N[2]': -1,
                 'theta': -np.pi/2.,
                 'phi': -np.pi,
                 'lat': -90,
                 'lon': -180}

        vmaxs = {'N[0]': 1,
                 'N[1]': 1,
                 'N[2]': 1,
                 'theta': np.pi/2.,
                 'phi': np.pi,
                 'lat': 90,
                 'lon': 180}

        values = arrays[array]
        if body is None:
            ma = np.ma.masked_invalid(values)
        else:
            ma = np.ma.masked_where(self.body != body, values)
        cmap = cmaps.get(array, 'viridis')
        vmin = vmins.get(array, ma.min())
        vmax = vmaxs.get(array, ma.max())
        fig, ax = plt.subplots()
        im = ax.imshow(ma, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('x (pixel)')
        ax.set_ylabel('y (pixel)')
        plt.colorbar(im, label=array)
        plt.show()


def normalize(vector):
    """Normalize a vector."""
    return vector / np.linalg.norm(vector)


def intersect(Ray, Star):
    """
    Intersection of a ray and a sphere.

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


def angle_between(v1, v2):
    """Get the angle in radians between two vectors."""
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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
    """Rotate coordinate basis for point P by specified angles."""
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
    """Rotate coordinate around axis u by angle theta."""
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
    Calculate the great circle distance between two points.

    (Points specified in decimal degrees)

    Returns distance in degress
    """
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    dist = 2*np.arcsin(np.sqrt(a))
    return np.degrees(dist)


def get_none_array(shape):
    """Get a numpy array of Nones with the specified shape."""
    arr = None
    for dim in shape:
        arr = [arr]*dim
    return np.array(arr)


def cart2sph(x, y, z):
    """Transform cartesian to spherical coordinates."""
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, el, az
