import torch
import numpy as np
#from torchmin import minimize, least_squares


class Circle3d(torch.nn.Module):

    def __init__(self, plane, r=None, planer=None, center=None):
        super().__init__()
        center = [0, 0, 0] if center is None else center
        r = 1 if r is None else r
        a, b, c, d = plane
        x0, y0, z0 = center
        c1 = d + x0 ** 2 + y0 ** 2 + z0 ** 2 - r**2
        c2, c3, c4 = -2*z0, -2*y0, -2*x0
        self.plane = torch.tensor([a, b, c, d], dtype=torch.float32, device='cpu')
        coefs = torch.tensor([c1, c2, c3, c4], dtype=torch.float32, device='cpu')
        self.coefs = torch.nn.Parameter(coefs, requires_grad=True)

    def forward(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        c1, c2, c3, c4 = self.coefs
        a, b, c, d = self.plane
        err = c1 + z * c2 + y * c3 + x * c4 + x ** 2 + y ** 2 + z ** 2 + a * x + b * y + c * z + d
        target = torch.zeros([points.shape[0], ]).float()
        return err, target

    def make_closure(self, points):
        a, b, c, d = self.plane
        points = torch.from_numpy(points).float()
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        target = torch.zeros([points.shape[0], ]).float()

        def closure(coefs):
            c1, c2, c3, c4 = coefs
            err = c1 + z * c2 + y * c3 + x * c4 + x ** 2 + y ** 2 + z ** 2 + a * x + b * y + c * z + d
            out = torch.sum(err - target)
            out = torch.abs(out)
            return out

        return closure

    def get_parameters(self, coefs=None):
        self.coefs.requires_grad = False
        c1, c2, c3, c4 = self.coefs if coefs is None else coefs
        a, b, c, d = self.plane
        x0, y0, z0 = -c4 / 2, -c3 / 2, -c2 / 2
        r = torch.sqrt(d + x0 ** 2 + y0 ** 2 + z0 ** 2 - c1)
        self.coefs.requires_grad = True
        return r.item(), x0.item(), y0.item(), z0.item()

    def __repr__(self):
        r, x0, y0, z0 = self.get_parameter()
        return f'Circle(r={r}, center={[x0, y0, z0]})'


class Annulus(torch.nn.Module):

    def __init__(self, r0=None, r1=None, center=None):
        super().__init__()

        r0 = torch.randn(()) if r0 is None else torch.tensor(float(r0))
        r1 = torch.randn(()) if r1 is None else torch.tensor(float(r1))

        if center is None:
            center = torch.rand((3,), dtype=torch.float, requires_grad=True)
        else:
            center = torch.tensor(center, dtype=torch.float, requires_grad=True)

        self.center = torch.nn.Parameter(center).float()

        x0, y0, z0 = self.center
        r0_coef = x0 ** 2 + y0 ** 2 + z0 ** 2 - r0 ** 2
        self.r0_coef = torch.nn.Parameter(r0_coef).float()
        r1_coef = x0 ** 2 + y0 ** 2 + z0 ** 2 - r1 ** 2
        self.r1_coef = torch.nn.Parameter(r1_coef).float()
        self.target = None

    def forward(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r0_coef, r1_coef = self.r0_coef, self.r1_coef
        x0, y0, z0 = self.center
        x_sq = x ** 2
        y_sq = y ** 2
        z_sq = z ** 2

        target = torch.torch.zeros([points.shape[0], ]).float()

        # x_sq + y_sq - r1 ** 2 <= 0
        # (x - x0)**2 + (y - y0)**2 - r1**2
        # x**2 -2*x*x0 + x0**2 + y**2 -2*y*y0 + y0**2 - r1**2
        # err_r0 = torch.relu(-(x_sq + y_sq - r0 ** 2))
        err_r0 = torch.relu(-(x_sq - 2 * x * x0 + y_sq - 2 * y * y0 + r0_coef))

        # x_sq + y_sq - r0 ** 2 > 0
        # err_r1 = torch.relu(x_sq + y_sq - r1 ** 2)
        err_r1 = torch.relu(x_sq - 2 * x * x0 + y_sq - 2 * y * y0 + r1_coef)

        err = err_r0 + err_r1

        return err, target

    def get_radii(self):
        x0, y0, z0 = self.center
        r0, r1 = (torch.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2 - r) for r in (self.r0_coef, self.r1_coef))
        return r0.item(), r1.item()

    def get_center(self):
        return self.center.tolist()

    def __repr__(self):
        r0, r1 = self.get_radii()
        return f'Annulus(r0={r0}, r1={r1}, center={self.center.tolist()})'


class Ellipse(torch.nn.Module):

    def __init__(self, a=0, b=0, c=0, d=0, e=0, f=1):
        super().__init__()

        a = torch.randn(()) if a is None else torch.tensor(float(a))
        b = torch.randn(()) if b is None else torch.tensor(float(b))
        c = torch.randn(()) if c is None else torch.tensor(float(c))
        d = torch.randn(()) if d is None else torch.tensor(float(d))
        e = torch.randn(()) if e is None else torch.tensor(float(e))
        f = 1 if f is None else f

        self.a = torch.nn.Parameter(a).float()
        self.b = torch.nn.Parameter(b).float()
        self.c = torch.nn.Parameter(c).float()
        self.d = torch.nn.Parameter(d).float()
        self.e = torch.nn.Parameter(e).float()
        self.f = f
        self.target = None

    def forward(self, points):
        x, y = points[:, 0], points[:, 1]
        # x0, y0 = self.center
        # x, y = x - x0, y - y0
        a, b, c, d, e, f = self.a, self.b, self.c, self.d, self.e, self.f
        a, b, c, d, e = a * f, b * f, c * f, d * f, e * f

        err = a * x ** 2 + b * x * y + c * y ** 2 + d * x + e * y

        target = -torch.torch.ones([points.shape[0], ]).float() * f

        return err, target

    def __repr__(self):
        return (f'Ellipse(a={self.a.item()}, b={self.b.item()}, c={self.c.item()},'
                f' d={self.d.item()}, e={self.e.item()}, f=1)')

    '''
    https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
    '''

    def get_parameters(self):
        a, b, c, d, e, = (t.detach().numpy() for t in [self.a, self.b, self.c, self.d, self.e])
        f = self.f

        disc = b ** 2 - 4 * a * c
        if disc > 0:
            raise ValueError('coefficients not represent an ellipse. Discriminant must be negative!')

        # The location of the ellipse centre.
        x0 = (2 * c * d - b * e) / disc
        y0 = (2 * a * e - b * d) / disc

        t = 2 * (a * e ** 2 + c * d ** 2 - b * d * e + f * disc)
        t2 = np.sqrt((a - c) ** 2 + b ** 2)
        ap = -np.sqrt(t * ((a + c) + t2)) / disc
        bp = -np.sqrt(t * ((a + c) - t2)) / disc

        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        width_gt_height = True
        if ap < bp:
            width_gt_height = False
            ap, bp = bp, ap

        # The eccentricity.
        r = (bp / ap) ** 2
        if r > 1:
            r = 1 / r
        ec = np.sqrt(1 - r)

        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if b == 0:
            phi = 0 if a < c else np.pi / 2
        else:
            phi = np.arctan2(-b, c - a) / 2
            if a > c:
                phi += np.pi / 2
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi / 2
        phi = phi % np.pi

        return x0, y0, ap, bp, ec, phi


class Torus(torch.nn.Module):

    def __init__(self, a=None, b=None, r=None, center=None):
        super().__init__()

        a = torch.randn(()) if a is None else torch.tensor(float(a))
        b = torch.randn(()) if b is None else torch.tensor(float(b))
        r = torch.randn(()) if r is None else torch.tensor(float(r))

        if center is not None:
            center = np.array(center).astype(float)
            center = torch.tensor(center)
            self.center = torch.nn.Parameter(center).float()
        else:
            self.center = None

        self.a = torch.nn.Parameter(a).float()
        self.b = torch.nn.Parameter(b).float()
        self.r = torch.nn.Parameter(r).float()

        self.target = None

    def forward(self, points):
        if self.center is None:
            center = points.mean(axis=0)
            # self.center = torch.nn.Parameter(center).float()

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        # x, y, z = x / x.max(), y / y.max(), z / z.max()
        # x0, y0, z0 = self.center
        # x, y, z = x - x0, y - y0, z - z0
        a, b, r = self.a, self.b, self.r
        x_sq = x ** 2
        y_sq = y ** 2
        z_sq = z ** 2

        err = (z_sq / (a ** 2) + (x_sq + y_sq + (r ** 2)) / (b ** 2) - 1) ** 2 - 4 * (r ** 2) * (x_sq + y_sq) / (b ** 4)
        target = torch.zeros([points.shape[0], ]).float()

        return err, target

    def __repr__(self):
        x, y, z = 0, 0, 0
        return f'Torus(a={self.a.item()}, b={self.b.item()}, r={self.r.item()}, center=(x={x}, y={y}, z={z}))'


def fit_model(model, data, eps=1e-4, max_iter=10000, verbose=False, loss_function=None, optimizer=None):
    if loss_function is None:
        loss_function = torch.nn.MSELoss(reduction='mean')
    if optimizer is None:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-2, max_iter=10, line_search_fn='strong_wolfe')
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    # points = points - points_mean
    data = torch.from_numpy(data).float()
    prev = np.inf
    for t in range(max_iter):

        loss = None

        def closure():
            nonlocal loss
            optimizer.zero_grad()
            err, target = model(data)
            loss = loss_function(err, target)
            loss.backward()
            return loss

        optimizer.step(closure)
        '''
        # Forward pass: Compute predicted y by passing x to the model
        err, target = model(data)
        loss = loss_function(err, target)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
        d = np.abs((prev - loss.item()))
        prev = loss.item()
        converged = d < eps

        if (t % 1e3 == 0 or converged) and verbose:
            print(f'Iteration {t}, loss={loss.item():.4f}, d={d:.3e}')

        # break if converged
        if converged:
            break

    return model, loss.item()