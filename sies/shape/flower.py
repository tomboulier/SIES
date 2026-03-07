import numpy as np
from .base import Shape

class Flower(Shape):
    def __init__(self, a, b, nb_points, nb_petals=5, epsilon=0.3, tau=0):
        """
        Flower shape (rotational symmetric shapes).
        a, b: typical size of the flower
        nb_points: number of discretization points
        nb_petals: number of petals
        epsilon: size of perturbation
        tau: percentage of damage (0 to 1) - Currently only tau=0 is fully supported
        """
        if tau != 0:
            # damaged flower not implemented for now to keep it simple,
            # as it's not used in the basic demo.
            raise NotImplementedError("Damaged flower (tau > 0) is not yet implemented.")

        self.axis_a = a
        self.axis_b = b
        self.nb_petals = nb_petals
        self.epsilon = epsilon
        self.tau = tau
        self.phi_rot = 0

        theta = 2 * np.pi * np.arange(nb_points) / nb_points
        k = 1 # fixed in Flower.m constructor as pertb = 1

        # Position
        radial_pert = (1 + epsilon * np.cos(nb_petals * theta)**k)
        points = np.vstack([
            a * np.cos(theta) * radial_pert,
            b * np.sin(theta) * radial_pert
        ])

        # Velocity (tangent)
        # Using the formula from make_flower.m
        tx = -a * (np.sin(theta) * radial_pert + k * epsilon * nb_petals * np.sin(nb_petals * theta) * np.cos(nb_petals * theta)**(k-1) * np.cos(theta))
        ty = b * (np.cos(theta) * radial_pert - k * epsilon * nb_petals * np.sin(nb_petals * theta) * np.cos(nb_petals * theta)**(k-1) * np.sin(theta))
        tvec = np.vstack([tx, ty])

        # Acceleration
        # Using the formula from make_flower.m for k=1
        ax = -a * (np.cos(theta) * (1 + epsilon * np.cos(nb_petals * theta)) - 2 * epsilon * nb_petals * np.sin(theta) * np.sin(nb_petals * theta) + epsilon * nb_petals**2 * np.cos(nb_petals * theta) * np.cos(theta))
        ay = -b * (np.sin(theta) * (1 + epsilon * np.cos(nb_petals * theta)) + 2 * epsilon * nb_petals * np.cos(theta) * np.sin(nb_petals * theta) + epsilon * nb_petals**2 * np.cos(nb_petals * theta) * np.sin(theta))
        avec = np.vstack([ax, ay])

        # Normal
        normal = np.vstack([tvec[1, :], -tvec[0, :]])
        normal /= np.sqrt(np.sum(normal**2, axis=0))

        super().__init__(points, tvec, avec, normal, center_of_mass=[0,0], name='Flower')

    def __mul__(self, s):
        new_obj = super().__mul__(s)
        new_obj.axis_a = self.axis_a * s
        new_obj.axis_b = self.axis_b * s
        new_obj.nb_petals = self.nb_petals
        new_obj.epsilon = self.epsilon
        new_obj.tau = self.tau
        new_obj.phi_rot = self.phi_rot
        return new_obj

    def __matmul__(self, phi):
        new_obj = super().__matmul__(phi)
        new_obj.axis_a = self.axis_a
        new_obj.axis_b = self.axis_b
        new_obj.nb_petals = self.nb_petals
        new_obj.epsilon = self.epsilon
        new_obj.tau = self.tau
        new_obj.phi_rot = self.phi_rot + phi
        return new_obj
