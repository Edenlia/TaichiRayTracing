import taichi as ti
import taichi.math as tm
import math
from ray import Ray


@ti.data_oriented
class Hittable:
    def __init__(self):
        pass

    @ti.func
    def hit(self, ray: Ray, t_min: ti.f32, t_max: ti.f32):
        raise NotImplementedError



@ti.data_oriented
class Sphere(Hittable):
    def __init__(self, center, radius, material):
        super().__init__()
        self.center = center
        self.radius = radius
        self.material = material

    @ti.func
    def hit(self, ray: Ray, t_min: ti.f32, t_max: ti.f32):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        half_b = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c
        is_hit = True
        if discriminant < 0:
            is_hit = False

        sqrtd = tm.sqrt(discriminant)
        root = (-half_b - sqrtd) / a
        if root < t_min or t_max < root:
            root = (-half_b + sqrtd) / a
            if root < t_min or t_max < root:
                is_hit = False

        hit_t = root
        hit_p = ray.at(hit_t)
        outward_normal = (hit_p - self.center) / self.radius
        front_face = ray.direction.dot(outward_normal) < 0 # ray is outside the sphere if true
        hit_normal = outward_normal
        hit_material_name = self.material.name
        hit_material_albedo = self.material.albedo
        if not front_face:
            hit_normal = -outward_normal

        return is_hit, hit_p, hit_normal, hit_t, hit_material_name, hit_material_albedo



