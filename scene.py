import taichi as ti
import taichi.math as tm
import math
from ray import Ray
from polygone import Sphere
import numpy as np

ti.init(arch=ti.gpu)

aspect_ratio = 16.0 / 9.0
width = 960
height = int(width / aspect_ratio)
samples_per_pixel = 100
max_depth = 50

canvas = ti.Vector.field(3, dtype=float, shape=(width, height))

gui = ti.GUI('Hello World!', (width, height))

M_Default = -1
M_Lambertian = 0
M_Metal = 1
M_Fuzz_Metal = 2


@ti.func
def random_vec3(vec_min: ti.f32, vec_max: ti.f32) -> tm.vec3:
    a = ti.random(dtype=ti.f32) * (vec_max - vec_min) + vec_min
    b = ti.random(dtype=ti.f32) * (vec_max - vec_min) + vec_min
    c = ti.random(dtype=ti.f32) * (vec_max - vec_min) + vec_min

    return tm.vec3([a, b, c])


@ti.func
def random_in_hemisphere(normal: tm.vec3) -> tm.vec3:
    in_unit_sphere = random_in_unit_sphere()
    if in_unit_sphere.dot(normal) <= 0.0:
        in_unit_sphere = -in_unit_sphere

    return in_unit_sphere


@ti.func
def random_in_unit_sphere() -> tm.vec3:
    p = random_vec3(-1.0, 1.0)
    while p.norm() >= 1.0:
        p = random_vec3(-1.0, 1.0)
    return p


@ti.func
def random_unit_vector() -> tm.vec3:
    return random_in_unit_sphere().normalized()


@ti.func
def degrees_to_radians(degrees: float) -> float:
    return degrees * math.pi / 180.0


@ti.func
def near_zero(vec: tm.vec3) -> bool:
    s = 1e-8
    return (vec[0] < s) and (vec[1] < s) and (vec[2] < s)


@ti.func
def reflect(vec: tm.vec3, normal: tm.vec3) -> tm.vec3:
    return vec - 2 * vec.dot(normal) * normal


@ti.func
def ray_color(ray, scene, depth) -> tm.vec3:
    pixel_color = tm.vec3([0.0, 0.0, 0.0])
    buffer_color = tm.vec3([1.0, 1.0, 1.0])
    scattered_ray = ray
    for i in range(depth):
        is_hit, hit_p, hit_normal, hit_t, hit_material_name, hit_material_albedo = scene.hit(scattered_ray, 0.001,
                                                                                             math.inf)
        if not is_hit:
            direction = scattered_ray.direction.normalized()
            t = 0.5 * (direction[1] + 1.0)
            pixel_color = (1.0 - t) * tm.vec3([1.0, 1.0, 1.0]) + t * tm.vec3([0.5, 0.7, 1.0])
            pixel_color *= buffer_color
            break

        if hit_material_name == M_Lambertian:
            is_scatter, attenuation, scatter_ray = Lambertian.scatter(scattered_ray, hit_normal, hit_p,
                                                                      hit_material_albedo)
            if not is_scatter:
                pixel_color = tm.vec3([0.0, 0.0, 0.0])
                break
            buffer_color *= attenuation
            scattered_ray = scatter_ray
        elif hit_material_name == M_Metal:
            is_scatter, attenuation, scatter_ray = Metal.scatter(scattered_ray, hit_normal, hit_p, hit_material_albedo)
            if not is_scatter:
                pixel_color = tm.vec3([0.0, 0.0, 0.0])
                break
            buffer_color *= attenuation
            scattered_ray = scatter_ray
        # target = hit_p + hit_normal + random_unit_vector()
        # scattered_ray = Ray(origin=hit_p, direction=target - hit_p)
    return pixel_color


# pixel sampling
@ti.func
def write_color(pixel_color: tm.vec3, samples_per_pixel: int):
    r = pixel_color[0]
    g = pixel_color[1]
    b = pixel_color[2]

    scale = 1.0 / samples_per_pixel
    r *= scale
    g *= scale
    b *= scale

    r = ti.sqrt(r)
    g = ti.sqrt(g)
    b = ti.sqrt(b)

    return tm.vec3([tm.clamp(r, 0.0, 0.999), tm.clamp(g, 0.0, 0.999), tm.clamp(b, 0.0, 0.999)])


@ti.kernel
def render(scene: ti.template()):
    for i, j in canvas:
        pixel_color = tm.vec3([0.0, 0.0, 0.0])
        for s in range(samples_per_pixel):
            u = (i + ti.random(dtype=ti.f32)) / width
            v = (j + ti.random(dtype=ti.f32)) / height
            ray = Ray(origin=camera.origin,
                      direction=camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin)

            pixel_color += ray_color(ray, scene, max_depth)

        canvas[i, j] = write_color(pixel_color, samples_per_pixel)


class Material:
    @staticmethod
    @ti.func
    def scatter(ray_in, hit_normal, hit_p):
        pass


@ti.data_oriented
class Lambertian(Material):  # material 1
    def __init__(self, albedo, name):
        self.albedo = albedo
        self.name = name

    @staticmethod
    @ti.func
    def scatter(ray_in, hit_normal, hit_p, albedo):
        scatter_direction = hit_normal + random_unit_vector()
        if near_zero(scatter_direction):
            scatter_direction = hit_normal
        scattered = Ray(origin=hit_p, direction=scatter_direction)
        attenuation = albedo
        is_scatter = True
        return is_scatter, attenuation, scattered

    def get_material_info(self):
        return {'albedo': self.albedo, 'name': self.name}


@ti.data_oriented
class Metal(Material):  # material 2
    def __init__(self, albedo, name):
        self.albedo = albedo
        self.name = name

    @staticmethod
    @ti.func
    def scatter(ray_in, hit_normal, hit_p, albedo):
        fuzz = 0.6  # the fuzz of the metal

        reflected = reflect(ray_in.normalized_direction(), hit_normal)
        scattered = Ray(origin=hit_p, direction=reflected + random_in_unit_sphere() * fuzz)
        attenuation = albedo
        is_scatter = scattered.normalized_direction().dot(hit_normal) > 0
        return is_scatter, attenuation, scattered

    def get_material_info(self):
        return {'albedo': self.albedo, 'fuzz': self.fuzz, 'name': self.name}


@ti.data_oriented
class Scene:
    def __init__(self):
        self.objList = []

    def add(self, obj):
        self.objList.append(obj)

    @ti.func
    def hit(self, ray, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max
        closest_p, closest_normal = tm.vec3([0.0, 0.0, 0.0]), tm.vec3([0.0, 0.0, 0.0])
        closest_material_name, closest_material_albedo = M_Default, tm.vec3([0.0, 0.0, 0.0])
        for i in ti.static(range(len(self.objList))):
            is_hit, hit_p, hit_normal, hit_t, hit_material_name, hit_material_albedo = self.objList[i].hit(ray, t_min,
                                                                                                           closest_so_far)
            if is_hit:
                hit_anything = True
                closest_so_far = hit_t
                closest_p = hit_p
                closest_normal = hit_normal
                closest_material_name = hit_material_name
                closest_material_albedo = hit_material_albedo
        # print(hit_material_name, hit_material_albedo)
        return hit_anything, closest_p, closest_normal, closest_so_far, closest_material_name, closest_material_albedo


@ti.data_oriented
class Camera:
    def __init__(self):
        self.aspect_ratio = aspect_ratio
        self.viewport_height = 2.0
        self.viewport_width = self.aspect_ratio * self.viewport_height
        self.focal_length = 1.0

        self.origin = tm.vec3([0.0, 0.0, 0.0])
        self.horizontal = tm.vec3([self.viewport_width, 0.0, 0.0])
        self.vertical = tm.vec3([0.0, self.viewport_height, 0.0])
        self.lower_left_corner = self.origin - self.horizontal / 2 - self.vertical / 2 - tm.vec3(
            [0.0, 0.0, self.focal_length])

    @ti.func
    def get_ray(self, u, v):
        return Ray(self.origin, self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin)


# Scene
material_ground = Lambertian(tm.vec3([0.8, 0.8, 0.0]), M_Lambertian)
material_center = Lambertian(tm.vec3([0.7, 0.3, 0.3]), M_Lambertian)
material_left = Metal(tm.vec3([0.8, 0.8, 0.8]), M_Metal)
material_right = Metal(tm.vec3([0.8, 0.6, 0.2]), M_Metal)

sphere1 = Sphere(tm.vec3([0, -100.5, -1]), 100, material_ground)
sphere2 = Sphere(tm.vec3([0, 0, -1]), 0.5, material_center)
sphere3 = Sphere(tm.vec3([-1, 0, -1]), 0.5, material_left)
sphere4 = Sphere(tm.vec3([1, 0, -1]), 0.5, material_right)
global_scene = Scene()
global_scene.add(sphere1)
global_scene.add(sphere2)
global_scene.add(sphere3)
global_scene.add(sphere4)

# Camera
camera = Camera()
# viewpoint_height = 2.0
# viewpoint_width = aspect_ratio * viewpoint_height
# focal_length = 1.0
#
# origin = tm.vec3([0.0, 0.0, 0.0])
# horizontal = tm.vec3([viewpoint_width, 0.0, 0.0])
# vertical = tm.vec3([0.0, viewpoint_height, 0.0])
# lower_left_corner = origin - horizontal / 2 - vertical / 2 - tm.vec3([0.0, 0.0, focal_length])

while gui.running:
    render(global_scene)
    gui.set_image(canvas)
    gui.show()
    while True:
        pass
