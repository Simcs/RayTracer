# RayTracer

***

Implemntation of ray tracing algorithm using python.  
This ray tracer has support for:
- Spheres and axis-aligned boxes
- Lambertian and Phong shading
- Arbitratry perspective cameras

<br>

Use XML file as input which contains sequences of nested elements.  
This XML file is allowed to contain tags of the following types:
- surface : This element describes a geometric object and supports two types (Sphere, Box).
- camera : This element describes the camera.
- image : This element specifies the size of the output image in pixels.
- light : This element describes a light. Its 3D position and RGB color.
- shader : This element describes how a surface should be shaded and supports two types (Lambertian, Phong).
