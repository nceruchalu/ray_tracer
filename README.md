# A CUDA RealTime Ray Tracer

|         |                                                              |
| ------- | ------------------------------------------------------------ |
| Author  | Nnoduka Eruchalu                                             |
| Date    | 03/07/2012                                                   |
| Course  | [Parallel Algorithms for Scientific Applications [ACM/CS 114]](http://www.cacr.caltech.edu/main/) |


## Technologies:
* [CUDA 4.0](https://developer.nvidia.com/cuda-toolkit-40)
* C
    * Would have preferred to use C++ and recursion but CUDA 4.0 doesn't officially support those. 
    * Had to make do with structs (for objects) and implement binary tree based stack (for recursion)

## About Ray Tracing:
Ray tracing is 3d graphics by tracing a path of light through pixels in a scene and simulating the effects of its encounters with virtual objects. It is based on the idead that you can model reflection and refraction by recursively following the path that these light rays take as they bounce through the 3d scene.

The problem with ray tracing is that it has a great computational cost so it is typically used in situations where the images can be rednered slowly ahead of time. 

However with the power of a CUDA-enabled NVIDIA GPU I've been able to create a ray tracer that actually renders 3d graphic scenes in realtime. This makes it possible to translate, rotate and zoom in/out of the scene.

Read the following for more background information on Ray Tracers:

* [ACM Siggraph Ray Tracing](http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtrace0.htm)
* [Paul Rademacher's "Ray Tracing: Graphics for the Masses"](http://www.cs.unc.edu/~rademach/xroads-RT/RTarticle.html)
* [FuzzyPhoton "What is Ray Tracing?"](http://fuzzyphoton.tripod.com/whatisrt.htm)


## Software Description
| Module             | Description                                             |
| ------------------ | ------------------------------------------------------- |
| `Makefile`         | Creates executable `ray_tracer`
| `ray_tracer.cu`    | Main Loop: initialize Pixel Buffer, Scene, mouse event handlers and start rendering |
| `tracer_kernel.cu` | Implement Ray Tracing logic and binary tree stack (used for recursion) |
| `scene.in`         | Sample input scene description file                     |


## Compiling
This program includes a `Makefile` for use in compiling.

To simplify compilation, the kernel file is simply included into the main file.
As a result, `make` might not detect changes in the kernel, so you should use
`make -B` (rebuild all).

## Running
First of all you need to be on a machine with a [CUDA-capable NVIDIA GPU](https://developer.nvidia.com/cuda-gpus)
To run, go to the folder where executables go and call the ray_tracer program with an input scene description file. More on creating input scene description files later.
```
./ray_tracer /path/to/scene.in
```
Note that on my CUDA machine setup, the executables folder is in `C/bin/linux/release`


## User Interface:
The User Interface is the mouse:

| Button       | Action      |
| ------------ | ----------- |
| `right`      | zoom        |
| `middle`     | translate   |
| `left`       | rotate      |

Click the `esc` key to close the window.


## Scene Description
A scene description input file is how the Ray Tracer knows what objects to draw and properties.

I came up with the syntax for this file and it's not very flexible so read the syntax rules that follow:

* 1023 characters maximum per line.

* Comments must be on separate lines that start with `#`. Yes I did this because I'm a python fan.

* A blank line should truly be blank! i.e. a line with anything other than a carriage return will be considered not-empty.
    * This means a line with just one `space character` will fail.

* Mark the end of the file with a `*` character on a new line.

* Each non-comment line of the file is a unique key-value pair representing an object's properties.
    * the key is always a single word
    * space(s) follows the key
    * the first non-space character following the key marks the beginning of the value.
    * The value could be single-value or multi-value

* Below are example formats of object property key-value pairs:
```
[property key]  [propert value]
c               x y z
r               x
```

* The first non-commented line in this file must have the key `type`
* See the sample scene description input file `scene.in` for all supported objects and sample property key-value representations


### Light Property Key's
All lights have the following properties:

| Key      | Description                |
| -------- | -------------------------- |
| `amb`    | ambient                    |
| `diff`   | diffuse                    |
| `spec`   | specular                   |
| `colors` |                            |
| `pos`    | position in 3d-coordinates |



### General Object Property Key's
Supported object types are:

* Sphere
* Box
* Plane
* Cylinder
* Cone

These objects all have the following properties:

| Key      | Description              |
| -------- | ------------------------ |
| `amb`    | ambient                  |
| `diff`   | diffuse                  |
| `spec`   | specular                 |
| `colors` |                          |
| `shiny`  | shininess                |
| `n`      | refraction index         |
| `kr`     | reflective coefficient   |
| `kt`     | transmittive coefficient |

### Object type specific properties
##### Sphere:

| Key      | Description              |
| -------- | ------------------------ |
| `c`      | center                   |
| `r`      | radius                   |

##### Box:

| Key      | Description              |
| -------- | ------------------------ |
| `min`    | minimum vertex           |
| `max`    | maximum vertex           |

##### Plane:
Plane equation is `Ax + By + Cx + D = 0` and this is represented by the by the key-value pair: 
```
pl    A B C D
```

| Key      | Description              |
| -------- | ------------------------ |
| `pl`     | plane equation representation. Value is of format `A B C D` given equation `Ax + By + Cx + D = 0`    |

##### Cylinder (vertical & capless):

| Key      | Description                       |
| -------- | --------------------------------- |
| `ymin`   | y-coordinate of cylinder's bottom |
| `ymax`   | y-coordinate of cylinder's top    |
| `r`      | radius of cylinder                |

##### Cone (vertical & capless):

| Key      | Description                       |
| -------- | --------------------------------- |
| `ymin`   | y-coordinate of cone's bottom     |
| `ymax`   | y-coordinate of cone's top        |
| `r`      | base radius of cone               |


## Screenshots
#### Scene On Program Start Using `scene.in`
[![Startup Image][on-start.jpg]][on-start.jpg]

#### Scene Rotation To State Showcasing Object Translucency
Observe the translucent cube in front of the green sphere
[![Translucency Demo][translucency.jpg]][translucency.jpg]

#### Scene Rotation To State Showcasing Object Reflections & Shadows
Observe reflections and shadows in the plane.
[![Reflections Demo][reflections.jpg]][reflections.jpg]


[on-start.jpg]: https://s3.amazonaws.com/projects.nnoduka.com/raytracer/on-start.jpg "Startup Scene generated from scene.in"
[translucency.jpg]: https://s3.amazonaws.com/projects.nnoduka.com/raytracer/translucency.jpg "Rotated scene showcasing translucency"
[reflections.jpg]: https://s3.amazonaws.com/projects.nnoduka.com/raytracer/reflections.jpg "Rotated scene showcasing reflections & shadows"
