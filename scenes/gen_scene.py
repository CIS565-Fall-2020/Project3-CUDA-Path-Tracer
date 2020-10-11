import random

mat_count = 30
sphere_size = 1


lines = []

for i in range(2, mat_count + 2):
  lines.append('MATERIAL ' + str(i))
  r = random.random()
  g = random.random()
  b = random.random()
  lines.append('RGB {} {} {}'.format(r,g,b))
  # 0 for diffuse, 1 for reflect, 2 for refract
  mat_type = random.randint(0, 2)
  is_diffuse = mat_type == 0
  is_reflect = mat_type == 1
  is_refract = mat_type == 2
  lines.append('SPECEX      0')
  specr = 0 if is_diffuse else random.random()
  specg = 0 if is_diffuse else random.random()
  specb = 0 if is_diffuse else random.random()
  lines.append('SPECRGB     {} {} {}'.format(specr, specg, specb))
  lines.append('REFL     {}'.format(1 if is_reflect else 0))
  lines.append('REFR     {}'.format(1 if is_refract else 0))
  lines.append('REFRIOR  {}'.format(1.52 if is_refract else 0))
  lines.append('EMITTANCE   0')
  lines.append(' ')

obj_lines = []
for i in range(6, mat_count + 6):
  obj_lines.append('OBJECT {}'.format(i))
  obj_lines.append('sphere')
  obj_lines.append('material {}'.format(i))
  row = (i - 6) // 6
  col = (i - 6) % 6
  width = 8
  height = 8
  pos_x = - width / 2 + col * width / 5
  pos_y = 5 - height / 2 + row * height / 4
  obj_lines.append('TRANS {} {} {}'.format(pos_x, pos_y, 0))
  obj_lines.append('ROTAT 0 0 0')
  obj_lines.append('SCALE 1 1 1')
  obj_lines.append(' ')
  

f = open('materials.txt', 'w')
f.write("""// Emissive material (light)
MATERIAL 0
RGB         1 1 1
SPECEX      0
SPECRGB     0 0 0
REFL        0
REFR        0
REFRIOR     0
EMITTANCE   5

// Diffuse white
MATERIAL 1
RGB         .98 .98 .98
SPECEX      0
SPECRGB     0 0 0
REFL        0
REFR        0
REFRIOR     0
EMITTANCE   0

""")

for line in lines:
  f.write(line + '\n')

f.write("""// Camera
CAMERA
RES         800 800
FOVY        45
ITERATIONS  5000
DEPTH       8
FILE        cornell
EYE         0.0 5 10.5
LOOKAT      0 5 0
UP          0 1 0
DOF         0
MOTION      0 0 0

// Ceiling light
OBJECT 0
cube
material 0
TRANS       0 10 0
ROTAT       0 0 0
SCALE       3 .3 3

// Floor
OBJECT 1
cube
material 1
TRANS       0 0 0
ROTAT       0 0 0
SCALE       10 .01 10

// Ceiling
OBJECT 2
cube
material 1
TRANS       0 10 0
ROTAT       0 0 90
SCALE       .01 10 10

// Back wall
OBJECT 3
cube
material 1
TRANS       0 5 -5
ROTAT       0 90 0
SCALE       .01 10 10

// Left wall
OBJECT 4
cube
material 1
TRANS       -5 5 0
ROTAT       0 0 0
SCALE       .01 10 10

// Right wall
OBJECT 5
cube
material 1
TRANS       5 5 0
ROTAT       0 0 0
SCALE       .01 10 10

""")

for line in obj_lines:
  f.write(line + '\n')