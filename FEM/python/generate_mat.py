length = 10.0
width = 10.0
height = 0.1
resolution_x = 100
resolution_y = 100

filename = 'mat' + str(resolution_x) + 'x' + str(resolution_y) + '.tobj'

dx = length / resolution_x
dy = width / resolution_y

vertices = []
tetrahedra = []

for j in range(resolution_y + 1):
    for i in range(resolution_x + 1):
        x = i * dx
        y = j * dy
        vertices.append((x, y, 0))
        vertices.append((x, y, height))

# order y x z
def vertex_index(i, j, z_top):
    return (j * (resolution_x + 1) * 2 + i * 2 + z_top) + 1

for j in range(resolution_y):
    for i in range(resolution_x):
        v0 = vertex_index(i, j, 1)
        v1 = vertex_index(i + 1, j, 1)
        v2 = vertex_index(i, j, 0)
        v3 = vertex_index(i + 1, j, 0)
        v4 = vertex_index(i, j + 1, 1)
        v5 = vertex_index(i + 1, j + 1, 1)
        v6 = vertex_index(i, j + 1, 0)
        v7 = vertex_index(i + 1, j + 1, 0)

        if (i + j) % 2 == 0:
            tetrahedra.append((v0, v3, v5, v6))
            tetrahedra.append((v0, v1, v3, v5))
            tetrahedra.append((v0, v2, v3, v6))
            tetrahedra.append((v0, v4, v5, v6))
            tetrahedra.append((v3, v5, v6, v7))
        else:
            tetrahedra.append((v0, v1, v2, v4))
            tetrahedra.append((v1, v2, v3, v7))
            tetrahedra.append((v1, v2, v4, v7))
            tetrahedra.append((v1, v4, v5, v7))
            tetrahedra.append((v2, v4, v6, v7))

with open(filename, 'w') as f:
    for v in vertices:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")

    for t in tetrahedra:
        f.write(f"t {t[0]} {t[1]} {t[2]} {t[3]}\n")
