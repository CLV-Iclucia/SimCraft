def generate_cube_tobj(size, resolution, filename):
    step = size / resolution

    vertices = []
    tetrahedra = []

    for k in range(resolution + 1):
        for j in range(resolution + 1):
            for i in range(resolution + 1):
                x = i * step
                y = j * step
                z = k * step
                vertices.append((x, y, z))

    def vertex_index(i, j, k):
        return (k * (resolution + 1) * (resolution + 1) + j * (resolution + 1) + i) + 1

    for k in range(resolution):
        for j in range(resolution):
            for i in range(resolution):
                v0 = vertex_index(i, j, k + 1)
                v1 = vertex_index(i + 1, j, k + 1)
                v2 = vertex_index(i, j, k)
                v3 = vertex_index(i + 1, j, k)
                v4 = vertex_index(i, j + 1, k + 1)
                v5 = vertex_index(i + 1, j + 1, k + 1)
                v6 = vertex_index(i, j + 1, k)
                v7 = vertex_index(i + 1, j + 1, k)

                if (i + j + k) % 2 == 0:
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

size = 2.0
resolution = 50
filename = 'cube' + str(resolution) + 'x' + str(resolution) + '.tobj'
generate_cube_tobj(size, resolution, filename)
