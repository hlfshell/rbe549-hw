def rotate_quad_tree(tree):
    if isinstance(tree, int):
        return tree

    rotated = []
    rotated.append(rotate_quad_tree(tree[3]))
    rotated.append(rotate_quad_tree(tree[0]))
    rotated.append(rotate_quad_tree(tree[1]))
    rotated.append(rotate_quad_tree(tree[2]))

    return rotated

tree = [[0,1,1,0],1,0,[0,1,0,0]]
print(tree)
print(rotate_quad_tree(tree))
