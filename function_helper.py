import pandas as pd


def read_file(path):
    data = pd.read_csv(path, sep="\s+", header=None,).to_numpy()

    return data


def read_coefficient_a():
    c_vector = read_file("fun2_A.txt").reshape(100, 500).T

    return c_vector


def read_coefficient_b():
    b_vector = read_file("fun2_b.txt").reshape(-1, 1)

    return b_vector


def read_coefficient_c():
    c_vector = read_file("fun2_c.txt").reshape(-1, 1)

    return c_vector

if __name__ == "__main__":
    print(read_coefficient_a().shape)
    print(read_coefficient_b().shape)
    print(read_coefficient_c().shape)

