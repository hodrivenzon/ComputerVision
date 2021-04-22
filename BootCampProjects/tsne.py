import numpy as np
import matplotlib.pyplot as plt



g_kernel = 1

def gaussian_function(x1, x2):
    return np.exp(-np.linalg.norm(x1 - x2, axis=1) ** 2 / (2 * g_kernel ** 2))

def t_distribution(x1, xi):
    return (1 + np.linalg.norm(x1 - xi, axis=1) ** 2) ** -1

# compute the distance between the neighbors of x1 and return a list of the k neighbors
# where k is the complexity
def k_neighbours(all_data, x_i_index, p_or_q='p', perpelexity=15):
    num_rows = len(all_data)
    x_i = all_data[x_i_index].reshape(1, -1)
    all_data_without_i = np.delete(all_data.copy(), x_i_index, axis=0)
    if p_or_q == 'p':
        probs = gaussian_function(x_i, all_data_without_i)
    else:
        probs = t_distribution(x_i, all_data_without_i)

    pts_prob_dist = np.column_stack((range(num_rows - 1), probs))
    pts_prob_dist_sorted = pts_prob_dist[pts_prob_dist[:, 1].argsort()][::-1, :]
    return pts_prob_dist_sorted[:perpelexity]

# compute the similarity pij between two xi,xj in the original space
# divide the distance between xi,xj by the sum of the distances of the k_neightbours where k is the complexity
def compute_pij(all_data, i_index, j_index, perpelexity=15):
    x_i = all_data[i_index].reshape(1, -1)
    x_j = all_data[j_index].reshape(1, -1)
    numerator = gaussian_function(x_i, x_j)
    pts_prob_dist = k_neighbours(all_data, i_index, 'p', perpelexity)
    denomerator = pts_prob_dist.sum(axis=0)[1]
    return numerator / denomerator

# compute the table p of the xij in the original space
def compute_all_p(all_data, perpelexity=15):
    num_rows = all_data.shape[0]
    table_pij = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(num_rows):
            if i != j:
                pij = compute_pij(all_data, i, j, perpelexity)
                pji = compute_pij(all_data, j, i, perpelexity)
                table_pij[i, j] = (pij + pji) / (2 * num_rows)
                # table[i,j]=pij
    return table_pij

# compute the similarity qij between two yi,yj in the new space
# divide the distance between yi,yj by the sum of the distances of the k_neightbours where k is the complexity
def compute_qij(y, y_i_index, y_j_index):
    y_i = y[y_i_index]
    y_j = y[y_j_index]
    numerator = (1 + np.linalg.norm(y_i - y_j) ** 2) ** (-1)
    pts_prob_dist = k_neighbours(y, y_i_index, 'q')
    denominator = pts_prob_dist.sum(axis=0)[1]
    return numerator / denominator

# compute the table q of the yij in the new space
def compute_all_q(y):
    num_rows = y.shape[0]
    table_qij = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(num_rows):
            if i != j:
                qij = compute_qij(y, i, j)
                table_qij[i, j] = qij
    return table_qij

# compute the erros between the 2 distributions using the KL-divergence
def kl_divergence(p, q):
    kl_div_table = p * np.log(p / q)
    kl_div_table[np.isnan(kl_div_table)] = 0
    kl_divergence_value = kl_div_table.sum()
    return kl_divergence_value

# apply gradient descent to lower the KL-divergence
# added momentum increase the speed
def gradient_descent(p, q, y, epochs=2000, learning_rate=200, momentum=0.99):
    num_rows = p.shape[0]
    history = np.zeros((num_rows, 2, y.shape[1]))
    for epoch in range(epochs):
        for i in range(num_rows):
            grad_value = 0
            for j in range(y.shape[0]):
                grad_value += 4 * (
                            (y[i] - y[j]) * (p[i, j] - q[i, j]) * (1 + np.linalg.norm(y[i] - y[j] ** 2)) ** -1)
            y[i] -= learning_rate * grad_value + momentum * (history[i, 1] - history[i, 0])
            history[i, 0] = history[i, 1]
            history[i, 1] = y[i]
        q = compute_all_q(y)
        if epoch % 100 == 0:
            print(kl_divergence(p, q))
    y -= np.mean(y)
    y /= np.std(y)
    return y




if __name__ == '__main__':
    x = np.random.rand(10, 3)
    x = np.tile(x, (2, 1))
    x[:10] *= 0.1
    label = ['blue'] * 10 + ['red'] * 10

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], color=label)
    plt.show()

    table_p = compute_all_p(x, perpelexity=20)

    # Probably not the right way to initialize the new space y
    # y is the initial placement of x in the lower dimension
    y = x.dot(np.random.rand(x.shape[1], 2))
    y -= np.mean(y)
    y /= np.std(y)
    table_q = compute_all_q(y)
    y = gradient_descent(table_p, table_q, y, epochs=4000)

    plt.scatter(y[:, 0], y[:, 1], color=label)
    plt.show()