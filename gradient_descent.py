from numpy import *
import matplotlib.pyplot as plt

def compute_error_of_given_points(c, m, points):
    totalError = 0

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + c)) ** 2
    
    return totalError / float(len(points))


def step_gradient(current_c, current_m, points, learning_rate):
    # gradient descent
    c_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        c_gradient += -(2 / N) * (y - ((current_m * x) + current_c))
        m_gradient += -(2 / N) * x * (y - ((current_m * x) + current_c))
    
    new_c = current_c - (learning_rate * c_gradient)
    new_m = current_m - (learning_rate * m_gradient)

    return [new_c, new_m]


def gradient_descent_runner(points, starting_c, starting_m, learning_rate, num_iterations):
    c = starting_c
    m = starting_m
    plt.scatter(points[:, 0], points[:, 1])
    for i in range(num_iterations):
        c, m = step_gradient(c, m, array(points), learning_rate)
        plt.plot(points[:, 0], m * points[:, 0] + c)
    
    return [c, m]


def run():
    points = genfromtxt('data/gradient_descent.csv', delimiter=',')
    learning_rate = 0.0001      # hyperparameters
    # y = mx + c
    initial_c = 0
    initial_m = 0
    num_iterations = 1000
    print("Starting gradient descent at c = {0}, m = {1}, error = {2}".format(initial_c, initial_m, compute_error_of_given_points(initial_c, initial_m, points)))
    print("Running...")
    [c, m] = gradient_descent_runner(points, initial_c, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, c, m, compute_error_of_given_points(c, m, points)))

    predict = list()
    for i in range(len(points)):
        predict.append(m * points[i, 0] + c)
    plt.show()


if __name__ == '__main__':
    run()
