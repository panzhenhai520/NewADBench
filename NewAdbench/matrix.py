import numpy as np
# Define the pairwise comparison matrix for criteria
criteria_matrix = np.array([
    [1, 4, 3, 1, 3, 4],
    [1/4, 1, 7, 3, 1/5, 1],
    [1/3, 1/7, 1, 1/5, 1/5, 1/6],
    [1, 1/3, 5, 1, 1, 3],
    [1/3, 5, 5, 1, 1, 3],
    [1/4, 1, 6, 3, 1/3, 1]
])

# Define the pairwise comparison matrices for alternatives for each criterion
learning_matrix = np.array([
    [1, 1/3, 1/2],
    [3, 1, 3],
    [2, 1/3, 1]
])

friends_matrix = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

school_life_matrix = np.array([
    [1, 5, 1],
    [1/5, 1, 1/5],
    [1, 5, 1]
])

vocational_training_matrix = np.array([
    [1, 9, 7],
    [1/9, 1, 1/5],
    [1/7, 5, 1]
])

college_prep_matrix = np.array([
    [1, 1/2, 1],
    [2, 1, 2],
    [1, 1/2, 1]
])

music_classes_matrix = np.array([
    [1, 6, 4],
    [1/6, 1, 1/3],
    [1/4, 3, 1]
])

# Function to calculate the weights from a pairwise comparison matrix
def calculate_weights(matrix):
    eigenvector, _ = np.linalg.eig(matrix)
    max_eigenvalue = max(eigenvector.real)
    weights = eigenvector.real / np.sum(eigenvector.real)
    return weights

# Calculate weights for criteria
criteria_weights = calculate_weights(criteria_matrix)

# Calculate weights for alternatives for each criterion
learning_weights = calculate_weights(learning_matrix)
friends_weights = calculate_weights(friends_matrix)
school_life_weights = calculate_weights(school_life_matrix)
vocational_training_weights = calculate_weights(vocational_training_matrix)
college_prep_weights = calculate_weights(college_prep_matrix)
music_classes_weights = calculate_weights(music_classes_matrix)

# Aggregate the weights for each school
school_scores = (learning_weights * criteria_weights[0] +
                 friends_weights * criteria_weights[1] +
                 school_life_weights * criteria_weights[2] +
                 vocational_training_weights * criteria_weights[3] +
                 college_prep_weights * criteria_weights[4] +
                 music_classes_weights * criteria_weights[5])

school_scores_normalized = school_scores / np.sum(school_scores)

print (school_scores_normalized)
