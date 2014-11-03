__author__ = 'Mark Laane'

import cv2
import numpy
import random
import matplotlib.pyplot as plt
import sys


image_size = (200, 200)  # All face images will be resized to this


def main():
    no_of_persons = 13  # Number of persons
    samples_person = 10  # Number of samples per person

    cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('cropped image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original image', 200, 140)
    cv2.moveWindow('original image', 10, 10)
    cv2.moveWindow('cropped image', 250, 10)

    all_face_vectors = load_all_face_vectors_from_disk(no_of_persons, samples_person)

    while True:
        function = input(
            "0)Exit\n"
            "1)Test variating number of training samples\n"
            "2)Test variating number of principal components\n"
            "\n"
            "Choose function:"
        )
        if function == "1":
            test_different_training(all_face_vectors, no_of_persons, samples_person)
        elif function == "2":
            test_different_number_of_PCs(all_face_vectors, no_of_persons, samples_person, 3)
        elif function == "0":
            return



def test_different_training(all_face_vectors, no_of_persons, samples_person):

    plot_number_of_training_samples = []
    plot_recognition_rate = []

    number_of_tests = 30
    for test_no in range(number_of_tests):
        sys.stdout.write("\r%d%%" % (test_no*100//number_of_tests))
        sys.stdout.flush()
        for samples_training in range(1, samples_person):
            result = train_and_test(all_face_vectors, no_of_persons, samples_person, samples_training)
            #print("Training samples:{} result:{}".format(samples_training, result))

            plot_number_of_training_samples.append(samples_training)
            plot_recognition_rate.append(result)

    print()

    #Plot results:
    plot_results(plot_number_of_training_samples, plot_recognition_rate, 1, samples_person-1)

def test_different_number_of_PCs(all_face_vectors, no_of_persons, samples_person, samples_training):
    plot_number_of_PCs = []
    plot_recognition_rate = []
    number_of_tests = 30
    for test_no in range(number_of_tests):
        sys.stdout.write("\r%d%%" % (test_no*100//number_of_tests))
        sys.stdout.flush()
        for number_of_PCs in range(1, samples_person):
            result = train_and_test(all_face_vectors, no_of_persons, samples_person, samples_training, number_of_PCs)
            #print("Principal components:{} result:{}".format(number_of_PCs, result))

            plot_number_of_PCs.append(number_of_PCs)
            plot_recognition_rate.append(result)
    print()

    #Plot results:
    plot_results(plot_number_of_PCs, plot_recognition_rate, 1, samples_person-1)

def plot_results(x_axis, y_axis, x_min, x_max):
    #Plot datapoints
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis, 'ro', alpha=0.3, label='Datapoints')

    #Plot mean
    plot_mean_y = []
    print(x_axis)
    print(y_axis)
    for group_no in range(x_max-x_min+1):
        print(group_no)
        group = y_axis[group_no::x_max-x_min+1]
        print(group)
        mean = sum(group) / len(group)
        plot_mean_y.append(mean)
    ax.plot(x_axis[:x_max-x_min+1], plot_mean_y, label='Mean', linestyle='--')

    ax.legend(loc='lower right')
    ax.axis([x_min-1, x_max+1, 0, 1])
    plt.show()

def old():
    """
    #Normalise eigen vectors (needed for reconstruction)
    #for count, vector in enumerate(eig_vectors):
    #    print(numpy.linalg.norm(vector))
    #    eig_vectors[count] = vector/numpy.linalg.norm(vector)


    #Calculate weights for each image
    train_weights = numpy.dot(eig_vectors, T.T).T

    #Show average face
    debug_average_face = train_average_face.copy()
    debug_average_face.shape = image_size[::-1]
    cv2.imshow('averageface', debug_average_face)

    show_reconstructed(eig_vectors, train_average_face, train_weights)
    show_eigenfaces(eig_vectors)
    #Display Debug images:
    cv2.waitKey(20)
    """
    pass

def choose_face_vectors(all_face_vectors, image_numbers):
    chosen_face_vectors = numpy.zeros(
        shape=(len(image_numbers), image_size[0] * image_size[1]),
        dtype=numpy.float64
    )
    for count, number in enumerate(image_numbers):
        chosen_face_vectors[count] = all_face_vectors[number]
    return chosen_face_vectors


def train_and_test(all_face_vectors, no_of_persons, samples_person, samples_training, principal_components=None):
    train_img_no, test_img_no = randomly_assign_images(no_of_persons, samples_person, samples_training)

    #Load training faces:
    train_faces = choose_face_vectors(all_face_vectors, train_img_no)

    #Calculate the average face
    train_average_face = train_faces.mean(axis=0)
    #Calculate Eigenvectors
    T = train_faces - train_average_face
    S = numpy.dot(T, T.transpose())
    eig_values, eig_vectors = numpy.linalg.eig(S)
    eig_vectors = numpy.dot(eig_vectors, T)
    
    if principal_components is not None:
        # Sort eigenvectors and eigenvalues by eigenvalues
        idx = numpy.argsort(-eig_values)
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[idx]
        #Take only top "principal_components" of eigenvectors
        eig_values = eig_values[:principal_components]
        eig_vectors = eig_vectors[:principal_components]
    
    #Calculate weights for each image
    train_weights = numpy.dot(eig_vectors, T.T).T

    #Load images for testing
    test_faces = choose_face_vectors(all_face_vectors, test_img_no)
    test_diff = test_faces - train_average_face
    test_weights = numpy.dot(eig_vectors, test_diff.T).T


    successes = 0
    tries = 0
    for point, img_no in zip(test_weights, test_img_no):
        closest_target_no, closest_distance = test_distance(point, train_weights)
        #print("Distance:", closest_distance)

        real_class,_ = img_no
        calculated_class = closest_target_no//samples_training
        success = real_class == calculated_class
        #print("Success:{} real:{}, calculated:{}".format(success, real_class, calculated_class))
        if success:
            successes +=1
        tries += 1

    return successes/tries


def test_distance(point, targets):
    closest_target_no = None
    closest_distance = float("inf")
    for count, target in enumerate(targets):
        distance = numpy.sqrt(numpy.sum((point - target)**2))
        if distance < closest_distance:
            closest_distance = distance
            closest_target_no = count
    return closest_target_no, closest_distance

def show_eigenfaces(eig_vectors):
    eigenface = eig_vectors[0]
    print("Best eigenface:\n", eigenface)
    assert isinstance(eigenface, numpy.ndarray)
    # eigenface *= eigenvalue
    eigenface.shape = image_size[::-1]
    min = eigenface.min()
    max = eigenface.max()
    cv2.imshow('eigenface1', ((eigenface - min) / (max - min)).astype(numpy.float32))
    eigenface = eig_vectors[1]
    assert isinstance(eigenface, numpy.ndarray)
    eigenface.shape = image_size[::-1]
    min = eigenface.min()
    max = eigenface.max()
    cv2.imshow('eigenface2', ((eigenface - min) / (max - min)).astype(numpy.float32))

def show_reconstructed(eig_vectors, train_average_face, weights):
    # Show reconstructed face:
    list_of_reconstructed = []
    #Reconstruct 4 first images
    for weight in weights[:4]:
        reconstructed = numpy.dot(weight, eig_vectors)
        reconstructed += train_average_face
        reconstructed.shape = image_size[::-1]
        list_of_reconstructed.append(reconstructed)
    reconstructed_faces = numpy.concatenate(list_of_reconstructed, axis=0)
    cv2.imshow('reconstructed', reconstructed_faces)

def randomly_assign_images(no_of_persons, samples_person, samples_training):
    training_image_numbers = []
    testing_image_numbers = []
    for person_no in range(no_of_persons):
        random_permutation = random.sample(range(samples_person), samples_person)
        training_image_numbers.extend([(person_no, sample_no) for sample_no in random_permutation[:samples_training]])
        testing_image_numbers.extend([(person_no, sample_no) for sample_no in random_permutation[samples_training:]])

    return training_image_numbers, testing_image_numbers


def load_all_face_vectors_from_disk(no_of_persons, samples_person):
    all_image_numbers = []
    for pers_no in range(no_of_persons):
        for sample_no in range(samples_person):
            all_image_numbers.append((pers_no, sample_no))
    all_face_vectors = load_face_vectors_from_disk(all_image_numbers, image_size)
    return all_face_vectors


def load_face_vectors_from_disk(image_numbers, image_size):
    """
    Loads face vectors from disk and places them in the dictionary

    :param image_numbers:
    :param image_size:
    :return: dictionary of face vectors
    """
    face_vectors = {}
    for count, nums in enumerate(image_numbers):
        person_no, sample_no = nums
        face_vector = load_face_vector(
            "./ImageDatabase/f_{}_{:02}.JPG".format(person_no + 1, sample_no + 1),
            image_size
        )
        face_vectors[nums] = face_vector
    return face_vectors


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


def load_face_vector(path, image_size):
    """
    Loads image file from path supplied, extracts a face from it, resizes it to 'image_size', turns it into grayscale
    and then into a float vector.

    :param path: path of the image
    :param image_size: target size of the image before turning it to a vector.
    """
    print("Loading \"{}\" ...".format(path))
    image = cv2.imread(filename=path, flags=0)
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(500, 500)
    )

    copy = image.copy()

    if len(faces) == 1:
        x, y, w, h = faces[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 40)
    else:
        raise Exception("Face detection failed")

    cropped = copy[y:y+h, x:x+w]
    resized = cv2.resize(cropped, image_size)
    vector = numpy.ravel(resized)

    cv2.imshow('cropped image', cropped)
    cv2.imshow('original image', image)
    cv2.waitKey(20)
    vector = vector.astype(numpy.float64, copy=False) / 255

    return vector


if __name__ == '__main__':
    main()



"""
        answer = None
    while answer is None:
        answer = input("Load cached eigenvectors and values? [Y/n]")
        if answer == "n" or answer == "N":
            print("Caching eigenvalues and eigenvectors...")
            numpy.save("CachedAverageFace.npy", train_average_face)
            numpy.save("CachedEigenValues.npy", eig_values)
            numpy.save("CachedEigenVectors.npy", eig_vectors)
        elif answer == "y" or answer == "Y" or answer == "":
            print("Loading cached eigenvalues and eigenvectors...")
            train_average_face = numpy.load("CachedAverageFace.npy")
            eig_values = numpy.load("CachedEigenValues.npy")
            eig_vectors = numpy.load("CachedEigenVectors.npy")
        else:
            answer = None
    """

    #print("eigenvector.shape:", eig_vectors.shape)
    #print("eigenvalues.shape:", eig_values.shape)
    #print("Eigenvalues:\n", eig_values)
