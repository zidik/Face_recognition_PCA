__author__ = 'Mark Laane'

import cv2
import random
import sys

import numpy
import matplotlib.pyplot as plt


image_size = (200, 200)  # All face images will be resized to this


def main():
    no_of_persons = 13  # Number of persons
    samples_person = 10  # Number of samples per person

    cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('cropped image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original image', 200, 140)
    cv2.moveWindow('original image', 10, 10)
    cv2.moveWindow('cropped image', 250, 10)

    all_image_numbers = []
    for pers_no in range(no_of_persons):
        for sample_no in range(samples_person):
            all_image_numbers.append((pers_no, sample_no))
    all_face_vectors = load_face_vectors_from_disk(all_image_numbers, image_size)

    while True:
        function = input(
            "0)Exit\n"
            "1)Test variating number of training samples\n"
            "2)Test variating number of principal components\n"
            "3)Live test\n"
            "4)Test image \"test.JPG\"\n"
            "\n"
            "Choose function:"
        )
        if function == "1":
            test_different_training(all_face_vectors, no_of_persons, samples_person)
        elif function == "2":
            test_different_number_of_principal_components(all_face_vectors, no_of_persons, samples_person, 3)
        elif function == "3":
            test_live(all_face_vectors, all_image_numbers, samples_person)
        elif function == "4":
            test_one_image(all_face_vectors, all_image_numbers, samples_person)
        elif function == "0":
            return


def test_different_training(all_face_vectors, no_of_persons, samples_person):
    plot_number_of_training_samples = []
    plot_recognition_rate = []

    number_of_tests = 30
    for test_no in range(number_of_tests):
        sys.stdout.write("\r%d%%" % (test_no * 100 // number_of_tests))
        sys.stdout.flush()
        for samples_training in range(1, samples_person):
            result = train_and_test(all_face_vectors, no_of_persons, samples_person, samples_training)
            # print("Training samples:{} result:{}".format(samples_training, result))

            plot_number_of_training_samples.append(samples_training)
            plot_recognition_rate.append(result)

    print()

    # Plot results:
    plot_results(plot_number_of_training_samples, plot_recognition_rate, 1, samples_person - 1)


def test_different_number_of_principal_components(all_face_vectors, no_of_persons, samples_person, samples_training):
    plot_number_of_principal_components = []
    plot_recognition_rate = []
    number_of_tests = 30
    for test_no in range(number_of_tests):
        sys.stdout.write("\r%d%%" % (test_no * 100 // number_of_tests))
        sys.stdout.flush()
        for number_of_PCs in range(1, samples_person):
            result = train_and_test(all_face_vectors, no_of_persons, samples_person, samples_training, number_of_PCs)
            # print("Principal components:{} result:{}".format(number_of_PCs, result))

            plot_number_of_principal_components.append(number_of_PCs)
            plot_recognition_rate.append(result)
    print()

    # Plot results:
    plot_results(plot_number_of_principal_components, plot_recognition_rate, 1, samples_person - 1)


def test_live(all_face_vectors, all_image_numbers, samples_person):
    # Load training faces:
    train_faces = choose_face_vectors(all_face_vectors, all_image_numbers)
    training_results = train(train_faces, None)

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            continue  # Let's just try again
        cv2.imshow('video', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            person_no, sample_no = test_one(training_results, frame, samples_person)
        except UserWarning:
            print("No face was found..")
        else:
            display_match(all_face_vectors, person_no, sample_no)

        key_no = cv2.waitKey(30) & 0xFF
        if key_no == ord('q'):
            print("Quitting...")
            break
    cap.release()


def test_one_image(all_face_vectors, all_image_numbers, samples_person):
    # Load training faces:
    train_faces = choose_face_vectors(all_face_vectors, all_image_numbers)
    training_results = train(train_faces, None)

    path = "./test.JPG"
    print("Loading \"{}\" ...".format(path))
    image = cv2.imread(filename=path, flags=0)
    person_no, sample_no = test_one(training_results, image, samples_person)
    display_match(all_face_vectors, person_no, sample_no)
    cv2.waitKey(20)


def test_one(training_results, image, samples_person):
    train_average_face, eig_vectors, train_weights = training_results
    face_vector = load_face_vector(image, image_size)
    # Hacky way to reuse code:
    test_face = choose_face_vectors({"1": face_vector, "2": face_vector})
    test_diff = test_face - train_average_face
    test_weights = numpy.dot(eig_vectors, test_diff.T).T
    closest_target_no, closest_distance = test_distance(test_weights[0], train_weights)
    calculated_class = closest_target_no // samples_person
    print("Class:{}".format(calculated_class))
    person_no = closest_target_no // samples_person
    sample_no = closest_target_no % samples_person
    return person_no, sample_no


def display_match(all_face_vectors, person_no, sample_no):
    match = all_face_vectors[(person_no, sample_no)].copy()
    match.shape = image_size
    cv2.imshow('MATCH', match)


def plot_results(x_axis, y_axis, x_min, x_max):
    # Plot datapoints
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis, 'ro', alpha=0.3, label='Datapoints')

    # Plot mean
    plot_mean_y = []
    for group_no in range(x_max - x_min + 1):
        group = y_axis[group_no::x_max - x_min + 1]
        mean = sum(group) / len(group)
        plot_mean_y.append(mean)
    ax.plot(x_axis[:x_max - x_min + 1], plot_mean_y, label='Mean', linestyle='--')

    ax.legend(loc='lower right')
    ax.axis([x_min - 1, x_max + 1, 0, 1])
    plt.show()


def train(train_faces, principal_components=None):
    # Calculate the average face
    train_average_face = train_faces.mean(axis=0)
    #Calculate Eigenvectors
    train_diff = train_faces - train_average_face
    eig_values, eig_vectors = numpy.linalg.eig(numpy.dot(train_diff, train_diff.transpose()))
    eig_vectors = numpy.dot(eig_vectors, train_diff)

    if principal_components is not None:
        # Sort eigenvectors and eigenvalues by eigenvalues
        idx = numpy.argsort(-eig_values)
        #eig_values = eig_values[idx]
        eig_vectors = eig_vectors[idx]
        #Take only top "principal_components" of eigenvectors
        #eig_values = eig_values[:principal_components]
        eig_vectors = eig_vectors[:principal_components]

    #Calculate weights for each image
    train_weights = numpy.dot(eig_vectors, train_diff.T).T

    return train_average_face, eig_vectors, train_weights


def train_and_test(all_face_vectors, no_of_persons, samples_person, samples_training, principal_components=None):
    train_img_no, test_img_no = randomly_assign_images(no_of_persons, samples_person, samples_training)
    # Load training faces:
    train_faces = choose_face_vectors(all_face_vectors, train_img_no)

    train_average_face, eig_vectors, train_weights = train(train_faces, principal_components)

    # Load images for testing
    test_faces = choose_face_vectors(all_face_vectors, test_img_no)
    test_diff = test_faces - train_average_face
    test_weights = numpy.dot(eig_vectors, test_diff.T).T

    successes = 0
    tries = 0
    for point, img_no in zip(test_weights, test_img_no):
        closest_target_no, closest_distance = test_distance(point, train_weights)
        #print("Distance:", closest_distance)

        real_class, _ = img_no
        calculated_class = closest_target_no // samples_training
        success = real_class == calculated_class
        #print("Success:{} real:{}, calculated:{}".format(success, real_class, calculated_class))
        if success:
            successes += 1
        tries += 1

    return successes / tries


def choose_face_vectors(all_face_vectors, image_numbers=None):
    if image_numbers is None:
        no_of_vectors = len(all_face_vectors)
    else:
        no_of_vectors = len(image_numbers)

    chosen_face_vectors = numpy.zeros(
        shape=(no_of_vectors, image_size[0] * image_size[1]),
        dtype=numpy.float64
    )
    if image_numbers is None:
        for count, face_vector in enumerate(all_face_vectors.values()):
            chosen_face_vectors[count] = face_vector
    else:
        for count, number in enumerate(image_numbers):
            chosen_face_vectors[count] = all_face_vectors[number]
    return chosen_face_vectors


def test_distance(point, targets):
    closest_target_no = None
    closest_distance = float("inf")
    for count, target in enumerate(targets):
        distance = numpy.sqrt(numpy.sum((point - target) ** 2))
        if distance < closest_distance:
            closest_distance = distance
            closest_target_no = count
    return closest_target_no, closest_distance


def randomly_assign_images(no_of_persons, samples_person, samples_training):
    training_image_numbers = []
    testing_image_numbers = []
    for person_no in range(no_of_persons):
        random_permutation = random.sample(range(samples_person), samples_person)
        training_image_numbers.extend([(person_no, sample_no) for sample_no in random_permutation[:samples_training]])
        testing_image_numbers.extend([(person_no, sample_no) for sample_no in random_permutation[samples_training:]])

    return training_image_numbers, testing_image_numbers


def load_face_vectors_from_disk(image_numbers, img_size):
    """
    Loads face vectors from disk and places them in the dictionary

    :param image_numbers:
    :param img_size:
    :return: dictionary of face vectors
    """
    face_vectors = {}
    for count, nums in enumerate(image_numbers):
        person_no, sample_no = nums
        path = "./ImageDatabase/f_{}_{:02}.JPG".format(person_no + 1, sample_no + 1)
        print("Loading \"{}\" ...".format(path))
        image = cv2.imread(filename=path, flags=0)
        face_vector = load_face_vector(image, img_size)
        face_vectors[nums] = face_vector
    return face_vectors


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


def load_face_vector(image, img_size):
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(image.shape[1] // 10, image.shape[0] // 10)
    )

    copy = image.copy()

    if len(faces) == 1:
        x, y, w, h = faces[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 40)
    else:
        raise UserWarning("Face detection failed")

    cropped = copy[y:y + h, x:x + w]
    resized = cv2.resize(cropped, img_size)
    vector = numpy.ravel(resized)

    cv2.imshow('cropped image', cropped)
    cv2.imshow('original image', image)
    cv2.waitKey(20)
    vector = vector.astype(numpy.float64, copy=False) / 255

    return vector


if __name__ == '__main__':
    main()