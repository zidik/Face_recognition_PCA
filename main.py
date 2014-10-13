__author__ = 'Mark'
import cv2
import numpy
import random


def main():
    image_size = (81, 54)   # All images will be resized to this
    no_of_persons = 1       # Number of persons
    samples_person = 10     # Number of samples per person
    samples_training = 4    # Number of samples used for training

    faces_training = numpy.zeros(
        (no_of_persons, samples_training, image_size[0]*image_size[1]),
        dtype=numpy.uint8
    )
    faces_test = numpy.zeros(
        (no_of_persons, samples_person - samples_training, image_size[0]*image_size[1]),
        dtype=numpy.uint8
    )

    load_face_images(
        faces_training, faces_test,
        no_of_persons, samples_person, samples_training, image_size
    )
    for person in range(no_of_persons):
        T = faces_training[person] - faces_training[person].mean()
        assert isinstance(T, numpy.ndarray)
        S = numpy.dot(T.transpose(), T)
        w, v= numpy.linalg.eig(S)
        

#difference = resized  - resized .mean()
#print(difference.mean())
#numpy.linalg.eig(difference*difference.T)


def load_face_images(
        faces_training, faces_test,
        no_of_persons, samples_person, samples_training, image_size
    ):
    for person_no in range(no_of_persons):
        for count, sample_no in enumerate(random.sample(range(samples_person), samples_person)):
            if count < samples_training:
                target_array = faces_training
            else:
                target_array = faces_test
                count -= samples_training
            target_array[person_no][count] = load_face_vector("./ImageDatabase/f_{}_{:02}.JPG".format(person_no+1, sample_no+1))


def load_face_vector(path):
    """
    Loads image file from path supplied, resizes it, turns it into grayscale and then into a vector.

    :param path: path of the image
    """
    print("Loading \"{}\" ...".format(path))
    original_image = cv2.imread(path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    vector = numpy.ravel(gray_image)
    resized = vector[::64*64]
    return resized




"""
cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
cv2.namedWindow('grayscale image', cv2.WINDOW_NORMAL)
cv2.imshow('original image', original_image)
cv2.imshow('grayscale image', gray_image)

win_height, win_width = tuple(element//10 for element in gray_image.shape)
cv2.resizeWindow('original image', win_width, win_height)
cv2.resizeWindow('grayscale image', win_width, win_height)


"""



if __name__ == '__main__':
    main()


