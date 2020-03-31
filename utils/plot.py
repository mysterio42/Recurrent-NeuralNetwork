import matplotlib.pyplot as plt


def display_digit(test_image, label_real, label_predicted):
    plt.title('Real Label: {}  Predicted Label {}'.format(label_real, int(label_predicted)))
    plt.imshow(test_image.reshape(28, 28))
    plt.show()
