__author__ = 'alex'

from matplotlib import pyplot


def get_value_coord(diction, value_idx=0):

    coord = []
    value = []
    for k in sorted(diction):
        if len(diction[k])<= value_idx:
            continue
        coord.append(k)
        value.append(diction[k][value_idx])

    return coord,value

def draw_loss(numbers, axe=None):
    """
    The function for drawing loss curve
    :param numbers:
    :return:
    """

    training_loss = numbers['Training']['loss']
    testing_loss = numbers['Testing']['loss']

    training_loss_iter, training_loss_total_value = get_value_coord(training_loss, 3)
    testing_loss_iter, testing_loss_total_value = get_value_coord(testing_loss, 2)




    if axe is None:
        axe = pyplot
    return axe.plot(training_loss_iter, training_loss_total_value), axe.plot(testing_loss_iter, testing_loss_total_value)

def draw_acc(numbers, axe=None):
    """
    Draw testing accuracy curve
    :param numbers:
    :return:
    """

    testing_acc = numbers['Testing']['accuracy']

    testing_acc_iter, testing_acc_value = get_value_coord(testing_acc,0)


    if axe is None:
        axe = pyplot
    return axe.plot(testing_acc_iter, testing_acc_value)

    # pyplot()


def draw_both(numbers):
    fig = pyplot.figure(num=1, figsize=(15,9))

    ax = fig.add_subplot(2, 1, 1)
    draw_loss(numbers, ax)
    ax = fig.add_subplot(2, 1, 2)
    draw_acc(numbers, ax)
    pyplot.show()


if __name__ == '__main__':

    from log_loader import *

    log_files = [
        '../models/googlenet/log/oct10_0_20000.22550',
        '../models/googlenet/log/oct10_20000_35000.28025',
        '../models/googlenet/log/oct11_35000_70000.5614',
        '../models/googlenet/log/oct11_70000_85000.26819',
        # '../models/googlenet/log/oct11_85000_100000.2021',
        '../models/googlenet/log/caffe.mmlab-107.alex.log.INFO.20141012-195207.1957'
    ]

    file_contents = load_log(log_files)

    numbers = select_log_part(file_contents)


    draw_both(numbers)
    # draw_loss(numbers)
    # draw_acc(numbers)
