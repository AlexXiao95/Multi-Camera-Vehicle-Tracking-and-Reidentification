__author__ = 'alex'

from log_loader import *
from progress_visulization import *
from matplotlib import animation

class animate_monitor():

    def __init__(self, log_files, fig):
        """
        Initialize and record log list
        :param log_files:
        :return:
        """
        self.log_files = log_files
        self.fig = fig
        self.axes = []

    def start(self):
        for i in xrange(1,3):
            self.axes.append(self.fig.add_subplot(2,1,i, sharex=self.axes[0] if i>1 else None))

        return self.show_value()

    def show_value(self, fd=None):
        """
        Internal function to read updated progress
        :return:
        """
        numbers = select_log_part(load_log(self.log_files))
        y0, y1 = draw_loss(numbers, self.axes[0])
        y2 = draw_acc(numbers, self.axes[1])
        # return self.fig

if __name__ == "__main__":
    log_files = [
        # '../models/googlenet/log/oct10_0_20000.22550',
        # '../models/googlenet/log/oct10_20000_35000.28025',
        # '../models/googlenet/log/oct11_35000_70000.5614',
        # '../models/googlenet/log/oct11_70000_85000.26819',
        # '../models/googlenet/log/oct12_85000_105000.24475',
        '../models/googlenet/log/caffe.mmlab-107.alex.log.INFO.20141013-173803.22149'
    ]

    fig = pyplot.figure(num=1, figsize=(15,9))

    am = animate_monitor(log_files, fig)

    anime = animation.FuncAnimation(fig, am.show_value, blit=False, interval=20000, init_func=am.start)

    pyplot.show()



