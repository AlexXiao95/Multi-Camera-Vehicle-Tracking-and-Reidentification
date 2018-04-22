__author__ = 'alex'

import re
import itertools


def load_log(log_names):
    raw_logs = []
    for log_name in log_names:
        with open(log_name) as input_file:
            lines = input_file.readlines()
            raw_logs.append(lines)

    return raw_logs


def select_log_part(raw_logs, part_identifiers=[('Testing','Testing', 4, 2), ('Training','loss', 4, 0)]):
    """
    A helper function for loading the part of log we are interested in
    :param part_identifiers:
    :return:
    """

    # # Remove initialization logs
    iteration_locator = re.compile('Iteration')
    real_logs = []
    for raw_log in raw_logs:
        temp = map(lambda x: iteration_locator.search(x), raw_log)
        if len([i for i, e in enumerate(temp) if e]) == 0:
            continue
        first_real_log_line = [i for i, e in enumerate(temp) if e][0]
        real_log = raw_log[first_real_log_line:]
        real_logs.append(real_log)

    log_list = list(itertools.chain(*real_logs))

    floating_number_detector = re.compile('=\s[0-9][0-9]*\.[0-9]+')

    iteration_line_locater = re.compile('Iteration ([0-9]+)')

    is_iteration_line = map(lambda x: iteration_line_locater.search(x), log_list)

    iterations = [(log_list[i], e.group(1), i) for i, e in enumerate(is_iteration_line) if e]

    floating_numbers = map(
        lambda x: floating_number_detector.search(x).group(0) if floating_number_detector.search(x) else None, log_list)

    numbers = {part: [] for part,_,  _, _ in part_identifiers}
    for line_str,line_iter, line_index in iterations:

        for part_name, part, loss_num, acc_num in part_identifiers:
            if part in line_str:
                max_num = loss_num + acc_num
                loss = []
                acc = []
                for i in xrange(max_num):
                    try:
                        line = log_list[line_index+i]
                    except IndexError:
                        break
                    if 'loss' in line:
                        if len(loss) <= loss_num:
                            loss.append(float(floating_numbers[line_index+i][2:]))

                    if 'accuracy' in line  or 'accuracy5' in line:
                        if len(acc) <= acc_num:
                            acc.append(float(floating_numbers[line_index+i][2:]))

                numbers[part_name].append((int(line_iter), loss, acc))

    for k, v in numbers.items():
        numbers[k] = {
            'loss': {obj[0]: obj[1] for obj in v},
            'accuracy': {obj[0]: obj[2] for obj in v}
        }

    return numbers


if __name__ == '__main__':
    log_files = ['../models/googlenet/log/oct10_0_20000.22550','../models/googlenet/log/oct10_20000_35000.28025']

    file_contents = load_log(log_files)

    read_log = select_log_part(file_contents)

    import pprint

    pprint.pprint(read_log)
