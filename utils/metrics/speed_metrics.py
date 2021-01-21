from datetime import datetime


def time_difference(callback: callable, arguments):
    start = datetime.now()
    result = callback(*arguments)
    return result, datetime.now() - start


def operation_per_second(callback: callable, arguments):
    result, time_diff = time_difference(callback, arguments)
    return result, 1 / (time_diff.seconds + (time_diff.microseconds / 1e6))
