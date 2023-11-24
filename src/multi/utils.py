"""common helpers"""

from pymor.core.logger import getLogger


class LogMixin(object):
    @property
    def logger(self):
        name = self.__class__.__module__
        return getLogger(name)
