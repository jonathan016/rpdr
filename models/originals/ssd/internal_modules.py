class PriorBoxesConfig:
    """Encapsulates prior boxes configuration for SSD model with VGG-16 backbone.

    Rather than passing dictionaries of configuration, this class encapsulates the configuration for prior boxes to
    be easily set and more verbose by implementing this class as a derivation of the builder pattern.
    """

    def __init__(self, config: dict = None):
        self.__config = {'conv4_3': 4,
                         'conv7': 6,
                         'conv8_2': 6,
                         'conv9_2': 6,
                         'conv10_2': 4,
                         'conv11_2': 4}

        if config: self.set_config(config)

    def set_config(self, config: dict):
        assert config.keys() == self.__config.keys()

        self.__config = config

    def value(self):
        return self.__config

    def all(self, item):
        self.conv4_3(item)
        self.conv7(item)
        self.conv8_2(item)
        self.conv9_2(item)
        self.conv10_2(item)
        self.conv11_2(item)

        return self

    def conv4_3(self, item):
        self.__config['conv4_3'] = item
        return self

    def conv7(self, item):
        self.__config['conv7'] = item
        return self

    def conv8_2(self, item):
        self.__config['conv8_2'] = item
        return self

    def conv9_2(self, item):
        self.__config['conv9_2'] = item
        return self

    def conv10_2(self, item):
        self.__config['conv10_2'] = item
        return self

    def conv11_2(self, item):
        self.__config['conv11_2'] = item
        return self
