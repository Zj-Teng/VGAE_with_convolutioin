import logging
import colorlog


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class _LoggerFactory(metaclass=_Singleton):
    # 日志颜色配置
    log_colors_config = {
        'DEBUG': 'white',  # cyan white
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    # 日志文件输出格式
    file_formatter = logging.Formatter(
        fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S'
    )

    # 日志控制台输出格式
    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s '
            'line:%(lineno)d [%(levelname)s] :\n %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S',
        log_colors=log_colors_config
    )

    @staticmethod
    def get_logger(**kwargs):
        """工厂方法，产生logging对象，并设置相应的输出等级和格式

        :param kwargs:
        :return: longging对象
        """
        k_len = len(kwargs)
        mapper = {
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        if k_len >= 2:
            logger = logging.getLogger(kwargs.get('name'))
            console_handler = logging.StreamHandler()
            level = mapper.get(kwargs.get('level'))

            if level is not None:
                console_handler.setLevel(level)
                logger.setLevel(level)
                console_handler.setFormatter(_LoggerFactory.console_formatter)
                logger.addHandler(console_handler)
            else:
                raise ArgsError('参数名称错误')

            if k_len == 5:
                filename = kwargs.get('file')
                mode = kwargs.get('mode')
                encoding = kwargs.get('encoding')
                print((filename, mode, encoding))

                if filename and mode and encoding is not None:
                    file_handler = logging.FileHandler(
                        filename=filename, mode=mode, encoding=encoding
                    )
                    file_handler.setLevel(level)
                    file_handler.setFormatter(_LoggerFactory.file_formatter)

                    logger.addHandler(file_handler)
                else:
                    raise ArgsError('参数名称错误')
        else:
            raise ArgsError('参数数量错误')

        return logger


class Logger(object):
    def __init__(self, **kwargs):
        logger_factory = _LoggerFactory()
        self.logger = logger_factory.get_logger(**kwargs)

    @property
    def get_logger(self):
        return self.logger


class ArgsError(Exception):
    def __init__(self, message):
        self.message = message
