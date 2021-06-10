import os
import logging
logs = set()

# LOGGER
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"

COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        msg = record.msg
        levelname = record.levelname
        if self.use_color and levelname in COLORS and COLORS[levelname] != WHITE:
            if isinstance(msg, str):
                msg_color = COLOR_SEQ % (30 + COLORS[levelname]) + msg + RESET_SEQ
                record.msg = msg_color
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def init_log(name, level=logging.DEBUG):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    FORMAT = f'[%(levelname)s]%(asctime)s-%(filename)s#%(lineno)d:%(message)s'
    formatter = ColoredFormatter(FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class cprint(object):
    RED = '\033[31m'       
    GREEN = '\033[32m'     
    YELLOW = '\033[33m'    
    BLUE = '\033[34m'      
    FUCHSIA = '\033[35m'   
    CYAN = '\033[36m'      
    WHITE = '\033[37m'     
    # default color
    RESET = '\033[0m'      
 
    @staticmethod
    def color_str(color, s):
        return '{}{}{}'.format(
            getattr(cprint, color),
            s,
            cprint.RESET
        )
 
    @staticmethod
    def red(s):
        print(cprint.color_str('RED', s))
 
    @staticmethod
    def green(s):
        print(cprint.color_str('GREEN', s))
 
    @staticmethod
    def yellow(s):
        print(cprint.color_str('YELLOW', s))
 
    @staticmethod
    def blue(s):
        print(cprint.color_str('BLUE', s))

    @staticmethod
    def fuchsia(s):
        print(cprint.color_str('FUCHSIA', s))
 
    @staticmethod
    def cyan(s):
        print(cprint.color_str('CYAN', s))
    
    @staticmethod
    def white(s):
        print(cprint.color_str('WHITE', s))

def main():
    cprint.red('Hello Test')
    cprint.green('Hello Test')
    cprint.yellow('Hello Test')
    cprint.blue('Hello Test')
    cprint.fuchsia('Hello Test')
    cprint.cyan('Hello Test')
    cprint.white('Hello Test')


if __name__ == '__main__':
    main()


