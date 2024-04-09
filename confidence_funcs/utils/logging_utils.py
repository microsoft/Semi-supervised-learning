import logging
import sys

def get_logger(log_file,stdout_redirect=False,level=logging.INFO):
    #log_file_name = os.path.basename(args.log_file).split(".")[0]+".log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file,filemode='w')
    #logging.propagate = False 
    logger = logging.getLogger('MyLogger')
    formatter = logging.Formatter(fmt='[%(asctime)s : %(levelname)-5s : %(module).10s : ] : %(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
    
    handler_file = logging.FileHandler(filename=log_file)
    logger.addHandler(handler_file)
    
    logger.setLevel(level)
    logger.propagate = False

    if(stdout_redirect):
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setFormatter(formatter)
    
    return logger 

def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.flush()
        handler.close()
    


