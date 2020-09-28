# coding=utf-8

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(module)s %(levelname)s: %(message)s'
    )

# logging.basicConfig(
#     level=logging.INFO,
#     filename=LOGGER_NAME+'.log',
#     filemode='a',
#     format='%(asctime)s %(module)s %(levelname)s: %(message)s'
#     )

logger = logging.getLogger('CIKM2020')
