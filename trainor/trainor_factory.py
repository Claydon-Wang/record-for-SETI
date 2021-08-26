import logging
from .classification import Classification,DSAN


class TrainorFacotry(object):
    """
    """
    def __init__(self):
        self.trainor_table = {
            "classification": Classification,
            "DSAN": DSAN,
        }

    def get_trainor(self, cfg, save_model_name):
        trainor_class = self.trainor_table.get(cfg.NAME)
        trainor_instance = trainor_class(cfg, save_model_name=save_model_name)
        logging.info(f"init {cfg.NAME} trainor success")
        return trainor_instance
         
