import tensorflow as tf


class BaseOptimizer:
    def __init__(self, sess, model, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess

    def optimize_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the optimization step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def optimize_step(self):
        """
        implement the logic of the optimization step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
