class Trainer(object):
    """Responsible for the wholet raining process."""
    def __init__(self, environment, actor, critic):
        super(Trainer, self).__init__()
        