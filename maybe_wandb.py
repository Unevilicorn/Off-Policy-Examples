import wandb

def get_wandb(get_real = False):
    if get_real:
        return wandb
    else:
        class FakeWandb:
            @staticmethod
            def init(*args, **kwargs):
                pass
            @staticmethod
            def log(*args, **kwargs):
                pass
            @staticmethod
            def finish(*args, **kwargs):
                pass
        return FakeWandb