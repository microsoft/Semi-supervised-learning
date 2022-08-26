"""
# TODO: put ema part into hook

before_run
    self.ema = EMA(self.model, self.ema_m)
    self.ema.register()
    if self.resume == True:
        self.ema.load(self.ema_model)

after_train_step
    if algorithm.ema is not None:
        algorithm.ema.update()
"""

