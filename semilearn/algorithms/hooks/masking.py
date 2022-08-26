
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from semilearn.core.hooks import Hook


class MaskingHook(Hook):
    """
    Base MaskingHook, used for computing the mask of unalebeld (consistency) loss
    define MaskingHook in each algorithm when needed, and call hook inside each train_step
    easy support for other settings
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def masking(self, algorithm, logits):
        raise NotImplementedError