# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os

import aim
from semilearn.core.utils import get_logger

from .hook import Hook


class AimHook(Hook):
    """
    A hook for tracking training progress with Aim.
    """

    def before_run(self, algorithm):
        """Setup the Aim tracking. Either create a new run or resume an existing one if
        the algorithm has a run hash.

        Args:
            algorithm (AlgorithmBase): The training algorithm.
        """
        # Initialize run
        name = algorithm.save_name
        project = algorithm.save_dir.split("/")[-1]
        repo = os.path.join(algorithm.args.save_dir, "aim", algorithm.args.save_name)

        logger = get_logger(
            name=algorithm.args.save_name,
            save_path=os.path.join(algorithm.args.save_dir, algorithm.args.save_name),
            level="INFO",
        )

        if hasattr(algorithm, "aim_run_hash"):
            # Resume tracking
            self.run = aim.Run(
                run_hash=algorithm.aim_run_hash,
                repo=repo,
            )
            logger.info(f"Resuming tracking of Run {algorithm.aim_run_hash}")
        else:
            # Start tracking a new run
            self.run = aim.Run(
                experiment=name,
                repo=repo,
                log_system_params=True,
            )
            algorithm.aim_run_hash = self.run.hash
            logger.info(f"Tracking new run, Run {algorithm.aim_run_hash}")

        # Save configuration
        self.run["hparams"] = algorithm.args.__dict__

        # Set tags
        benchmark = f"benchmark: {project}"
        dataset = f"dataset: {algorithm.args.dataset}"
        data_setting = "setting: {}_lb{}_{}_ulb{}_{}".format(
            algorithm.args.dataset,
            algorithm.args.num_labels,
            algorithm.args.lb_imb_ratio,
            algorithm.args.ulb_num_labels,
            algorithm.args.ulb_imb_ratio,
        )
        alg = f"alg: {algorithm.args.algorithm}"
        imb_alg = f"imb_alg: {algorithm.args.imb_algorithm}"
        self.run.add_tag(benchmark)
        self.run.add_tag(dataset)
        self.run.add_tag(data_setting)
        self.run.add_tag(alg)
        self.run.add_tag(imb_alg)

    def after_train_step(self, algorithm):
        """Log the metric values in the log dictionary to Aim.

        Args:
            algorithm (AlgorithmBase): The training algorithm.
        """
        if self.every_n_iters(algorithm, algorithm.num_log_iter):
            for key, item in algorithm.log_dict.items():
                self.run.track(item, name=key, step=algorithm.it)

        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            self.run.track(
                algorithm.best_eval_acc, name="eval/best-acc", step=algorithm.it
            )
