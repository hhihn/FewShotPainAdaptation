class LosoTrainingRunner:
    """Internal owner for the LOSO training workflow.

    The current workflow is retained on the learner as a private implementation while
    the public facade delegates here. The runner exists as the orchestration boundary
    and can absorb the remaining loop code without changing the public API.
    """

    def __init__(self, learner):
        self.learner = learner

    def train(
        self,
        training_progress_output_dir: str = "outputs/training_progress",
        save_model_architecture_first_run: bool = True,
        model_architecture_output_path: str = "outputs/model_architecture/model_summary.txt",
    ):
        return self.learner._run_loso_training_workflow(
            training_progress_output_dir=training_progress_output_dir,
            save_model_architecture_first_run=save_model_architecture_first_run,
            model_architecture_output_path=model_architecture_output_path,
        )
