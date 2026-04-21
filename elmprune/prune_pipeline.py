import gc
import torch
from pathlib import Path
from tqdm.auto import tqdm
from .utils import find_best_val_loss_files, mirror_copy_files, collect_infos, get_and_load_model
from .utils import get_first_dataloader_image, save_pruned_model, get_val_dataloader_fold
from . import ELMImportanceProcessor, ImportanceProcessorConfig, PruneConfig, PruneProcessor, PruneVerboseLevel

class PrunePipeline:

    pruned_suffix = "-pruned"
    prune_percentages_default = [0.2, 0.4, 0.6, 0.8]
    supported_prune_types = ["elm", "mag", "random", "tests"]

    def __init__(self, dataloaders, models_root_path, task, subset, prune_percentages=None, prune_mode='elm'):
        self.src_root = Path(models_root_path) / task / subset
        self.dst_root = Path(models_root_path) / (task + self.pruned_suffix) / subset
        self.src_files = find_best_val_loss_files(self.src_root)
        self.prune_percentages = self.prune_percentages_default if prune_percentages is None else prune_percentages
        self.dataloaders = dataloaders
        self.prune_mode = prune_mode

        if prune_mode not in self.supported_prune_types:
            raise Exception("Prune type currently not supported!")

        if prune_mode in ["mag", "random"]:
            raise Exception(f"Prune type {prune_mode} will be implemented in future!")

    def execute(self):
        mirror_copy_files(self.src_root, self.dst_root)
        dst_infos = collect_infos(self.dst_root)

        for dst_file_model in tqdm(dst_infos, desc="Pruning runner", position=0):
            dense_model = get_and_load_model(
                dst_file_model["model"],
                dst_file_model["backbone"],
                dst_file_model["full_path"]
            )
            dataloader = get_val_dataloader_fold(self.dataloaders, dst_file_model["fold"])
            input_example = get_first_dataloader_image(dataloader)

            if self.prune_mode == 'elm':
                importances_dict = self.__get_importances_elm_prune(dense_model, dataloader)
            elif self.prune_mode == "tests":
                importances_dict = self.__get_importances_tests()

            for importance_type, importance_values in importances_dict.items():
                selection_scope = "global" if importance_type == "elm_global" else "local"

                for target_param_reduction in tqdm(self.prune_percentages, desc="Target param reduction", position=0):
                    print(f"Prune for {importance_type} and target reduction {target_param_reduction:.2f}")

                    cfg = PruneConfig(
                        importance_type=importance_type,
                        target_param_reduction=target_param_reduction,
                        selection_scope=selection_scope,
                        min_channels_abs=8,
                        min_keep_ratio=0.20,
                        max_layer_prune_ratio=0.8,
                        per_step_layer_ratio=0.08,
                        round_to=8,
                        verbose=PruneVerboseLevel.BASIC,
                    )

                    prune_processor = PruneProcessor(
                        dense_model,
                        importance_values,
                        input_example,
                        cfg
                    )

                    pruned_model = prune_processor.execute()

                    path_out = Path(dst_file_model["full_path"]).resolve().parent
                    print("Final params:", sum(p.numel() for p in pruned_model.parameters()))
                    save_pruned_model(pruned_model, input_example, path_out, importance_type, target_param_reduction)

            dense_model = None
            gc.collect()
            torch.cuda.empty_cache()

    def __get_importances_elm_prune(self, model, dataloader):
        print("Extracting features for ELM...")
        elm_importance_processor = ELMImportanceProcessor(ImportanceProcessorConfig(), model, dataloader)
        print("Getting importances for layerwise...")
        layerwise_importances = elm_importance_processor.compute_elm_layerwise_importances()
        print("Getting importances for filterwise...")
        filterwise_importances = elm_importance_processor.compute_elm_filterwise_importances()
        print("Getting importances for global...")
        global_importances = elm_importance_processor.compute_elm_global_importances()
        return  {
                    "elm_layerwise": layerwise_importances, 
                    "elm_filterwise": filterwise_importances, 
                    "elm_global": global_importances
                }
    
    def __get_importances_tests(self):
        from elmprune.utils import load_dict
        print("Getting importances for test...")
        importances = load_dict(Path("test_dict.json"))
        return  {
                    "tests": importances,
                }