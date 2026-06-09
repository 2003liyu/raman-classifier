import torch
import numpy as np
import pandas as pd
import os

from captum.attr import DeepLift


class DeepLiftVerifier:
    def __init__(self, classifier):

        self.classifier = classifier
        self.device = classifier.device


        self.interpreted_model = classifier.get_model_for_interpretation()


        self.interpreted_model.eval()



    def generate_report(self, raw_data_np, labels_np, target_class_id, raman_shift):

        indices = np.where(labels_np == target_class_id)[0]
        if len(indices) == 0:
            print(f"❌ 错误: 找不到类别 ID 为 {target_class_id} 的样本数据。")
            return None

        class_data = raw_data_np[indices]


        input_dim = class_data.shape[1]
        expected_dim = len(raman_shift)
        assert input_dim == expected_dim, \
            f"维度不匹配！输入特征({input_dim}) != 波数长度({expected_dim})。请检查是否误传了降维数据。"


        inputs = torch.tensor(class_data, dtype=torch.float32).to(self.device)
        inputs.requires_grad_(True)


        baseline = torch.zeros_like(inputs)
        dl = DeepLift(self.interpreted_model)

        try:

            attributions = dl.attribute(inputs, baselines=baseline, target=target_class_id)


            weights = attributions.detach().cpu().numpy().mean(axis=0)
            avg_raw_spectrum = class_data.mean(axis=0)


            max_val = np.max(np.abs(weights)) + 1e-9
            norm_weights = weights / max_val


            self._save_to_csv(raman_shift, norm_weights, avg_raw_spectrum, target_class_id)
            return norm_weights

        except Exception as e:
            print(f" DeepLIFT 计算出错: {e}")
            return None

    def _save_to_csv(self, raman_shift, weights, avg_raw, class_id):

        label_name = self.classifier.data_manager.label_of(class_id)
        model_name = self.classifier.base_name


        weights = weights.flatten()
        avg_raw = avg_raw.flatten()
        raman_shift = np.array(raman_shift).flatten()

        assert len(raman_shift) == len(weights) == len(avg_raw), \
            f"保存失败：维度不一致！Raman:{len(raman_shift)}, Weights:{len(weights)}"

        df = pd.DataFrame({
            'Raman_Shift': raman_shift,
            'Contribution_Weight': weights,
            'Average_Intensity': avg_raw
        })

        filename = f"DL_gMLP_KD_Aug_{model_name}_{label_name}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ 报告已生成: {filename}")