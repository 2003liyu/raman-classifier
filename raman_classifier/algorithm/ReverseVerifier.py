import torch
import numpy as np
import matplotlib.pyplot as plt


class PWMVerifier:
    def __init__(self, classifier_obj):

        self.classifier = classifier_obj
        self.model = classifier_obj.model
        self.device = classifier_obj.device

    def get_saliency_map(self, data: np.ndarray, target_class: int):

        self.model.eval()


        inputs = torch.from_numpy(data).float().to(self.device)
        inputs.requires_grad = True


        outputs = self.model(inputs)


        score = outputs[:, target_class].sum()


        self.model.zero_grad()
        score.backward()


        saliency = inputs.grad.data.abs().cpu().numpy()


        if saliency.ndim == 3:
            saliency = saliency.squeeze(1)

        return saliency

    def generate_pwm_report(self, data: np.ndarray, labels: np.ndarray, target_class_id: int, class_name: str):

        target_idx = np.where(labels == target_class_id)[0]

        if len(target_idx) == 0:
            raise ValueError(f"训练集中没有找到类别 ID 为 {target_class_id} 的样本")


        if len(target_idx) > 200:
            target_idx = target_idx[:200]

        sub_data = data[target_idx]


        saliency = self.get_saliency_map(sub_data, target_class=target_class_id)


        avg_saliency = np.mean(saliency, axis=0)


        pwm_weights = (avg_saliency - avg_saliency.min()) / (avg_saliency.max() - avg_saliency.min() + 1e-8)

        return pwm_weights