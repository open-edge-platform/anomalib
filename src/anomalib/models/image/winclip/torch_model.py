# Original Code
# https://github.com/caoyunkang/WinClip.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import open_clip
import torch
from open_clip.tokenizer import tokenize
from torch import nn
import torch.nn.functional as F

# from .third_party import CLIPAD
from .ad_prompts import *
from .prompting import create_prompt_ensemble
from .utils import cosine_similarity, harmonic_aggregation

valid_backbones = ["ViT-B-16-plus-240"]
valid_pretrained_datasets = ["laion400m_e32"]


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]


def _convert_to_rgb(image):
    return image.convert("RGB")


class patch_scale:
    def __init__(self, image_size):
        self.h, self.w = image_size

    def make_mask(self, patch_size=16, kernel_size=16, stride_size=16):
        self.patch_size = patch_size
        self.patch_num_h = self.h // self.patch_size
        self.patch_num_w = self.w // self.patch_size
        ###################################################### patch_level
        self.kernel_size = kernel_size // patch_size
        self.stride_size = stride_size // patch_size
        self.idx_board = torch.arange(1, self.patch_num_h * self.patch_num_w + 1, dtype=torch.float32).reshape(
            (1, 1, self.patch_num_h, self.patch_num_w)
        )
        patchfy = torch.nn.functional.unfold(self.idx_board, kernel_size=self.kernel_size, stride=self.stride_size)
        return patchfy

class WinClipModel(nn.Module):
    def __init__(self, model_name="ViT-B-16-plus-240"):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained="laion400m_e31")
        # self.model.visual.output_tokens = True
        self.mask = patch_scale((240, 240))

        self.text_embeddings: torch.Tensor | None = None

    def multiscale(self):
        pass

    def encode_text(self, text):
        return self.model.encode_text(text)

    def encode_image(self, image, patch_size, mask=True):
        self.model.visual.output_tokens = True
        # TODO: Investigate if this is actually needed
        self.model.visual.final_ln_after_pool = True

        # register hook to retrieve feature map
        feature_map = {}

        def get_feature_map(name):
            def hook(model, input, output):
                feature_map[name] = input[0].detach()

            return hook

        self.model.visual.patch_dropout.register_forward_hook(get_feature_map("patch_dropout"))

        # get patch embeddings
        image_embeddings, patch_embeddings = self.model.encode_image(image)
        intermediate_tokens = feature_map["patch_dropout"]

        # get window embeddings
        # TODO: make masks in init (same for every batch)
        large_scale = self.mask.make_mask(kernel_size=48, patch_size=patch_size).squeeze().cuda()
        mid_scale = self.mask.make_mask(kernel_size=32, patch_size=patch_size).squeeze().cuda()

        large_scale_embeddings = self._get_window_embeddings(intermediate_tokens, large_scale).squeeze()
        mid_scale_embeddings = self._get_window_embeddings(intermediate_tokens, mid_scale).squeeze()

        return (
            large_scale_embeddings,
            mid_scale_embeddings,
            patch_embeddings.unsqueeze(2),
            image_embeddings,
            large_scale,
            mid_scale,
        )

    def _get_window_embeddings(self, x, mask_scale):
        x_select = []
        mask_scale = mask_scale.T
        mask_num, L = mask_scale.shape
        class_index = torch.zeros((mask_scale.shape[0], 1), dtype=torch.int32).to(mask_scale)
        mask_scale = torch.cat((class_index, mask_scale.int()), dim=1)

        for i in mask_scale:
            try:
                x_select.append(torch.index_select(x, 1, i.int()))
            except:
                print("i", i)
        x_scale = torch.cat(x_select)  #

        x_scale = self.model.visual.patch_dropout(x_scale)
        x_scale = self.model.visual.ln_pre(x_scale)
        # print("x_scale", x_scale.shape)
        x_scale = x_scale.permute(1, 0, 2)  # NLD -> LND
        x_scale = self.model.visual.transformer(x_scale)
        x_scale = x_scale.permute(1, 0, 2)  # LND -> NLD
        # print(x_scale.shape)
        if self.model.visual.attn_pool is not None:
            x_scale = self.model.visual.attn_pool(x_scale)
            x_scale = self.model.visual.ln_post(x_scale)
            pooled, tokens = self.model.visual._global_pool(x_scale)
        else:
            pooled, tokens = self.model.visual._global_pool(x_scale)
            pooled = self.model.visual.ln_post(pooled)

        if self.model.visual.proj is not None:
            pooled = pooled @ self.model.visual.proj

        # if self.model.visual.output_tokens:
        #     return pooled, tokens
        return pooled.reshape((mask_num, x.shape[0], 1, -1)).permute(1, 0, 2, 3)

    def forward(self, x):
        # x = torch.load("/home/djameln/WinCLIP-pytorch/image_wnclp.pt")
        (
            large_scale_embeddings,
            mid_scale_embeddings,
            patch_embeddings,
            image_embeddings,
            large_scale,
            mid_scale,
        ) = self.encode_image(x, patch_size=16, mask=True)

        # get anomaly scores
        image_scores = cosine_similarity(image_embeddings, self.text_embeddings)[..., -1]
        large_scale_sim = cosine_similarity(large_scale_embeddings, self.text_embeddings)[..., -1]
        mid_scale_sim = cosine_similarity(mid_scale_embeddings, self.text_embeddings)[..., -1]

        large_scale_scores = harmonic_aggregation(large_scale_sim, (15, 15), large_scale)
        mid_scale_scores = harmonic_aggregation(mid_scale_sim, (15, 15), mid_scale)

        multiscale_scores = 3 / (1 / large_scale_scores + 1 / mid_scale_scores + 1 / image_scores.view(-1, 1, 1))
        pixel_scores = F.interpolate(
            multiscale_scores.unsqueeze(1),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        return image_scores, pixel_scores

    def collect_text_embeddings(self, object: str, device: torch.device | None = None):
        # collect prompt ensemble
        normal_prompts, anomalous_prompts = create_prompt_ensemble(object)
        # tokenize
        normal_tokens = tokenize(normal_prompts)
        anomalous_tokens = tokenize(anomalous_prompts)
        # encode
        normal_embeddings = self.model.encode_text(normal_tokens)
        anomalous_embeddings = self.model.encode_text(anomalous_tokens)
        # average
        normal_embeddings = torch.mean(normal_embeddings, dim=0, keepdim=True)
        anomalous_embeddings = torch.mean(anomalous_embeddings, dim=0, keepdim=True)
        # normalize
        text_embeddings = torch.cat((normal_embeddings, anomalous_embeddings))
        # text_embeddings /= text_embeddings.norm(dim=1, keepdim=True)
        # move to device
        if device is not None:
            text_embeddings = text_embeddings.to(device)
        # store
        self.text_embeddings = text_embeddings.detach()


class WinClipAD(torch.nn.Module):
    def __init__(
        self,
        # output_size=(256, 256),
        backbone="ViT-B-16-plus-240",
        pretrained_dataset="laion400m_e32",
        scales=(2, 3),
        precision="fp32",
        # **kwargs
    ):
        """:param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        """
        super().__init__()

        # self.out_size_h, self.out_size_w = output_size
        self.precision = precision  # -40% GPU memory (2.8G->1.6G) with slight performance drop

        self.get_model(backbone, pretrained_dataset, scales)
        self.phrase_form = "{}"

        # version v1: no norm for each of linguistic embedding
        # version v1:    norm for each of linguistic embedding
        self.version = "V1"  # V1:
        # visual textual, textual_visual
        self.fusion_version = "textual_visual"

        print(f"fusion version: {self.fusion_version}")

    def get_model(self, backbone, pretrained_dataset, scales):
        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model = CLIPAD.create_model(
            model_name=backbone,
            pretrained=pretrained_dataset,
            scales=scales,
            precision=self.precision,
        )
        tokenizer = get_tokenizer(backbone)
        model.eval()

        self.masks = model.visual.masks
        self.scale_begin_indx = model.visual.scale_begin_indx
        self.model = model
        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.grid_size = model.visual.grid_size
        self.visual_gallery = None
        print("self.grid_size", self.grid_size)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):
        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image)
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        text_features = self.model.encode_text(text)
        return text_features

    def build_text_feature_gallery(self, category: str):
        normal_phrases = []
        abnormal_phrases = []

        # some categories can be renamed to generate better embedding
        # if category == 'grid':
        #    category  = 'chain-link fence'
        # if category == 'toothbrush':
        #    category = 'brush' #'brush' #
        for template_prompt in template_level_prompts:
            # normal prompts
            for normal_prompt in state_level_normal_prompts:
                phrase = template_prompt.format(normal_prompt.format(category))
                normal_phrases += [phrase]

            # abnormal prompts
            for abnormal_prompt in state_level_abnormal_prompts:
                phrase = template_prompt.format(abnormal_prompt.format(category))
                abnormal_phrases += [phrase]

        normal_phrases = self.tokenizer(normal_phrases)
        abnormal_phrases = self.tokenizer(abnormal_phrases)

        if self.version == "V1":
            normal_text_features = self.encode_text(normal_phrases)
            abnormal_text_features = self.encode_text(abnormal_phrases)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_phrases.size()[0]):
                normal_text_feature = self.encode_text(normal_phrases[phrase_id].unsqueeze(0))
                normal_text_feature = normal_text_feature / normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()
            abnormal_text_features = []
            for phrase_id in range(abnormal_phrases.size()[0]):
                abnormal_text_feature = self.encode_text(abnormal_phrases[phrase_id].unsqueeze(0))
                abnormal_text_feature = abnormal_text_feature / abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        self.avr_normal_text_features = avr_normal_text_features
        self.avr_abnormal_text_features = avr_abnormal_text_features
        self.text_features = torch.cat([self.avr_normal_text_features, self.avr_abnormal_text_features], dim=0)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def build_image_feature_gallery(self, normal_images):
        self.visual_gallery = []
        visual_features = self.encode_image(normal_images)

        for scale_index in range(len(self.scale_begin_indx)):
            if scale_index == len(self.scale_begin_indx) - 1:
                scale_features = visual_features[self.scale_begin_indx[scale_index] :]
            else:
                scale_features = visual_features[
                    self.scale_begin_indx[scale_index] : self.scale_begin_indx[scale_index + 1]
                ]

            self.visual_gallery += [torch.cat(scale_features, dim=0)]

    def calculate_textual_anomaly_score(self, visual_features):
        device = visual_features[0].device
        N = visual_features[0].shape[0]
        scale_anomaly_scores = []
        token_anomaly_scores = torch.zeros((N, self.grid_size[0] * self.grid_size[1]), device=device)
        token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]), device=device)
        for indx, (features, mask) in enumerate(zip(visual_features, self.masks)):
            text_features = self.text_features.to(features.device)
            normality_and_abnormality_score = (100.0 * features @ text_features.T).softmax(dim=-1)
            normality_score = normality_and_abnormality_score[:, 0]
            normality_and_abnormality_score[:, 1]
            normality_score = normality_score

            mask = mask.reshape(-1)
            cur_token_anomaly_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])).to(normality_score.device)
            if self.precision == "fp16":
                cur_token_anomaly_score = cur_token_anomaly_score.half()
            cur_token_anomaly_score[:, mask] = (1.0 / normality_score).unsqueeze(1)
            # cur_token_anomaly_score[:, mask] = (1. - normality_score).unsqueeze(1)
            cur_token_weight = torch.zeros((N, self.grid_size[0] * self.grid_size[1])).to(
                cur_token_anomaly_score.device,
            )
            cur_token_weight[:, mask] = 1.0

            if indx in self.scale_begin_indx[1:]:
                # deal with the first two scales
                token_anomaly_scores = token_anomaly_scores / token_weights
                scale_anomaly_scores.append(token_anomaly_scores)

                # another scale, calculate from scratch
                token_anomaly_scores = torch.zeros((N, self.grid_size[0] * self.grid_size[1]), device=device)
                token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]), device=device)

            token_weights += cur_token_weight
            token_anomaly_scores += cur_token_anomaly_score

        # deal with the last one
        token_anomaly_scores = token_anomaly_scores / token_weights
        scale_anomaly_scores.append(token_anomaly_scores)

        scale_anomaly_scores = torch.stack(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = torch.mean(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = 1.0 - 1.0 / scale_anomaly_scores

        anomaly_map = scale_anomaly_scores.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)
        return anomaly_map

    def calculate_visual_anomaly_score(self, visual_features):
        N = visual_features[0].shape[0]
        device = visual_features[0].device
        scale_anomaly_scores = []
        token_anomaly_scores = torch.zeros((N, self.grid_size[0] * self.grid_size[1])).to(device)
        token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1])).to(device)

        cur_scale_indx = 0
        cur_visual_gallery = self.visual_gallery[cur_scale_indx]

        for indx, (features, mask) in enumerate(zip(visual_features, self.masks)):
            normality_score = 0.5 * (1 - (features @ cur_visual_gallery.T).max(dim=1)[0])
            normality_score = normality_score

            mask = mask.reshape(-1).to(device)
            cur_token_anomaly_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])).to(device)
            if self.precision == "fp16":
                cur_token_anomaly_score = cur_token_anomaly_score.half()
            cur_token_anomaly_score[:, mask] = normality_score.unsqueeze(1)
            # cur_token_anomaly_score[:, mask] = (1. - normality_score).unsqueeze(1)
            cur_token_weight = torch.zeros((N, self.grid_size[0] * self.grid_size[1])).to(device)
            cur_token_weight[:, mask] = 1.0

            if indx in self.scale_begin_indx[1:]:
                cur_scale_indx += 1
                cur_visual_gallery = self.visual_gallery[cur_scale_indx]
                # deal with the first two scales
                token_anomaly_scores = token_anomaly_scores / token_weights
                scale_anomaly_scores.append(token_anomaly_scores)

                # another scale, calculate from scratch
                token_anomaly_scores = torch.zeros((N, self.grid_size[0] * self.grid_size[1])).to(device)
                token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1])).to(device)

            token_weights += cur_token_weight
            token_anomaly_scores += cur_token_anomaly_score

        # deal with the last one
        token_anomaly_scores = token_anomaly_scores / token_weights
        scale_anomaly_scores.append(token_anomaly_scores)

        scale_anomaly_scores = torch.stack(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = torch.mean(scale_anomaly_scores, dim=0)

        anomaly_map = scale_anomaly_scores.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)
        return anomaly_map

    def forward(self, images):
        visual_features = self.encode_image(images)
        textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features)
        if self.visual_gallery is not None:
            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
        else:
            visual_anomaly_map = textual_anomaly_map

        if self.fusion_version == "visual":
            anomaly_map = visual_anomaly_map
        elif self.fusion_version == "textual":
            anomaly_map = textual_anomaly_map
        else:
            anomaly_map = 1.0 / (1.0 / textual_anomaly_map + 1.0 / visual_anomaly_map)

        # anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)
        am_np = anomaly_map.squeeze(1).cpu().numpy()

        am_np_list = []

        for i in range(am_np.shape[0]):
            # am_np[i] = gaussian_filter(am_np[i], sigma=4)
            am_np_list.append(am_np[i])

        return am_np_list

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()
